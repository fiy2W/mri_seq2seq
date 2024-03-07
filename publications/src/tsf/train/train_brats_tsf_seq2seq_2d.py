import argparse
import os
import sys
import logging
import yaml
import random
from tqdm import tqdm
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import sys
sys.path.append('./publications/')

from src.seq2seq.utils import poly_lr, Recorder, Plotter, save_grid_images, torch_PSNR
from src.seq2seq.losses import PerceptualLoss
from src.seq2seq.dataloader.brats import Dataset_brats
from src.seq2seq.models.seq2seq import Generator
from src.tsf.models.tsf_seq2seq import TSF_seq2seq


def train(args, net, seq2seq, device):
    train_data = Dataset_brats(args, mode='train')
    valid_data = Dataset_brats(args, mode='valid')

    n_train = len(train_data)
    n_valid = len(valid_data)

    train_loader = DataLoader(train_data, batch_size=1, shuffle=True, num_workers=4, pin_memory=True)
    valid_loader = DataLoader(valid_data, batch_size=1, shuffle=False, num_workers=2, pin_memory=True)

    c_in = args['seq2seq']['c_in']
    c_s = args['seq2seq']['c_s']
    epochs = args['train']['epochs']
    lr = np.float32(args['train']['lr'])
    dir_visualize = args['train']['vis']
    dir_checkpoint = args['train']['ckpt']
    rep_step = args['train']['rep_step']
    crop_size = args['train']['crop_size']
    valid_size = args['train']['valid_size']
    lambda_rec = args['train']['lambda_rec']
    lambda_per = args['train']['lambda_per']
    
    logging.info(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {1}
        Learning rate:   {lr}
        Training size:   {n_train}
        Valid size:      {n_valid}
        Device:          {device.type}
    ''')
    seq2seq_param = list(seq2seq.decoder.parameters())+list(seq2seq.dec_convlstm.parameters())+list(seq2seq.style_fc.parameters())
    optimizer = torch.optim.Adam(list(net.parameters())+seq2seq_param, lr=lr)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lambda epoch: poly_lr(epoch, epochs, lr, min_lr=1e-6)/lr)
    perceptual = PerceptualLoss().to(device=device)
        
    recorder = Recorder(['train_loss', 'psnr'])
    plotter = Plotter(dir_visualize, keys1=['train_loss'], keys2=['psnr'])
    
    with open(os.path.join(dir_checkpoint, 'log.csv'), 'w') as f:
        f.write('epoch,train_loss,psnr\n')

    total_step = 0
    best_psnr = 0
    for epoch in range(epochs):
        if epoch!=0:
            scheduler.step()
        net.train()
        seq2seq.train()
        train_losses = []
        with tqdm(total=n_train*rep_step, desc=f'Epoch {epoch + 1}/{epochs}', unit='img') as pbar:
            for batch in train_loader:
                img_t1 = batch['t1']
                img_t1ce = batch['t1ce']
                img_t2 = batch['t2']
                img_flair = batch['flair']
                flags = [i[0] for i in batch['flag']]
                path = batch['path'][0][0]
                if len(flags)==0:
                    raise Exception('No available sequence in {}!'.format(path))

                d = img_t1.shape[3]
                img_t1 = torch.cat([img_t1[:,:,:,i:d-c_in+i+1] for i in range(c_in)], dim=2)[0].permute(2,0,1,3,4)
                img_t1ce = torch.cat([img_t1ce[:,:,:,i:d-c_in+i+1] for i in range(c_in)], dim=2)[0].permute(2,0,1,3,4)
                img_t2 = torch.cat([img_t2[:,:,:,i:d-c_in+i+1] for i in range(c_in)], dim=2)[0].permute(2,0,1,3,4)
                img_flair = torch.cat([img_flair[:,:,:,i:d-c_in+i+1] for i in range(c_in)], dim=2)[0].permute(2,0,1,3,4)

                for _ in range(rep_step):
                    nd, nw, nh = crop_size

                    d,t,c,w,h = img_t1.shape
                    rd = random.randint(0, d-nd-1) if d>nd else 0
                    rw = random.randint(0, w-nw-1) if w>nw else 0
                    rh = random.randint(0, h-nh-1) if h>nh else 0

                    inputs = [
                        img_t1[rd:rd+nd,:,:,rw:rw+nw,rh:rh+nh].to(device=device, dtype=torch.float32),
                        img_t1ce[rd:rd+nd,:,:,rw:rw+nw,rh:rh+nh].to(device=device, dtype=torch.float32),
                        img_t2[rd:rd+nd,:,:,rw:rw+nw,rh:rh+nh].to(device=device, dtype=torch.float32),
                        img_flair[rd:rd+nd,:,:,rw:rw+nw,rh:rh+nh].to(device=device, dtype=torch.float32),
                    ]

                    random.shuffle(flags)
                    source_seqs = flags[:random.randint(1, len(flags)) if len(flags) > 1 else 1]
                    target_request_seq = [i for i in flags if i not in source_seqs]
                    if len(target_request_seq)==0:
                        random.shuffle(flags)
                        target_seq = flags[0]
                    else:
                        random.shuffle(target_request_seq)
                        target_seq = target_request_seq[0]
                        

                    source_imgs = [inputs[source_seq] for source_seq in source_seqs]
                    target_img = inputs[target_seq]

                    if torch.abs(torch.mean(source_imgs[0]) + 1)<1e-8:
                        continue

                    source_code = torch.from_numpy(np.array([1 if i==source_seqs[0] else 0 for i in range(c_s)])).reshape((1,c_s)).to(device=device, dtype=torch.float32)
                    target_code = torch.from_numpy(np.array([1 if i==target_seq else 0 for i in range(c_s)])).reshape((1,c_s)).to(device=device, dtype=torch.float32)
                    
                    skip_attn = random.randint(0, 1)>0.5

                    output_source = net(seq2seq, source_imgs, source_seqs, source_code, n_outseq=source_imgs[0].shape[1], task_attn=True, skip_attn=skip_attn)
                    output_target = net(seq2seq, source_imgs, source_seqs, target_code, n_outseq=target_img.shape[1], task_attn=True, skip_attn=True)
                    output_target_A = net(seq2seq, source_imgs, source_seqs, target_code, n_outseq=target_img.shape[1], task_attn=True, skip_attn=False)
                    
                    loss_rec = nn.SmoothL1Loss()(output_target, target_img) + nn.SmoothL1Loss()(output_target_A, target_img) + nn.SmoothL1Loss()(output_source, source_imgs[0])
                    loss_per = perceptual(output_target[0,:,:,:].reshape(-1,1,nw,nh)/2+0.5, target_img[0,:,:,:].reshape(-1,1,nw,nh)/2+0.5) + \
                        perceptual(output_target_A[0,:,:,:].reshape(-1,1,nw,nh)/2+0.5, target_img[0,:,:,:].reshape(-1,1,nw,nh)/2+0.5) + \
                        perceptual(output_source[0,:,:,:].reshape(-1,1,nw,nh)/2+0.5, source_imgs[0][0,:,:,:].reshape(-1,1,nw,nh)/2+0.5)
                    
                    loss = lambda_rec*loss_rec + lambda_per*loss_per

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    train_losses.append(loss_rec.item())
                    pbar.set_postfix(**{'rec': loss_rec.item(), 'per': loss_per.item()})
                    pbar.update(1)

                    if (total_step % args['train']['vis_steps']) == 0:
                        with torch.no_grad():
                            output1 = net(seq2seq, source_imgs, source_seqs, torch.from_numpy(np.array([1 if i==0 else 0 for i in range(c_s)])).reshape((1,c_s)).to(device=device, dtype=torch.float32), n_outseq=1)
                            output2 = net(seq2seq, source_imgs, source_seqs, torch.from_numpy(np.array([1 if i==1 else 0 for i in range(c_s)])).reshape((1,c_s)).to(device=device, dtype=torch.float32), n_outseq=1)
                            output3 = net(seq2seq, source_imgs, source_seqs, torch.from_numpy(np.array([1 if i==2 else 0 for i in range(c_s)])).reshape((1,c_s)).to(device=device, dtype=torch.float32), n_outseq=1)
                            output4 = net(seq2seq, source_imgs, source_seqs, torch.from_numpy(np.array([1 if i==3 else 0 for i in range(c_s)])).reshape((1,c_s)).to(device=device, dtype=torch.float32), n_outseq=1)
                            
                            view_list = [
                                [source_imgs[0][:,0:1,(c_in-1)//2], output_source[:,0:1,(c_in-1)//2], target_img[:,0:1,(c_in-1)//2], output_target[:,0:1,(c_in-1)//2]],
                                [inputs[0][:,0:1,(c_in-1)//2], inputs[1][:,0:1,(c_in-1)//2], inputs[2][:,0:1,(c_in-1)//2], inputs[3][:,0:1,(c_in-1)//2]],
                                [output1[:,0:1,(c_in-1)//2], output2[:,0:1,(c_in-1)//2], output3[:,0:1,(c_in-1)//2], output4[:,0:1,(c_in-1)//2]],
                            ]

                            output1 = net(seq2seq, source_imgs, source_seqs, torch.from_numpy(np.array([1 if i==0 else 0 for i in range(c_s)])).reshape((1,c_s)).to(device=device, dtype=torch.float32), n_outseq=1, skip_attn=True)
                            output2 = net(seq2seq, source_imgs, source_seqs, torch.from_numpy(np.array([1 if i==1 else 0 for i in range(c_s)])).reshape((1,c_s)).to(device=device, dtype=torch.float32), n_outseq=1, skip_attn=True)
                            output3 = net(seq2seq, source_imgs, source_seqs, torch.from_numpy(np.array([1 if i==2 else 0 for i in range(c_s)])).reshape((1,c_s)).to(device=device, dtype=torch.float32), n_outseq=1, skip_attn=True)
                            output4 = net(seq2seq, source_imgs, source_seqs, torch.from_numpy(np.array([1 if i==3 else 0 for i in range(c_s)])).reshape((1,c_s)).to(device=device, dtype=torch.float32), n_outseq=1, skip_attn=True)

                            view_list.append(
                                [output1[:,0:1,(c_in-1)//2], output2[:,0:1,(c_in-1)//2], output3[:,0:1,(c_in-1)//2], output4[:,0:1,(c_in-1)//2]],
                            )
                        
                        
                        save_grid_images(view_list, os.path.join(dir_visualize, '{:03d}.jpg'.format(epoch+1)), clip_range=(-1,1), normalize=True)
                        torch.cuda.empty_cache()
                    
                    if (total_step % args['train']['ckpt_steps']) == 0:
                        torch.save(net.state_dict(), os.path.join(dir_checkpoint, 'ckpt_tmp.pth'))
                        torch.save(seq2seq.state_dict(), os.path.join(dir_checkpoint, 'ckpt_seq2seq_tmp.pth'))

                    total_step += 1
                    if total_step > args['train']['total_steps']:
                        #torch.save(net.state_dict(), os.path.join(dir_checkpoint, 'ckpt_latest.pth'))
                        return
                #break
        
        net.eval()
        seq2seq.eval()
        valid_psnrs = []
        torch.cuda.empty_cache()
        with torch.no_grad():
            for batch in valid_loader:
                img_t1 = batch['t1']
                img_t1ce = batch['t1ce']
                img_t2 = batch['t2']
                img_flair = batch['flair']
                flags = [i[0] for i in batch['flag']]
                path = batch['path'][0][0]
                if len(flags)==0:
                    raise Exception('No available sequence in {}!'.format(path))

                d = img_t1.shape[3]
                img_t1 = torch.cat([img_t1[:,:,:,i:d-c_in+i+1] for i in range(c_in)], dim=2)[0].permute(2,0,1,3,4)
                img_t1ce = torch.cat([img_t1ce[:,:,:,i:d-c_in+i+1] for i in range(c_in)], dim=2)[0].permute(2,0,1,3,4)
                img_t2 = torch.cat([img_t2[:,:,:,i:d-c_in+i+1] for i in range(c_in)], dim=2)[0].permute(2,0,1,3,4)
                img_flair = torch.cat([img_flair[:,:,:,i:d-c_in+i+1] for i in range(c_in)], dim=2)[0].permute(2,0,1,3,4)

                nd, nw, nh = valid_size

                d,t,c,w,h = img_t1.shape
                rd = (d-nd)//2 if d>nd else 0 #random.randint(0, d-nd-1) if d>nd else 0
                rw = (w-nw)//2 if w>nw else 0 #random.randint(0, w-nw-1) if w>nw else 0
                rh = (h-nh)//2 if h>nh else 0 #random.randint(0, h-nh-1) if h>nh else 0

                inputs = [
                    img_t1[rd:rd+nd,:,:,rw:rw+nw,rh:rh+nh].to(device=device, dtype=torch.float32),
                    img_t1ce[rd:rd+nd,:,:,rw:rw+nw,rh:rh+nh].to(device=device, dtype=torch.float32),
                    img_t2[rd:rd+nd,:,:,rw:rw+nw,rh:rh+nh].to(device=device, dtype=torch.float32),
                    img_flair[rd:rd+nd,:,:,rw:rw+nw,rh:rh+nh].to(device=device, dtype=torch.float32),
                ]

                tgt_flags = [tgt for tgt in range(4) if tgt not in flags] if args['data']['nomiss'] else flags
                for tgt in tgt_flags:
                    source_imgs = [inputs[src] for src in flags]
                    target_img = inputs[tgt]
                    target_code = torch.from_numpy(np.array([1 if i==tgt else 0 for i in range(c_s)])).reshape((1,c_s)).to(device=device, dtype=torch.float32)
                    output_target = net(seq2seq, source_imgs, flags, target_code, n_outseq=target_img.shape[1])
                    psnr = torch_PSNR(output_target, target_img, data_range=2.).item()
                    valid_psnrs.append(psnr)
                #break
        
        valid_psnrs = np.mean(valid_psnrs)
        train_losses = np.mean(train_losses)
        recorder.update({'train_loss': train_losses, 'psnr': valid_psnrs})
        plotter.send(recorder.call())
        if best_psnr<valid_psnrs:
            best_psnr = valid_psnrs
            torch.save(net.state_dict(), os.path.join(dir_checkpoint, 'ckpt_best.pth'))
            torch.save(seq2seq.state_dict(), os.path.join(dir_checkpoint, 'ckpt_seq2seq_best.pth'))
            with open(os.path.join(dir_checkpoint, 'log.csv'), 'a+') as f:
                f.write('{},{},{}\n'.format(epoch+1, train_losses, valid_psnrs))
        #torch.save(net.state_dict(), os.path.join(dir_checkpoint, 'ckpt_latest.pth'))
        torch.cuda.empty_cache()

def get_args():
    parser = argparse.ArgumentParser(description='Train TSF-seq2seq model',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-c', '--config', dest='config', type=str, default='config/tsf_seq2seq_brats_2d.yaml',
                        help='config file')
    parser.add_argument('-l', '--load', dest='load', type=str, default=None,
                        help='Load model from a .pth file')
    parser.add_argument('-m', '--seq2seq', dest='seq2seq', type=str, default=None,
                        help='Load seq2seq model from a .pth file')
    parser.add_argument('-d', '--device', dest='device', type=str, default='cpu',
                        help='cuda or cpu')
    
    return parser.parse_args()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    args = get_args()
    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    dir_checkpoint = config['train']['ckpt']
    if not os.path.exists(dir_checkpoint):
        os.makedirs(dir_checkpoint)
    
    dir_visualize = config['train']['vis']
    if not os.path.exists(dir_visualize):
        os.makedirs(dir_visualize)

    device = torch.device(args.device)
    logging.info(f'Using device {device}')

    seq2seq = Generator(config)
    seq2seq.to(device=device)
    pretrained_weight = config['seq2seq']['pretrain']
    load_dict = torch.load(pretrained_weight, map_location=device)
    seq2seq.load_state_dict(load_dict)

    net = TSF_seq2seq(config)
    net.to(device=device)

    if args.load:
        load_dict = torch.load(args.load, map_location=device)
        net.load_state_dict(load_dict)
        print('[*] Load model from', args.load)
    
    if args.seq2seq:
        load_dict = torch.load(args.seq2seq, map_location=device)
        seq2seq.load_state_dict(load_dict)
        print('[*] Load seq2seq model from', args.seq2seq)
        
    try:
        train(
            config,
            net=net,
            seq2seq=seq2seq,
            device=device,
        )
    except KeyboardInterrupt:
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)