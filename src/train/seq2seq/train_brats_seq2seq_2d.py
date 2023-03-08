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
sys.path.append('./src/')

from utils import poly_lr, Recorder, Plotter, save_grid_images, torch_PSNR
from losses import PerceptualLoss
from dataloader.brats import Dataset_brats
from models.seq2seq.seq2seq import Generator


def train(args, net, device):
    train_data = Dataset_brats(args, mode='train')
    valid_data = Dataset_brats(args, mode='valid')

    n_train = len(train_data)#len(dataset) - n_val
    n_valid = len(valid_data)#len(dataset) - n_val

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
    lambda_cyc = args['train']['lambda_cyc']
    
    logging.info(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {1}
        Learning rate:   {lr}
        Training size:   {n_train}
        Valid size:      {n_valid}
        Device:          {device.type}
    ''')
    
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lambda epoch: poly_lr(epoch, epochs, lr, min_lr=1e-6)/lr)
    perceptual = PerceptualLoss().to(device=device)
        
    recorder = Recorder(['train_loss', 'psnr'])
    plotter = Plotter(dir_visualize, keys1=['train_loss'], keys2=['psnr'])
    
    with open(os.path.join(dir_checkpoint, 'log.csv'), 'w') as f:
        f.write('epoch,train_loss,psnr\n')

    total_step = 0
    best_psnr = 0
    nan_times = 0
    for epoch in range(epochs):
        if epoch!=0:
            scheduler.step()
        net.train()
        train_losses = []
        with tqdm(total=n_train*rep_step, desc=f'Epoch {epoch + 1}/{epochs}', unit='img') as pbar:
            #loader_T2 = iter(train_loader_T2)
            for batch in train_loader:
                img_t1 = batch['t1']
                img_t1ce = batch['t1ce']
                img_t2 = batch['t2']
                img_flair = batch['flair']
                #segs = batch['segs']
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
                    source_seq = flags[0]
                    random.shuffle(flags)
                    target_seq = flags[0]

                    source_img = inputs[source_seq]
                    target_img = inputs[target_seq]

                    source_code = torch.from_numpy(np.array([1 if i==source_seq else 0 for i in range(c_s)])).reshape((1,c_s)).to(device=device, dtype=torch.float32)
                    target_code = torch.from_numpy(np.array([1 if i==target_seq else 0 for i in range(c_s)])).reshape((1,c_s)).to(device=device, dtype=torch.float32)
                    
                    output_source = net(source_img, source_code, n_outseq=source_img.shape[1])
                    output_target = net(source_img, target_code, n_outseq=target_img.shape[1])
                    output_cyc = net(output_target, source_code, n_outseq=source_img.shape[1])
                    
                    loss_rec = nn.SmoothL1Loss()(output_target, target_img) + nn.SmoothL1Loss()(output_source, source_img)
                    loss_cyc = nn.SmoothL1Loss()(output_cyc, source_img)
                    loss_per = perceptual(output_target[0].reshape(-1,1,nw,nh)/2+0.5, target_img[0].reshape(-1,1,nw,nh)/2+0.5) + \
                        perceptual(output_source[0].reshape(-1,1,nw,nh)/2+0.5, source_img[0].reshape(-1,1,nw,nh)/2+0.5)
                    
                    loss = lambda_rec*loss_rec + lambda_per*loss_per + lambda_cyc*loss_cyc
                    
                    optimizer.zero_grad()
                    loss.backward()
                    #torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0, norm_type=2)
                    optimizer.step()
                    

                    train_losses.append(loss_rec.item())
                    pbar.set_postfix(**{'rec': loss_rec.item(), 'per': loss_per.item(), 'cyc': loss_cyc.item()})
                    pbar.update(1)

                    if (total_step % args['train']['vis_steps']) == 0:
                        with torch.no_grad():
                            output1 = net(source_img, torch.from_numpy(np.array([1 if i==0 else 0 for i in range(c_s)])).reshape((1,c_s)).to(device=device, dtype=torch.float32), n_outseq=1)
                            output2 = net(source_img, torch.from_numpy(np.array([1 if i==1 else 0 for i in range(c_s)])).reshape((1,c_s)).to(device=device, dtype=torch.float32), n_outseq=1)
                            output3 = net(source_img, torch.from_numpy(np.array([1 if i==2 else 0 for i in range(c_s)])).reshape((1,c_s)).to(device=device, dtype=torch.float32), n_outseq=1)
                            output4 = net(source_img, torch.from_numpy(np.array([1 if i==3 else 0 for i in range(c_s)])).reshape((1,c_s)).to(device=device, dtype=torch.float32), n_outseq=1)
                            
                        view_list = [
                                [source_img[:,0:1,(c_in-1)//2], output_source[:,0:1,(c_in-1)//2], target_img[:,0:1,(c_in-1)//2], output_target[:,0:1,(c_in-1)//2]],
                                [inputs[0][:,0:1,(c_in-1)//2], inputs[1][:,0:1,(c_in-1)//2], inputs[2][:,0:1,(c_in-1)//2], inputs[3][:,0:1,(c_in-1)//2]],
                                [output1[:,0:1,(c_in-1)//2], output2[:,0:1,(c_in-1)//2], output3[:,0:1,(c_in-1)//2], output4[:,0:1,(c_in-1)//2]],
                            ]
                        
                        save_grid_images(view_list, os.path.join(dir_visualize, '{:03d}.jpg'.format(epoch+1)), clip_range=(-1,1), normalize=True)
                        torch.cuda.empty_cache()
                    
                    if (total_step % args['train']['ckpt_steps']) == 0:
                        torch.save(net.state_dict(), os.path.join(dir_checkpoint, 'ckpt_tmp.pth'))

                    total_step += 1
                    if total_step > args['train']['total_steps']:
                        #torch.save(net.state_dict(), os.path.join(dir_checkpoint, 'ckpt_latest.pth'))
                        return
                #break
        
        net.eval()
        valid_psnrs = []
        torch.cuda.empty_cache()
        with torch.no_grad():
            for batch in valid_loader:
                img_t1 = batch['t1']
                img_t1ce = batch['t1ce']
                img_t2 = batch['t2']
                img_flair = batch['flair']
                #segs = batch['segs']
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

                for src in flags:
                    source_img = inputs[src]
                    for tgt in flags:
                        target_img = inputs[tgt]
                        target_code = torch.from_numpy(np.array([1 if i==tgt else 0 for i in range(c_s)])).reshape((1,c_s)).to(device=device, dtype=torch.float32)
                        output_target = net(source_img, target_code, n_outseq=target_img.shape[1])
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
            with open(os.path.join(dir_checkpoint, 'log.csv'), 'a+') as f:
                f.write('{},{},{}\n'.format(epoch+1, train_losses, valid_psnrs))
        #torch.save(net.state_dict(), os.path.join(dir_checkpoint, 'ckpt_latest.pth'))
        torch.cuda.empty_cache()

def get_args():
    parser = argparse.ArgumentParser(description='Train the VGG on images and target label',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-c', '--config', dest='config', type=str, default='config/seq2seq_breast_3d.yaml',
                        help='config file')
    parser.add_argument('-l', '--load', dest='load', type=str, default=None,
                        help='Load model from a .pth file')
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

    net = Generator(config)
    net.to(device=device)

    if args.load:
        load_dict = torch.load(args.load, map_location=device)
        net.load_state_dict(load_dict)
        print('[*] Load model from', args.load)
    
    try:
        train(
            config,
            net=net,
            device=device,
        )
    except KeyboardInterrupt:
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)