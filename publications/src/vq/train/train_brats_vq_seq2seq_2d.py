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
import torchvision

import sys
sys.path.append('./publications/')

from src.seq2seq.utils import poly_lr, Recorder, Plotter, save_grid_images, torch_PSNR
from src.seq2seq.losses import PerceptualLoss
from src.vq.dataloader.brats import Dataset_brats
from src.vq.dataloader.augmentation import random_aug
from src.vq.models.vqseq2seq import Generator
from src.vq.loss import SSIMLoss


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
    lambda_ssim = args['train']['lambda_ssim']
    lambda_per = args['train']['lambda_per']
    lambda_con = args['train']['lambda_con']
    lambda_vq = args['train']['lambda_vq']
    pretrain_epochs_contrast = args['train']['pretrain_epochs_contrast']
    pretrain_epochs_rand = args['train']['pretrain_epochs_rand']
    
    logging.info(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {1}
        Learning rate:   {lr}
        Training size:   {n_train}
        Valid size:      {n_valid}
        Device:          {device.type}
    ''')
    
    optimizer = torch.optim.AdamW(net.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.05)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lambda epoch: poly_lr(epoch, epochs, lr, min_lr=1e-6)/lr)
    perceptual = PerceptualLoss().to(device=device)
    recorder = Recorder(['rec', 'con', 'vq', 'psnr'])
    plotter = Plotter(dir_visualize, keys2=['psnr'], keys1=['rec', 'con', 'vq'])
    
    with open(os.path.join(dir_checkpoint, 'log.csv'), 'w') as f:
        f.write('epoch,rec,con,vq,psnr\n')

    total_step = 0
    best_psnr = 0
    for epoch in range(epochs):
        if epoch!=0:
            scheduler.step()
        net.train()
        train_metrics = {
            'total': [], 'rec': [], 'con': [], 'vq': [],
        }
        print('Epoch {}, learning rate={}'.format(epoch, optimizer.param_groups[0]['lr']))

        with tqdm(total=n_train*rep_step, desc=f'Epoch {epoch + 1}/{epochs}', unit='img') as pbar:
            for batch in train_loader:
                img_t1 = batch['t1']
                img_t1ce = batch['t1ce']
                img_t2 = batch['t2']
                img_flair = batch['flair']
                flags = [i[0] for i in batch['flag']]

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
                        img_t1[rd:rd+nd,:,0,rw:rw+nw,rh:rh+nh].to(device=device, dtype=torch.float32),
                        img_t1ce[rd:rd+nd,:,0,rw:rw+nw,rh:rh+nh].to(device=device, dtype=torch.float32),
                        img_t2[rd:rd+nd,:,0,rw:rw+nw,rh:rh+nh].to(device=device, dtype=torch.float32),
                        img_flair[rd:rd+nd,:,0,rw:rw+nw,rh:rh+nh].to(device=device, dtype=torch.float32),
                    ]

                    source_seq = random.choice(flags)
                    target_seq = [fi for fi in flags if fi!=source_seq][0]
                    random_seq = random.choice([0,1,2,3])

                    source_img = inputs[source_seq]
                    target_img = inputs[target_seq]

                    source_code = torch.from_numpy(np.array([1 if i==source_seq else 0 for i in range(c_s)])).reshape((1,c_s)).to(device=device, dtype=torch.float32)
                    target_code = torch.from_numpy(np.array([1 if i==target_seq else 0 for i in range(c_s)])).reshape((1,c_s)).to(device=device, dtype=torch.float32)
                    random_code = torch.from_numpy(np.array([1 if i==random_seq else 0 for i in range(c_s)])).reshape((1,c_s)).to(device=device, dtype=torch.float32)
                    
                    source_img_aug = random_aug(source_img/2+0.5)*2-1
                    target_img_aug = random_aug(target_img/2+0.5)*2-1

                    if torch.abs(torch.mean(source_img_aug) + 1)<1e-8 or torch.abs(torch.mean(target_img_aug) + 1)<1e-8 or torch.std(source_img_aug)<1e-8 or torch.std(target_img_aug)<1e-8:
                        continue

                    if epoch>pretrain_epochs_rand:
                        random_code = torch.rand_like(random_code) if random.random()>0.5 else random_code
                        
                        if random.random()>0.5:
                            with torch.no_grad():
                                source_img_aug = net.forward_single(source_img_aug, random_code)[0].detach()

                        rec_src2tgt, rec_tgt2src, rec_src2src, rec_tgt2tgt, rec_rand2src, rec_rand2tgt, c_src, c_tgt, loss_vq = net(source_img_aug, target_img_aug, source_code, target_code, z_sample=True)
                    else:
                        rec_src2tgt, rec_tgt2src, rec_src2src, rec_tgt2tgt, c_src, c_tgt, loss_vq = net(source_img_aug, target_img_aug, source_code, target_code)
                        rec_rand2src = rec_src2src
                        rec_rand2tgt = rec_tgt2tgt

                    data_range = torch.clamp(torch.max(target_img.max().unsqueeze(0),source_img.max().unsqueeze(0)), min=2)
                    loss_rec = nn.L1Loss()(rec_src2src, source_img) + nn.L1Loss()(rec_src2tgt, target_img) + \
                        nn.L1Loss()(rec_tgt2src, source_img) + nn.L1Loss()(rec_tgt2tgt, target_img) + \
                        nn.L1Loss()(rec_rand2src, source_img) + nn.L1Loss()(rec_rand2tgt, target_img)
                    loss_ssim = SSIMLoss()(rec_src2src, source_img, data_range) + \
                        SSIMLoss()(rec_src2tgt, target_img, data_range) + \
                        SSIMLoss()(rec_tgt2src, source_img, data_range) + \
                        SSIMLoss()(rec_tgt2tgt, target_img, data_range) + \
                        SSIMLoss()(rec_rand2src, source_img, data_range) + \
                        SSIMLoss()(rec_rand2tgt, target_img, data_range)
                    loss_per = perceptual(rec_src2src/2+0.5, source_img/2+0.5) + perceptual(rec_src2tgt/2+0.5, target_img/2+0.5) + \
                        perceptual(rec_tgt2src/2+0.5, source_img/2+0.5) + perceptual(rec_tgt2tgt/2+0.5, target_img/2+0.5) + \
                        perceptual(rec_rand2src/2+0.5, source_img/2+0.5) + perceptual(rec_rand2tgt/2+0.5, target_img/2+0.5)
                    
                    loss_con_mse, loss_con_contrast = net.con_loss(c_src, c_tgt, source_img, target_img, downsample=4)
                    
                    if epoch>pretrain_epochs_contrast:
                        loss_con = loss_con_mse + loss_con_contrast
                    else:
                        loss_con = loss_con_mse
                    
                    loss = lambda_rec*loss_rec + lambda_ssim*loss_ssim + lambda_per*loss_per + lambda_con*loss_con + lambda_vq*loss_vq
                    
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    train_metrics['total'].append(loss.item())
                    train_metrics['rec'].append(loss_rec.item())
                    train_metrics['con'].append(loss_con.item())
                    train_metrics['vq'].append(loss_vq.item())
                    pbar.set_postfix(**{'vq': loss_vq.item(), 'rec': loss_rec.item(), 'ssim': loss_ssim.item(), 'per': loss_per.item(), 'con': loss_con.item()})
                    pbar.update(1)

                    if (total_step % args['train']['vis_steps']) == 0:
                        with torch.no_grad():
                            output0, _, _, _ = net.forward_single(source_img, torch.from_numpy(np.array([1 if i==0 else 0 for i in range(c_s)])).reshape((1,c_s)).to(device=device, dtype=torch.float32))
                            output1, _, _, _ = net.forward_single(source_img, torch.from_numpy(np.array([1 if i==1 else 0 for i in range(c_s)])).reshape((1,c_s)).to(device=device, dtype=torch.float32))
                            output2, _, _, _ = net.forward_single(source_img, torch.from_numpy(np.array([1 if i==2 else 0 for i in range(c_s)])).reshape((1,c_s)).to(device=device, dtype=torch.float32))
                            output3, _, _, _ = net.forward_single(source_img, torch.from_numpy(np.array([1 if i==3 else 0 for i in range(c_s)])).reshape((1,c_s)).to(device=device, dtype=torch.float32))
                            
                            vimage = torch.stack([
                                source_img, source_img_aug, rec_src2tgt, target_img, target_img, target_img_aug, rec_tgt2src, source_img,
                                rec_src2src, source_img, rec_tgt2tgt, target_img, rec_rand2src, source_img, rec_rand2tgt, target_img,
                                inputs[0], output0, inputs[1], output1, inputs[2], output2, inputs[3], output3,
                                    ], dim=1)[:,:].reshape(-1,1,output1.shape[2],output1.shape[3])
                            vimage = torch.clamp(vimage, min=-1, max=1)/2+0.5
                            torchvision.utils.save_image(vimage, os.path.join(dir_visualize, '{}.jpg'.format(epoch)))
                    
                    if (total_step % args['train']['ckpt_steps']) == 0:
                        torch.save(net.state_dict(), os.path.join(dir_checkpoint, 'ckpt_tmp.pth'))

                    total_step += 1
                    if total_step > args['train']['total_steps']:
                        #torch.save(net.state_dict(), os.path.join(dir_checkpoint, 'ckpt_latest.pth'))
                        return
                #break

        net.eval()
        test_psnr = []
        with torch.no_grad():
            for batch in valid_loader:
                img_t1 = batch['t1']
                img_t1ce = batch['t1ce']
                img_t2 = batch['t2']
                img_flair = batch['flair']
                
                flags = [i[0] for i in batch['flag']]

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
                    img_t1[rd:rd+nd,:,0,rw:rw+nw,rh:rh+nh].to(device=device, dtype=torch.float32),
                    img_t1ce[rd:rd+nd,:,0,rw:rw+nw,rh:rh+nh].to(device=device, dtype=torch.float32),
                    img_t2[rd:rd+nd,:,0,rw:rw+nw,rh:rh+nh].to(device=device, dtype=torch.float32),
                    img_flair[rd:rd+nd,:,0,rw:rw+nw,rh:rh+nh].to(device=device, dtype=torch.float32),
                ]

                for src in flags:
                    source_img = inputs[src]
                    for tgt in flags:
                        target_img = inputs[tgt]
                        target_code = torch.from_numpy(np.array([1 if i==tgt else 0 for i in range(c_s)])).reshape((1,c_s)).to(device=device, dtype=torch.float32)
                        output_target, _, _, _ = net.forward_single(source_img, target_code)
                        psnr = torch_PSNR(target_img, output_target, data_range=2)
                        test_psnr.append(psnr.item())
                #break
        
        loss = np.mean(test_psnr)
        recorder.update({'vq': np.mean(train_metrics['vq']), 'rec': np.mean(train_metrics['rec']), 'con': np.mean(train_metrics['con']), 'psnr': np.mean(test_psnr)})
        plotter.send(recorder.call())
        if best_psnr<loss:
            best_psnr = loss
            torch.save(net.state_dict(), os.path.join(dir_checkpoint, 'ckpt_best.pth'))
            with open(os.path.join(dir_checkpoint, 'log.csv'), 'a+') as f:
                f.write('{},{},{},{},{}\n'.format(epoch+1, np.mean(train_metrics['rec']), np.mean(train_metrics['con']), np.mean(train_metrics['vq']), np.mean(test_psnr)))

        torch.cuda.empty_cache()

def get_args():
    parser = argparse.ArgumentParser(description='Train seq2seq model',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-c', '--config', dest='config', type=str, default='config/seq2seq_brats_2d_missing.yaml',
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