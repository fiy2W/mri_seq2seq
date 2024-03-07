import argparse
import os
import sys
import logging
import yaml
import random
from tqdm import tqdm
import numpy as np
import SimpleITK as sitk

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import sys
sys.path.append('./publications/')

from src.seq2seq.models.seq2seq import Generator
from src.tsf.models.tsf_seq2seq import TSF_seq2seq


def get_args():
    parser = argparse.ArgumentParser(description='Test seq2seq model',
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

    device = torch.device(args.device)
    logging.info(f'Using device {device}')

    net = TSF_seq2seq(config)
    net.to(device=device)

    if args.load:
        load_dict = torch.load(args.load, map_location=device)
        net.load_state_dict(load_dict)
        print('[*] Load model from', args.load)

    c_in = config['seq2seq']['c_in']
    c_s = config['seq2seq']['c_s']
    
    with torch.no_grad():
        net.eval()
        N = 4
        for tgt in range(4):
            for src in range(1, 16):
                srcs = bin(src)[2:].zfill(4)
                source = [i for i in range(4) if srcs[i]=='1']

                target_code = torch.from_numpy(np.array([1 if i==tgt else 0 for i in range(c_s)])).reshape((1,c_s)).to(device=device, dtype=torch.float32)
                params = net.output_param(source_seqs=source, target_seq=target_code)

                print('source:', source, 'target:', tgt, 'Weights:', params.cpu().reshape(-1).numpy().tolist())