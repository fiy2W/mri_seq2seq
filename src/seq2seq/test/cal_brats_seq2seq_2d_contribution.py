import argparse
import numpy as np


def get_args():
    parser = argparse.ArgumentParser(description='Calculate metrics C_t and C_d',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-f', '--file', dest='file', type=str, default=None,
                        help='results file')
    parser.add_argument('-n', '--nseq', dest='nseq', type=int, default=None,
                        help='number of sequence')
    
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()

    N = args.nseq
    
    result_psnr = [[[] for i in range(N)] for i in range(N)]
    result_ssim = [[[] for i in range(N)] for i in range(N)]
    result_lpips = [[[] for i in range(N)] for i in range(N)]
    all_psnr = []
    all_ssim = []
    all_lpips = []
    all_psnr2 = []
    all_ssim2 = []
    all_lpips2 = []
    with open(args.file, 'r') as f:
        strs = f.readlines()
        for line in strs[1:]:
            raw = line.split(',')
            i = np.int32(raw[1])
            j = np.int32(raw[2])
            psnr = np.float32(raw[3])
            ssim = np.float32(raw[4])
            lpips = np.float32(raw[5])
            result_psnr[i][j].append(psnr)
            result_ssim[i][j].append(ssim)
            result_lpips[i][j].append(lpips)
            all_psnr.append(psnr)
            all_ssim.append(ssim)
            all_lpips.append(lpips)
            if i!=j:
                all_psnr2.append(psnr)
                all_ssim2.append(ssim)
                all_lpips2.append(lpips)

    A = [[0 for i in range(N)] for i in range(N)]

    for i in range(N):
        for j in range(N):    
            npsnr = (np.mean(result_psnr[i][j])-np.mean(all_psnr))/np.std(all_psnr)
            nssim = (np.mean(result_ssim[i][j])-np.mean(all_ssim))/np.std(all_ssim)
            nlpips = (np.mean(result_lpips[i][j])-np.mean(all_lpips))/np.std(all_lpips)
            A[i][j] = npsnr + nssim - nlpips


    for i in range(N):
        metric_ct = 0
        metric_cd = 0
        for j in range(N):
            metric_ct += A[i][j]
            metric_cd -= A[j][i]
        
        metric_ct /= N
        metric_cd /= N
        print('seq:', i, 'metric_ct:', metric_ct, 'metric_cd:', metric_cd)