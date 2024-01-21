# TSF-Seq2Seq

Pytorch implementation for paper **[An Explainable Deep Framework: Towards Task-Specific Fusion for Multi-to-One MRI Synthesis](https://doi.org/10.1007/978-3-031-43999-5_5)**

<p align="center">
<img src="./asset/overview.png" alt="intro" width="100%"/>
</p>

## Abstract
> Multi-sequence MRI is valuable in clinical settings for reliable diagnosis and treatment prognosis, but some sequences may be unusable or missing for various reasons. To address this issue, MRI synthesis is a potential solution. Recent deep learning-based methods have achieved good performance in combining multiple available sequences for missing sequence synthesis. Despite their success, these methods lack the ability to quantify the contributions of different input sequences and estimate the quality of generated images, making it hard to be practical. Hence, we propose an explainable task-specific synthesis network, which adapts weights automatically for specific sequence generation tasks and provides interpretability and reliability from two sides: (1) visualize the contribution of each input sequence in the fusion stage by a trainable task-specific weighted average module; (2) highlight the area the network tried to refine during synthesizing by a task-specific attention module. We conduct experiments on the BraTS2021 dataset of 1251 subjects, and results on arbitrary sequence synthesis indicate that the proposed method achieves better performance than the state-of-the-art methods.

## Training
If you would like to train models with different settings, you can define a `yaml` file by yourself and use the following script.
If you want to train the model with your data, you will likely need to customize your dataloader and training file.
```sh
# Train TSF-Seq2Seq with pre-trained Seq2Seq
python src/tsf/train/train_brats_tsf_seq2seq_2d.py \
    -d cuda:0 \                                            # set device
    -c config/tsf_seq2seq_brats_2d.yaml \                  # load configuration
    -m ckpt/seq2seq/brats/2d/seq2seq_brats_2d_complete.pth # load pre-trained seq2seq weights
```

## Evaluation
### Synthesis Performance
Evaluate the model with three reconstruction metrics: PSNR, SSIM, and LPIPS.

Install package for LPIPS.
```sh
pip install lpips
```

Inference model and save predicted images, then calculate and save the metrics.
```sh
python src/tsf/test/test_brats_tsf_2d_metrics.py \
    -d cuda:0 \                                        # set device
    -c config/tsf_seq2seq_brats_2d.yaml \              # load configuration
    -m ckpt/tsf_seq2seq/brats/2d/ckpt_seq2seq_best.pth # load seq2seq weights
    -l ckpt/tsf_seq2seq/brats/2d/ckpt_best.pth \       # load tsf weights
    -o results/tsf_seq2seq/brats/2d/                   # direction to save results and metrcis
```

Quantitative results for sequence translation in the paper.

<p align="center">
<img src="./asset/table1.png" alt="table1" width="50%"/>
</p>

### Sequence Contribution
Output sequence weights $\omega$ for specific task code $c$.
```sh
python src/tsf/test/cal_tsf_weigts.py \
    -d cuda:0 \                                  # set device
    -c config/tsf_seq2seq_brats_2d.yaml \        # load configuration
    -l ckpt/tsf_seq2seq/brats/2d/ckpt_best.pth \ # load tsf weights
```

Visualization of weights of sequences for synthesis task.

<p align="center">
<img src="./asset/fig3.png" alt="fig3" width="50%"/>
</p>

### Task-Specific Enhanced Map (TSEM)
Calculate TSEM for each combination of sequences.
```sh
python src/tsf/test/pred_brats_tsf_2d_tsem.py \
    -d cuda:0 \                                        # set device
    -c config/tsf_seq2seq_brats_2d.yaml \              # load configuration
    -m ckpt/tsf_seq2seq/brats/2d/ckpt_seq2seq_best.pth # load seq2seq weights
    -l ckpt/tsf_seq2seq/brats/2d/ckpt_best.pth \       # load tsf weights
    -o results/tsf_seq2seq/brats/2d/                   # direction to save results and metrcis
```

Visualization of TSEM.

<p align="center">
<img src="./asset/fig4.png" alt="fig4" width="50%"/>
</p>

## Pre-trained Models
Pre-trained Seq2Seq could be found [here](../seq2seq/README.md#pre-trained-models).

## Citation
If this repository is useful for your research, please cite:

```bib
@inproceedings{han2023explainable,
  title={An Explainable Deep Framework: Towards Task-Specific Fusion for Multi-to-One MRI Synthesis},
  author={Han, Luyi and Zhang, Tianyu and Huang, Yunzhi and Dou, Haoran and Wang, Xin and Gao, Yuan and Lu, Chunyao and Tan, Tao and Mann, Ritse},
  booktitle={International Conference on Medical Image Computing and Computer-Assisted Intervention},
  pages={45--55},
  year={2023},
  organization={Springer}
}
```
