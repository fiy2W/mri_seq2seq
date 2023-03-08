# Seq2Seq: Sequence-to-Sequence Generator
Seq2Seq is a synthesis model for multi-sequence MRIs.

## Training
If you would like to train models with different settings, you can define a `yaml` file by yourself and use the following script.
If you want to train the model with your data, you will likely need to customize your dataloader and training file.
```sh
# Seq2Seq
python src/train/seq2seq/train_brats_seq2seq_2d.py \
    -d cuda:0 \                                           # set device
    -c config/seq2seq_brats_2d_missing.yaml \             # load configuration
    -l ckpt/seq2seq/brats/2d/seq2seq_brats_2d_missing.pth # load pre-trained weights or omit this to train from beginning
```

## Pre-trained Models
Download the configuration file and corresponding pre-trained weight from [here](https://drive.google.com/drive/folders/1aygogtHrr1WqHSWAgovHUgEpe-4lQd38?usp=sharing).

A list of pre-trained models is available as follows.
| Model | Dataset | Description | Config | Weights |
|:-|:-|:-|:-|:-|
| Seq2Seq | BraTS2021 | Training with incomplete dataset. | [seq2seq_brats_2d_missing.yaml](https://drive.google.com/file/d/1MN1AAMthuClarT16Jiiy9nae0t3NOGQt/view?usp=sharing) | [seq2seq_brats_2d_missing.pth](https://drive.google.com/file/d/1jPlOSpZQs_PMb4nnB89VU59nQ6grhV5f/view?usp=sharing) |
| Seq2Seq | BraTS2021 | Training with complete dataset. | [seq2seq_brats_2d_complete.yaml](https://drive.google.com/file/d/1EOBMEZFXk1jqHstxq208p13YuObED-Ay/view?usp=sharing) | [seq2seq_brats_2d_complete.pth](https://drive.google.com/file/d/19c4F3Pw_T2zye35d9fdwIHs4uKt7QRal/view?usp=sharing) |

## Seq2Seq Paper
If you use Seq2Seq or some part of the code, please cite (see [bibtex](./citations.bib)):

* Seq2Seq: an arbitrary sequence to a target sequence synthesis, the sequence contribution ranking, and associated imaging-differentiation maps.
  
  **Synthesis-based Imaging-Differentiation Representation Learning for Multi-Sequence 3D/4D MRI**
  [![arXiv](https://img.shields.io/badge/arXiv-2302.00517-red)](https://arxiv.org/abs/2302.00517)