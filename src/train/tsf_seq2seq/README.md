# TSF-Seq2Seq

Pytorch implementation for paper **[An Explainable Deep Framework: Towards Task-Specific Fusion for Multi-to-One MRI Synthesis](https://arxiv.org/abs/2307.00885)**

<p align="center">
<img src="./asset/overview.png" alt="intro" width="85%"/>
</p>

## Abstract
> Multi-sequence MRI is valuable in clinical settings for reliable diagnosis and treatment prognosis, but some sequences may be unusable or missing for various reasons. To address this issue, MRI synthesis is a potential solution. Recent deep learning-based methods have achieved good performance in combining multiple available sequences for missing sequence synthesis. Despite their success, these methods lack the ability to quantify the contributions of different input sequences and estimate the quality of generated images, making it hard to be practical. Hence, we propose an explainable task-specific synthesis network, which adapts weights automatically for specific sequence generation tasks and provides interpretability and reliability from two sides: (1) visualize the contribution of each input sequence in the fusion stage by a trainable task-specific weighted average module; (2) highlight the area the network tried to refine during synthesizing by a task-specific attention module. We conduct experiments on the BraTS2021 dataset of 1251 subjects, and results on arbitrary sequence synthesis indicate that the proposed method achieves better performance than the state-of-the-art methods.

## Training
If you would like to train models with different settings, you can define a `yaml` file by yourself and use the following script.
If you want to train the model with your data, you will likely need to customize your dataloader and training file.
```sh
# Train TSF-Seq2Seq with pre-trained Seq2Seq
python src/train/seq2seq/train_brats_tsf_seq2seq_2d.py \
    -d cuda:0 \                                            # set device
    -c config/tsf_seq2seq_brats_2d.yaml \                  # load configuration
    -m ckpt/seq2seq/brats/2d/seq2seq_brats_2d_complete.pth # load pre-trained seq2seq weights
```

## Pre-trained Models
Pre-trained Seq2Seq could be found [here](../seq2seq/README.md).

## Citation
If this repository is useful for your research, please cite:

```bib
@article{han2023explainable,
  title={An Explainable Deep Framework: Towards Task-Specific Fusion for Multi-to-One MRI Synthesis},
  author={Han, Luyi and Zhang, Tianyu and Huang, Yunzhi and Dou, Haoran and Wang, Xin and Gao, Yuan and Lu, Chunyao and Tao, Tan and Mann, Ritse},
  journal={arXiv preprint arXiv:2307.00885},
  year={2023}
}
```
