# Installation instructions
## Setting up Paths
nnSeq2Seq relies on environment variables to know where raw data, preprocessed data, and trained model weights are stored. To use the full functionality of nnSeq2Seq, the following three environment variables must be set:
- nnSeq2Seq_raw
- nnSeq2Seq_preprocessed
- nnSeq2Seq_results

### How to set environment variables
Refer to [nnU-Net](https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/set_environment_variables.md) for settings.

## Installation
### Create environment and install PyTorch
Create a virtual environment with Conda. Make sure you install Python>=3.10
```sh
conda create -n nnseq2seq python=3.10
conda activate nnseq2seq
```

Install PyTorch as described on their website (conda/pip).
Select your preferences and run the install command from [here](https://pytorch.org/get-started/locally/) to install the latest version. Make sure you install PyTorch>=2.0

For example, run the command as follows,
```
pip install torch torchvision torchaudio
```

### Install nnSeq2Seq
Install from source code.
```sh
git clone https://github.com/fiy2W/mri_seq2seq.git
cd mri_seq2seq
pip install -e .
```