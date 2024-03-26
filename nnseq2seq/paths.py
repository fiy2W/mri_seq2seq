import os

"""
PLEASE READ paths.md FOR INFORMATION TO HOW TO SET THIS UP
"""

nnSeq2Seq_raw = os.environ.get('nnSeq2Seq_raw')
nnSeq2Seq_preprocessed = os.environ.get('nnSeq2Seq_preprocessed')
nnSeq2Seq_results = os.environ.get('nnSeq2Seq_results')

if nnSeq2Seq_raw is None:
    print("nnSeq2Seq_raw is not defined and nnSeq2Seq can only be used on data for which preprocessed files "
          "are already present on your system. nnSeq2Seq cannot be used for experiment planning and preprocessing like "
          "this. If this is not intended, please read documentation/setting_up_paths.md for information on how to set "
          "this up properly.")

if nnSeq2Seq_preprocessed is None:
    print("nnSeq2Seq_preprocessed is not defined and nnSeq2Seq can not be used for preprocessing "
          "or training. If this is not intended, please read documentation/setting_up_paths.md for information on how "
          "to set this up.")

if nnSeq2Seq_results is None:
    print("nnSeq2Seq_results is not defined and nnSeq2Seq cannot be used for training or "
          "inference. If this is not intended behavior, please read documentation/setting_up_paths.md for information "
          "on how to set this up.")