# Seq2Seq: Sequence-to-Sequence Generator
We are committed to exploring the application of synthesis or fusion for multi-sequence MRI (also including other modalities such as CT) in clinical settings.

Seq2Seq is a series of dynamic multi-domain models that can translate an arbitrary sequence to a target sequence.
- If you are looking for a straightforward way to use it without much thought, please try [nnSeq2Seq](#nnseq2seq).
- To learn more information about our work, please refer to our [publications](#publications).

## <span id = "nnseq2seq">nnSeq2Seq</span>
Referring to [nnU-Net](https://github.com/MIC-DKFZ/nnUNet), we propose nnSeq2Seq, a tool for adaptively training Seq2Seq models with a given dataset. It will analyze the provided training cases and automatically configure a matching synthesis pipeline. No expertise is required on your end! You can easily train the models and use them for your application.

### What functions does nnSeq2Seq have?
- One image $\rightarrow$ one image
  - [x] Missing sequence/modality synthesis
  - [x] Imaging differentiation map generation
- Multiple images $\rightarrow$ one image
  - [ ] Missing sequence/modality synthesis
- Metrics calculation
  - [ ] Sequence contribution

### How to get started?
Read these:
- [Installation instructions](./nnseq2seq/docs/installation_instructions.md)
- [Dataset conversion](./nnseq2seq/docs/dataset_format.md)
- [Usage instructions](./nnseq2seq/docs/how_to_use_nnseq2seq.md)

Additional information:

## <span id = "publications">Publications</span>
Follow for our publications, which contain new features that have not yet been added to nnSeq2Seq.

If you use Seq2Seq or some part of the code, please cite (see [bibtex](./citations.bib)):

  * Seq2Seq: an arbitrary sequence to a target sequence synthesis, the sequence contribution ranking, and associated imaging-differentiation maps.
  
    **Synthesis-based Imaging-Differentiation Representation Learning for Multi-Sequence 3D/4D MRI**  
Medical Image Analysis. [![doi](https://img.shields.io/badge/DOI-8A2BE2)](https://doi.org/10.1016/j.media.2023.103044) [![arXiv](https://img.shields.io/badge/arXiv-2302.00517-red)](https://arxiv.org/abs/2302.00517) [![code](https://img.shields.io/badge/code-brightgreen)](publications/src/seq2seq/README.md)

  * TSF-Seq2Seq: an explainable task-specific synthesis network, which adapts weights automatically for specific sequence generation tasks and provides interpretability and reliability.
  
    **An Explainable Deep Framework: Towards Task-Specific Fusion for Multi-to-One MRI Synthesis**  
MICCAI2023. [![doi](https://img.shields.io/badge/DOI-8A2BE2)](https://doi.org/10.1007/978-3-031-43999-5_5) [![arXiv](https://img.shields.io/badge/arXiv-2307.00885-red)](https://arxiv.org/abs/2307.00885) [![code](https://img.shields.io/badge/code-brightgreen)](publications/src/tsf/README.md)