# Seq2Seq: Sequence-to-Sequence Generator
[![Translate](https://img.shields.io/badge/翻译-中文-red)](README_cn.md)

We are committed to exploring the application of synthesis or fusion for multi-sequence MRI (also including other modalities such as CT) in clinical settings.

Seq2Seq is a series of dynamic multi-domain models that can translate an arbitrary sequence to a target sequence.
- If you are looking for a straightforward way to use it without much thought, please try [nnSeq2Seq](#nnseq2seq).
- To learn more information about our work, please refer to our [publications](#publications).

## <span id = "nnseq2seq">nnSeq2Seq</span>
Referring to [nnU-Net](https://github.com/MIC-DKFZ/nnUNet), we propose nnSeq2Seq, a tool for adaptively training Seq2Seq models with a given dataset. It will analyze the provided training cases and automatically configure a matching synthesis pipeline. No expertise is required on your end! You can easily train the models and use them for your application.

### What functions does nnSeq2Seq have?
- Input one/multiple images $\rightarrow$ output one image
  - [x] Missing sequence/modality synthesis
    - Input: [CT] $\rightarrow$ Output: [PET]
    - Input: [T1, T2, DWI, ADC] $\rightarrow$ Output: [DCE]
  - [x] Segmentation
    - Input: [DCE] $\rightarrow$ Output: [Tumor]
    - Input: [T1, T1Gd, T2, FLAIR] $\rightarrow$ Output: [Tumor]
  - [ ] Inpainting
  - [ ] Deblur and super-resolution
  - [x] vector quantized common (VQC)-latent space compression
    - Input: [CT] $\rightarrow$ Output: [VQC]
    - Input: [T1, T2] $\rightarrow$ Output: [VQC]
- Explainability and visualization
  - [x] Imaging differentiation map
  - [x] Synthesis-based sequence contribution
  - [x] Task-specific sequence contribution

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

  * VQ-Seq2Seq: a generative model that compresses discrete representations of each sequence to estimate the Gaussian distribution of vector-quantized common (VQC) latent space between multiple sequences.

    **Non-Adversarial Learning: Vector-Quantized Common Latent Space for Multi-Sequence MRI**
MICCAI2024. [![doi](https://img.shields.io/badge/DOI-8A2BE2)](
https://doi.org/10.1007/978-3-031-72120-5_45) [![arXiv](https://img.shields.io/badge/arXiv-2407.02911-red)](https://arxiv.org/abs/2407.02911) [![code](https://img.shields.io/badge/code-brightgreen)](publications/src/vq/README.md)

## <span id = "acknowledgements">Acknowledgements</span>
<img src="documentation/assets/logo-radboudumc.png" height="100px"/>
<img src="documentation/assets/logo-netherlands-cancer-institute.png" height="100px"/>
<img src="documentation/assets/logo-macao-polytechnic-university.png" height="100px"/>

## <span id = "contact">Contact</span>
For any code-related problems or questions please [open an issue](https://github.com/fiy2W/mri_seq2seq/issues/new) or concat us by emails.

- [Ritse.Mann@radboudumc.nl](mailto:Ritse.Mann@radboudumc.nl) (Ritse Mann)
- [taotan@mpu.edu.mo](mailto:taotan@mpu.edu.mo) (Tao Tan)
- [Luyi.Han@radboudumc.nl](mailto:Luyi.Han@radboudumc.nl) (Luyi Han)