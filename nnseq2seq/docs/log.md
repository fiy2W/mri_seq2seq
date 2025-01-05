## Todo
- [ ] Improve 3D model.
- [ ] Pre-trained models.
- [ ] Build docker.
- [ ] Add document and example for different tasks.

## Update log
### v0.1.2 (2025-01-06)
**`[New]:`**
- Add inference API.
- Add fusion image as new output.
- Add `EMA` model.
- Add `WarmupCosineLRScheduler`.

**`[Update]:`**
- Update the implementation of `HyperConv`.
- Update model architectures.
- Update weight of perceptual loss.
- Sequence with `'input'` in `channel_names` will be set as input domain (only used as input channel), sequence with `'output'` in `channel_names` will be set as output domain (only used as output channel).
- Update documents.

### v0.1.1 (2024-07-06)
**`[Update]:`**
- Update Seq2Seq models with VQ-U-Net architecture.
- Update target sequence sampling and model training strategy.
- Update inference results.
- Update adversarial loss with `hinge` loss.
- Update normalization and add new normalization for pre-contrast DCE-MRI and wash-in (subtraction) images.
- Update GPU memory limited to 24G.
- Update documents.

### v0.1.0 (2024-06-16)
**`[New]:`**
- Apply vector quantized common (VQC)-latent space for models (`VQ-Seq2Seq`).
- Add task-specific fusion (TSF) for models (`TSF-Seq2Seq`).
- Add new training strategy suitable for missing data.
- Sequence with `'anchor'` in `channel_names` will be set as anchor domain, add new anchor domain (Canny edge detection) and corresponding normalization scheme.
- Add discriminator and adversarial learning.
- Add segmentor, ignore to calculate segmentation loss for the subject with `'foreground'` in `labels`.
- New visualization during training.

**`[Update]:`**
- Update Seq2Seq models with self-attention and cross-attention.
- Update GPU memory limited to 12G.
- Update minimal batch size to 1.
- Update documents.

**`[Fix]:`**
- Remove data split in ``nnseq2seq/dataset_conversion/Dataset001_BraTS.py``.
- Fix bug of mismatch between normalization scheme and sequence name.

### v0.0.1 (2024-03-26)
**`[New]:`**
- Initial release.