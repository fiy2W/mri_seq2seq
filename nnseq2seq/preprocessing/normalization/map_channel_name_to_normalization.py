from typing import Type

from nnseq2seq.preprocessing.normalization.default_normalization_schemes import CTNormalization, NoNormalization, \
    ZScoreNormalization, RescaleTo01Normalization, RGBTo01Normalization, ImageNormalization, \
    Rescale0_995to01Normalization, CT005_995to01Normalization, CannyNormalization, \
    precontrast_995to01Normalization, washin_995to01Normalization

channel_name_to_normalization_mapping = {
    #'CT': CTNormalization,
    #'noNorm': NoNormalization,
    #'zscore': ZScoreNormalization,
    #'rescale_to_0_1': RescaleTo01Normalization,
    #'rgb_to_0_1': RGBTo01Normalization,
    'rescale_to_0_995': Rescale0_995to01Normalization,
    'CT': CT005_995to01Normalization,
    'anchor_CT': CT005_995to01Normalization,
    'Canny': CannyNormalization,
    'fat_saturated_DCE_pre_contrast': precontrast_995to01Normalization,
    'washin': washin_995to01Normalization,
}


def get_normalization_scheme(channel_name: str) -> Type[ImageNormalization]:
    """
    If we find the channel_name in channel_name_to_normalization_mapping return the corresponding normalization. If it is
    not found, use the default (ZScoreNormalization)
    """
    norm_scheme = channel_name_to_normalization_mapping.get(channel_name)
    if norm_scheme is None:
        norm_scheme = Rescale0_995to01Normalization
    # print('Using %s for image normalization' % norm_scheme.__name__)
    return norm_scheme