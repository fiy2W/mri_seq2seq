import SimpleITK as sitk

import torch


def histMatch(src, tgt, hist_level: int=1024, match_points: int=7, is_torch=False, is_arr=False):
    """ histogram matching from source image to target image
    src:          SimpleITK Image variable for source image
    tgt:          SimpleITK Image variable for target image
    hist_level:   number of histogram levels
    match_points: number of match points
    """
    if is_torch:
        dtype = tgt.dtype
        src = src.to(dtype=torch.float32).numpy()
        tgt = tgt.to(dtype=torch.float32).numpy()
        is_arr = True
    if is_arr:
        src = sitk.GetImageFromArray(src)
        tgt = sitk.GetImageFromArray(tgt)

    matcher = sitk.HistogramMatchingImageFilter()
    matcher.SetNumberOfHistogramLevels(hist_level)
    matcher.SetNumberOfMatchPoints(match_points)
    matcher.ThresholdAtMeanIntensityOn()
    dist = matcher.Execute(src, tgt)

    if is_arr:
        dist = sitk.GetArrayFromImage(dist)
    if is_torch:
        dist = torch.from_numpy(dist).to(dtype=dtype)

    return dist


def linearMatch(src, tgt):
    theta = (src*tgt).mean() / (src*src).mean()
    return src*theta