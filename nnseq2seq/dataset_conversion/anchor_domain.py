import numpy as np
import SimpleITK as sitk


def CannyEdgeDetection(itk_imgs, low_thresh_percent: float=5, high_thresh_percent: float=95, gaussian_variance: float=5, maximum_error: float=0.5):
    """
    itk_imgs:            SimpleITK Image variable for input image
    low_thresh_percent:  Percentile of gradient for lower threshold
    high_thresh_percent: Percentile of gradient for upper threshold
    gaussian_variance:   Variance of Gaussian smooth
    maximum_error:       Maximum error of Gaussian smooth
    """
    itk_imgs = sitk.Cast(itk_imgs, sitk.sitkFloat32)
    
    gradient_magnitude_image = sitk.GetArrayFromImage(sitk.GradientMagnitude(itk_imgs))
    gradient_magnitude_image = gradient_magnitude_image[gradient_magnitude_image>0]

    low_thresh_p = low_thresh_percent
    high_thresh_p = high_thresh_percent
    gaussian_var = gaussian_variance
    for i in range(11):
        low_threshold = np.percentile(gradient_magnitude_image, low_thresh_p)
        high_threshold = np.percentile(gradient_magnitude_image, high_thresh_p)
        
        canny_filter = sitk.CannyEdgeDetectionImageFilter()
        canny_filter.SetLowerThreshold(low_threshold)
        canny_filter.SetUpperThreshold(high_threshold)
        canny_filter.SetVariance(gaussian_var)
        canny_filter.SetMaximumError(maximum_error)
        edge_imgs = canny_filter.Execute(itk_imgs)

        if np.mean(sitk.GetArrayFromImage(edge_imgs))>0.05:
            break
        else:
            low_thresh_p = low_thresh_percent*(9-i)*0.1
            high_thresh_p = high_thresh_percent*(10-i)*0.1
            gaussian_var = gaussian_variance*(10-i)*0.1
    return edge_imgs


def GetForegroundMask(itk_imgs, mask_label=1):
    arr = sitk.GetArrayFromImage(itk_imgs)
    mask = (arr>0)*mask_label
    mask = sitk.GetImageFromArray(mask)
    mask.CopyInformation(itk_imgs)
    mask = sitk.Cast(mask, sitk.sitkInt8)
    return mask