from typing import Tuple, Union, List
import numpy as np
from nnseq2seq.imageio.base_reader_writer import BaseReaderWriter
import SimpleITK as sitk


class SimpleITKIO(BaseReaderWriter):
    supported_file_endings = [
        '.nii.gz',
        '.nrrd',
        '.mha'
    ]

    def read_images(self, image_fnames: Union[List[str], Tuple[str, ...]]) -> Tuple[np.ndarray, dict]:
        images = []
        spacings = []
        origins = []
        directions = []

        spacings_for_nnseq2seq = []
        for f in image_fnames:
            itk_image = sitk.ReadImage(f)
            spacings.append(itk_image.GetSpacing())
            origins.append(itk_image.GetOrigin())
            directions.append(itk_image.GetDirection())
            npy_image = sitk.GetArrayFromImage(itk_image)
            if npy_image.ndim == 2:
                # 2d
                npy_image = npy_image[None, None]
                max_spacing = max(spacings[-1])
                spacings_for_nnseq2seq.append((max_spacing * 999, *list(spacings[-1])[::-1]))
            elif npy_image.ndim == 3:
                # 3d, as in original nnseq2seq
                npy_image = npy_image[None]
                spacings_for_nnseq2seq.append(list(spacings[-1])[::-1])
            elif npy_image.ndim == 4:
                # 4d, multiple modalities in one file
                spacings_for_nnseq2seq.append(list(spacings[-1])[::-1][1:])
                pass
            else:
                raise RuntimeError(f"Unexpected number of dimensions: {npy_image.ndim} in file {f}")

            images.append(npy_image)
            spacings_for_nnseq2seq[-1] = list(np.abs(spacings_for_nnseq2seq[-1]))

        if not self._check_all_same([i.shape for i in images]):
            print('ERROR! Not all input images have the same shape!')
            print('Shapes:')
            print([i.shape for i in images])
            print('Image files:')
            print(image_fnames)
            raise RuntimeError()
        if not self._check_all_same(spacings):
            print('ERROR! Not all input images have the same spacing!')
            print('Spacings:')
            print(spacings)
            print('Image files:')
            print(image_fnames)
            raise RuntimeError()
        if not self._check_all_same(origins):
            print('WARNING! Not all input images have the same origin!')
            print('Origins:')
            print(origins)
            print('Image files:')
            print(image_fnames)
            print('It is up to you to decide whether that\'s a problem. You should run nnSeq2Seq_plot_dataset_pngs to verify '
                  'that segmentations and data overlap.')
        if not self._check_all_same(directions):
            print('WARNING! Not all input images have the same direction!')
            print('Directions:')
            print(directions)
            print('Image files:')
            print(image_fnames)
            print('It is up to you to decide whether that\'s a problem. You should run nnSeq2Seq_plot_dataset_pngs to verify '
                  'that segmentations and data overlap.')
        if not self._check_all_same(spacings_for_nnseq2seq):
            print('ERROR! Not all input images have the same spacing_for_nnseq2seq! (This should not happen and must be a '
                  'bug. Please report!')
            print('spacings_for_nnseq2seq:')
            print(spacings_for_nnseq2seq)
            print('Image files:')
            print(image_fnames)
            raise RuntimeError()

        stacked_images = np.vstack(images)
        dict = {
            'sitk_stuff': {
                # this saves the sitk geometry information. This part is NOT used by nnSeq2Seq!
                'spacing': spacings[0],
                'origin': origins[0],
                'direction': directions[0]
            },
            # the spacing is inverted with [::-1] because sitk returns the spacing in the wrong order lol. Image arrays
            # are returned x,y,z but spacing is returned z,y,x. Duh.
            'spacing': spacings_for_nnseq2seq[0]
        }
        return stacked_images.astype(np.float32), dict

    def read_seg(self, seg_fname: str) -> Tuple[np.ndarray, dict]:
        return self.read_images((seg_fname, ))

    def write_seg(self, seg: np.ndarray, output_fname: str, properties: dict) -> None:
        assert seg.ndim == 3, 'segmentation must be 3d. If you are exporting a 2d segmentation, please provide it as shape 1,x,y'
        output_dimension = len(properties['sitk_stuff']['spacing'])
        assert 1 < output_dimension < 4
        if output_dimension == 2:
            seg = seg[0]

        itk_image = sitk.GetImageFromArray(seg.astype(np.float32))
        itk_image.SetSpacing(properties['sitk_stuff']['spacing'])
        itk_image.SetOrigin(properties['sitk_stuff']['origin'])
        itk_image.SetDirection(properties['sitk_stuff']['direction'])

        sitk.WriteImage(itk_image, output_fname, True)