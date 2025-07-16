from typing import Tuple, Union, List, Optional
import os
import numpy as np

import torch
import torch.nn.functional as F
from torch._dynamo import OptimizedModule

import sys
sys.path.append('.')
import nnseq2seq
from nnseq2seq.inference.predict_from_raw_data import nnSeq2SeqPredictor
from nnseq2seq.utilities.find_class_by_name import recursive_find_python_class
from batchgenerators.utilities.file_and_folder_operations import join
from nnseq2seq.utilities.plans_handling.plans_handler import ConfigurationManager


class nnSeq2SeqPipeline(object):
    def __init__(self,
                 device,
                 step_size=0.5,
                 disable_tta=False,
                 verbose=False,
                 disable_progress_bar=False,
                 infer_all=True,
                 infer_input=False,
                 infer_segment=False,
                 infer_translate=False,
                 infer_translate_target='all',
        ):
        assert device in ['cpu', 'cuda',
                            'mps'], f'-device must be either cpu, mps or cuda. Other devices are not tested/supported. Got: {device}.'
        if device == 'cpu':
            # let's allow torch to use hella threads
            import multiprocessing
            torch.set_num_threads(multiprocessing.cpu_count())
            device = torch.device('cpu')
        elif device == 'cuda':
            # multithreading in torch doesn't help nnSeq2Seq if run on GPU
            torch.set_num_threads(1)
            torch.set_num_interop_threads(1)
            device = torch.device('cuda')
        else:
            device = torch.device('mps')


        self.predictor = nnSeq2SeqPredictor(
            tile_step_size=step_size,
            use_gaussian=True,
            use_mirroring=not disable_tta,
            perform_everything_on_device=True,
            device=device,
            verbose=verbose,
            verbose_preprocessing=verbose,
            allow_tqdm=not disable_progress_bar)
        
        
        infer_translate_target = [infer_translate_target] if infer_translate_target == 'all' else [int(i) for i in infer_translate_target]

        self.predictor.infer_translate_target = infer_translate_target
        if infer_all:
            self.predictor.infer_input = True
            self.predictor.infer_segment = True
            self.predictor.infer_translate = True
        else:
            self.predictor.infer_input = infer_input
            self.predictor.infer_segment = infer_segment
            self.predictor.infer_translate = infer_translate
        
    
    def initialize_from_trained_model_folder(self, model_training_output_dir: str,
                                             use_folds: Union[Tuple[Union[int, str]], None],
                                             checkpoint_name: str = 'checkpoint_final.pth'):
        self.predictor.initialize_from_trained_model_folder(
            model_training_output_dir=model_training_output_dir,
            use_folds=use_folds,
            checkpoint_name=checkpoint_name,
        )


    def load_state_dict_for_one_fold(self,
                                     use_fold: int):
        params = self.predictor.list_of_parameters[use_fold]

        # messing with state dict names...
        if not isinstance(self.predictor.network, OptimizedModule):
            self.predictor.network.load_state_dict(params)
        else:
            self.predictor.network._orig_mod.load_state_dict(params)

        self.predictor.network = self.predictor.network.to(device=self.predictor.device, dtype=torch.half)
        self.predictor.network.eval()
    

    def data_normalize(self, data: np.ndarray, seg: np.ndarray, available_sequence: List[int]) -> np.ndarray:
        configuration_manager = self.predictor.configuration_manager
        foreground_intensity_properties_per_channel = self.predictor.plans_manager.foreground_intensity_properties_per_channel

        for cseq in available_sequence:
            scheme = configuration_manager.normalization_schemes[cseq]
            normalizer_class = recursive_find_python_class(join(nnseq2seq.__path__[0], "preprocessing", "normalization"),
                                                           scheme,
                                                           'nnseq2seq.preprocessing.normalization')
            if normalizer_class is None:
                raise RuntimeError(f'Unable to locate class \'{scheme}\' for normalization')
            normalizer = normalizer_class(use_mask_for_norm=configuration_manager.use_mask_for_norm[cseq],
                                          intensityproperties=foreground_intensity_properties_per_channel[str(cseq)])
            data[cseq] = normalizer.run(data[cseq], seg[0])
        return data


    def predict_from_image_files(self,
                                 input_folder,
                                 output_folder,
                                 save_probabilities=False,
                                 continue_prediction=False,
                                 npp=3,
                                 nps=3,
                                 prev_stage_predictions=None,
                                 num_parts=1,
                                 part_id=0,
        ):
        os.makedirs(output_folder, exist_ok=True)

        # slightly passive aggressive haha
        assert part_id < num_parts, 'Do you even read the documentation? See nnSeq2Seqv2_predict -h.'

        self.predictor.predict_from_files(input_folder, output_folder, save_probabilities=save_probabilities,
                                 overwrite=not continue_prediction,
                                 num_processes_preprocessing=npp,
                                 num_processes_segmentation_export=nps,
                                 folder_with_segs_from_prev_stage=prev_stage_predictions,
                                 num_parts=num_parts,
                                 part_id=part_id)
        
    def predict_from_files_sequential(self,
                                 input_folder,
                                 output_folder,
                                 save_probabilities=False,
                                 overwrite=True,
                                 folder_with_segs_from_prev_stage=None,
        ):
        os.makedirs(output_folder, exist_ok=True)
        self.predictor.predict_from_files_sequential(input_folder, output_folder, save_probabilities=save_probabilities,
                                                     overwrite=overwrite, folder_with_segs_from_prev_stage=folder_with_segs_from_prev_stage)

    def predict_from_image_volume(self,
                                  data: np.array,
                                  properties=None,
        ):
        d, w, h = data.shape[1:4]
        assert d%16==0 and w%16==0 and h%16==0, 'data shape should be divisible by 16, but got ({}, {}, {})'.format(d, w, h)
        data = np.array(data, dtype=np.float32)
        data = self.data_normalize(data=data, seg=np.ones_like(data),
                               available_sequence=properties['available_channel'])
        data = torch.from_numpy(data).to(dtype=torch.float32)
        
        prediction, prediction_mask = self.predictor.predict_logits_from_preprocessed_data(data=data, properties=properties)
        prediction = torch.clamp(prediction, min=0, max=1).to(dtype=torch.float32).numpy()
        prediction_mask = torch.argmax(prediction_mask, dim=0).to(dtype=torch.int16).numpy()
        
        return prediction, prediction_mask
    

    def predict_from_image_slice_tensor(self,
                                        data: torch.Tensor,
                                        properties=None,
        ):
        with torch.no_grad():
            src_code = torch.from_numpy(np.array([1 if i in properties['available_channel'] and i not in properties['output_domain'] else 0 for i in range(properties['num_channel'])])).unsqueeze(0).to(self.predictor.device, dtype=data.dtype, non_blocking=True)
            if len(data.shape)==4:
                src_code_unsqueeze = src_code.unsqueeze(-1).unsqueeze(-1)
            elif len(data.shape)==5:
                src_code_unsqueeze = src_code.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)

            data = data*src_code_unsqueeze
            prediction, prediction_mask = self.predictor.network(data, data, src_code)
        return prediction, prediction_mask


if __name__ == '__main__':
    pass