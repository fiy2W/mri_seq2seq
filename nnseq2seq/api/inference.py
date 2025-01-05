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
                 infer_fusion=False,
                 infer_latent=False,
                 infer_map=False,
                 infer_translate_target='all',
                 infer_latent_level='all',
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
        infer_latent_level = [infer_latent_level] if infer_latent_level == 'all' else [int(i) for i in infer_latent_level]

        self.predictor.infer_translate_target = infer_translate_target
        self.predictor.infer_latent_level = infer_latent_level
        if infer_all:
            self.predictor.infer_input = True
            self.predictor.infer_segment = True
            self.predictor.infer_translate = True
            self.predictor.infer_fusion = True
            self.predictor.infer_latent = True
            self.predictor.infer_map = True
        else:
            self.predictor.infer_input = infer_input
            self.predictor.infer_segment = infer_segment
            self.predictor.infer_translate = infer_translate
            self.predictor.infer_fusion = infer_fusion
            self.predictor.infer_latent = infer_latent
            self.predictor.infer_map = infer_map
        
    
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

        for c, cseq in enumerate(available_sequence):
            scheme = configuration_manager.normalization_schemes[cseq]
            normalizer_class = recursive_find_python_class(join(nnseq2seq.__path__[0], "preprocessing", "normalization"),
                                                           scheme,
                                                           'nnseq2seq.preprocessing.normalization')
            if normalizer_class is None:
                raise RuntimeError(f'Unable to locate class \'{scheme}\' for normalization')
            normalizer = normalizer_class(use_mask_for_norm=configuration_manager.use_mask_for_norm[cseq],
                                          intensityproperties=foreground_intensity_properties_per_channel[str(cseq)])
            data[c] = normalizer.run(data[c], seg[0])
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


    def predict_from_image_volume(self,
                                  data: np.array,
                                  tgt_seq: int,
                                  properties=None,
        ):
        d, w, h = data.shape[1:4]
        assert d%16==0 and w%16==0 and h%16==0, 'data shape should be divisible by 16, but got ({}, {}, {})'.format(d, w, h)
        data = np.array(data, dtype=np.float32)
        data = self.data_normalize(data=data, seg=np.ones_like(data),
                               available_sequence=properties['available_channel'])
        data = torch.from_numpy(data).to(dtype=torch.float32)
        
        target_code = F.one_hot(torch.from_numpy(np.array([tgt_seq], dtype=np.int64)),
                        num_classes=properties['num_channel']).to(self.predictor.device, dtype=data.dtype, non_blocking=True)
        
        self.predictor.tsf_weight = []
        prediction, prediction_mask, prediction_fusion, prediction_latent = self.predictor.predict_logits_from_preprocessed_data(data=data, target_code=target_code, properties=properties)
        prediction = torch.clamp(prediction[0], min=0).to(dtype=torch.float32).numpy()
        prediction_mask = torch.argmax(prediction_mask, dim=0).to(dtype=torch.int16).numpy()
        prediction_fusion = torch.clamp(prediction_fusion, min=0).permute(1,2,3,0).to(dtype=torch.float32).numpy()
        prediction_latent = [F.interpolate(p.unsqueeze(0).to(dtype=torch.float32),
                                           scale_factor=(1, 0.5**(len(prediction_latent)-i-1), 0.5**(len(prediction_latent)-i-1)) if self.predictor.network.ndim==2 else 0.5**(len(prediction_latent)-i-1),
                                           mode='trilinear')[0].permute(1,2,3,0).numpy() for i, p in enumerate(prediction_latent)]
        prediction_latent_indice = [self._convert_latent_space_to_indice(p) for p in prediction_latent]
        return prediction, prediction_mask, prediction_fusion, prediction_latent, prediction_latent_indice
    
    def _convert_latent_space_to_indice(self, zq, batch_size=1):
        with torch.no_grad():
            if self.predictor.network.ndim==2:
                d, w, h, c = zq.shape
                zq = torch.from_numpy(zq).to(dtype=torch.float32)
                zq = zq.permute(0,3,1,2)
                vqindice = []
                for bi in range(0, d, batch_size):
                    _, _, (_, _, ind) = self.predictor.network.image_encoder.quantize(zq[bi:bi+batch_size].to(device=self.predictor.device))
                    vqindice.append(ind.cpu().numpy())
                vqindice = np.concatenate(vqindice, axis=0).reshape(d, w, h)
            elif self.predictor.network.ndim==3:
                zq = torch.from_numpy(zq).to(dtype=torch.float32, device=self.predictor.device)
                zq = zq.permute(3,0,1,2).unsqueeze(0)
                _, _, (_, _, vqindice) = self.predictor.network.image_encoder.quantize(zq)
                vqindice = vqindice[0].cpu().numpy()
            vqindice = np.array(vqindice, np.int16)
        return vqindice
    
    def _convert_latent_indice_to_space(self, ind):
        with torch.no_grad():
            e_dim = self.predictor.network.image_encoder.quantize.e_dim

            if self.predictor.network.ndim==2:
                d, w, h = ind.shape
                ind = torch.from_numpy(ind).to(dtype=torch.int32, device=self.predictor.device)
                vq_shape = (d,w,h,e_dim)
                ind = ind.reshape(d,w,h)
                zq = self.predictor.network.image_encoder.quantize.get_codebook_entry(ind, vq_shape)
                zq = zq.reshape(d, e_dim, w, h).permute(1,0,2,3)
            elif self.predictor.network.ndim==3:
                d, w, h = ind.shape
                ind = torch.from_numpy(ind).to(dtype=torch.int32, device=self.predictor.device)
                vq_shape = (1,d,w,h,e_dim)
                ind = ind.reshape(1,d,w,h)
                zq = self.predictor.network.image_encoder.quantize.get_codebook_entry(ind, vq_shape)
        return zq

    def predict_from_image_slice_tensor(self,
                                        data: torch.Tensor,
                                        tgt_seq: int,
                                        properties=None,
        ):
        with torch.no_grad():
            target_code = F.one_hot(torch.from_numpy(np.array([tgt_seq], dtype=np.int64)),
                            num_classes=properties['num_channel']).to(self.predictor.device, dtype=data.dtype, non_blocking=True)
            src_code = torch.from_numpy(np.array([1 if i in properties['available_channel'] and (i!=tgt_seq or len(properties['available_channel'])==1) else 0 for i in range(properties['num_channel'])])).unsqueeze(0).to(self.predictor.device, dtype=target_code.dtype, non_blocking=True)
            src_all_code = torch.from_numpy(np.array([1 if i in properties['available_channel'] else 0 for i in range(properties['num_channel'])])).unsqueeze(0).to(self.predictor.device, dtype=target_code.dtype, non_blocking=True)

            prediction, prediction_latent, latent_src_all_seg, prediction_fusion = self.predictor.network.infer(data, src_all_code, src_code, target_code)
            prediction_mask = self.predictor.network.segmentor(latent_src_all_seg)
        return prediction, prediction_mask, prediction_fusion, prediction_latent


    def predict_from_latent_volume(self,
                                   latents: List[np.array],
                                   tgt_seq: int,
                                   properties=None,
                                   latent_focal: str='dispersion',
                                   use_ind: bool=False,
        ):
        if use_ind:
            latents = [self._convert_latent_indice_to_space(lat).to(dtype=torch.float32) for lat in latents]
        else:
            latents = [torch.from_numpy(lat).to(dtype=torch.float32).permute(3,0,1,2) for lat in latents]
        d, w, h = latents[-1].shape[1:]
        for i, lat in enumerate(latents):
            ld, lw, lh = lat.shape[1:]
            scale = 2**(len(latents)-i-1)
            if self.predictor.network.ndim==2:
                assert d==ld and w//scale==lw and h//scale==lh, 'latent {} shape should be ({}, {}, {}), but got ({}, {}, {})'.format(i, d, w//scale, h//scale, ld, lw, lh)
            else:
                assert d//scale==ld and w//scale==lw and h//scale==lh, 'latent {} shape should be ({}, {}, {}), but got ({}, {}, {})'.format(i, d//scale, w//scale, h//scale, ld, lw, lh)
        
        target_code = F.one_hot(torch.from_numpy(np.array([tgt_seq], dtype=np.int64)),
                        num_classes=properties['num_channel']).to(self.predictor.device, dtype=latents[-1].dtype, non_blocking=True)
        
        prediction = self.predictor.predict_logits_from_latents(latent=latents, target_code=target_code, latent_focal=latent_focal)
        prediction = torch.clamp(prediction[0], min=0).to(dtype=torch.float32).numpy()
        return prediction
    


if __name__ == '__main__':
    pass