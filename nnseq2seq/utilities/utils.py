import os.path
from functools import lru_cache
from typing import Union

from batchgenerators.utilities.file_and_folder_operations import *
import numpy as np
import re

from nnseq2seq.paths import nnSeq2Seq_raw


def get_identifiers_from_splitted_dataset_folder(folder: str, file_ending: str):
    files = subfiles(folder, suffix=file_ending, join=False)
    # all files have a 4 digit channel index (_XXXX)
    crop = len(file_ending) + 5
    files = [i[:-crop] for i in files]
    # only unique image ids
    files = np.unique(files)
    return files


def create_lists_from_splitted_dataset_folder(folder: str, file_ending: str, identifiers: List[str] = None) -> List[
    List[str]]:
    """
    does not rely on dataset.json
    """
    if identifiers is None:
        identifiers = get_identifiers_from_splitted_dataset_folder(folder, file_ending)
    files = subfiles(folder, suffix=file_ending, join=False, sort=True)
    list_of_lists = []
    for f in identifiers:
        p = re.compile(re.escape(f) + r"_\d\d\d\d" + re.escape(file_ending))
        list_of_lists.append([join(folder, i) for i in files if p.fullmatch(i)])
    return list_of_lists


def get_filenames_of_train_images_and_targets(raw_dataset_folder: str, dataset_json: dict = None):
    if dataset_json is None:
        dataset_json = load_json(join(raw_dataset_folder, 'dataset.json'))

    if 'dataset' in dataset_json.keys():
        dataset = dataset_json['dataset']
        for k in dataset.keys():
            dataset[k]['label'] = os.path.abspath(join(raw_dataset_folder, dataset[k]['label'])) if not os.path.isabs(dataset[k]['label']) else dataset[k]['label']
            dataset[k]['images'] = [os.path.abspath(join(raw_dataset_folder, i)) if not os.path.isabs(i) else i for i in dataset[k]['images']]
    else:
        identifiers = get_identifiers_from_splitted_dataset_folder(join(raw_dataset_folder, 'imagesTr'), dataset_json['file_ending'])
        images = create_lists_from_splitted_dataset_folder(join(raw_dataset_folder, 'imagesTr'), dataset_json['file_ending'], identifiers)
        segs = [join(raw_dataset_folder, 'labelsTr', i + dataset_json['file_ending']) for i in identifiers]
        dataset = {i: {'images': im, 'label': se} for i, im, se in zip(identifiers, images, segs)}
    return dataset


if __name__ == '__main__':
    print(get_filenames_of_train_images_and_targets(join(nnSeq2Seq_raw, 'Dataset002_Heart')))