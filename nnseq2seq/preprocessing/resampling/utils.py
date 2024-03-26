from typing import Callable

import nnseq2seq
from batchgenerators.utilities.file_and_folder_operations import join
from nnseq2seq.utilities.find_class_by_name import recursive_find_python_class


def recursive_find_resampling_fn_by_name(resampling_fn: str) -> Callable:
    ret = recursive_find_python_class(join(nnseq2seq.__path__[0], "preprocessing", "resampling"), resampling_fn,
                                      'nnseq2seq.preprocessing.resampling')
    if ret is None:
        raise RuntimeError("Unable to find resampling function named '%s'. Please make sure this fn is located in the "
                           "nnseq2seq.preprocessing.resampling module." % resampling_fn)
    else:
        return ret