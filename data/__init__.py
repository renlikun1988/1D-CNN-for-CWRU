# -*- coding: utf-8 -*-
"""
Created on Thu Apr 19 10:32:59 2018

@author: rlk
"""


def create_dataset(data_dir, train):
    dataset = None
    if data_dir == 'data/DE.h5' or data_dir == 'data/FE.h5' :
        from .custom_dataset import CWRUdata
        dataset = CWRUdata(data_dir, train)
    else:
        raise ValueError("Dataset [%s] not recognized." % data_dir)
    print("dataset [%s] was created" % data_dir)
    return dataset

"""
    elif opt.dataset_mode == 'unaligned':
        from data.unaligned_dataset import UnalignedDataset
        dataset = UnalignedDataset()
    elif opt.dataset_mode == 'single':
        from data.single_dataset import SingleDataset
        dataset = SingleDataset()
"""
    
