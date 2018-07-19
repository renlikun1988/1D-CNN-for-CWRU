# -*- coding: utf-8 -*-
"""
Created on Thu Apr 19 10:32:08 2018

@author: rlk
"""

def create_model(model_name, model_param):
    model = None
    if model_name == 'plain_cnn':
        from .costumed_model import CWRUcnn
        model = CWRUcnn(**model_param)
    else:
        raise NotImplementedError('model [%s] not implemented.' % model_name)
    print("model [%s] was created" % model_name)
    return model
    """
    elif model_name == 'pix2pix':
        from .pix2pix_model import Pix2PixModel
        model = Pix2PixModel()
    elif model_name == 'test':
        assert(opt.dataset_mode == 'single')
        from .test_model import TestModel
        model = TestModel()
    """
    