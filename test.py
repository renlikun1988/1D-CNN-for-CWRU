# -*- coding: utf-8 -*-
"""
Created on Thu Apr 19 10:31:26 2018

@author: rlk
"""
from config import Config
from data import create_dataset
from models import create_model
from torch.utils.data import DataLoader
import torch
from utils import check_accuracy

opt = Config()
test_dataset = create_dataset(opt.test_dir)
test_loader = DataLoader(test_dataset, batch_size=opt.batch_size, shuffle=True)
print('#testing images = %d' % len(test_dataset))
model = create_model(opt.model)
model.load_state_dict(torch.load('results//201806111701_24.pth'))

use_cuda = torch.cuda.is_available()
if use_cuda:
    print('CUDA is available') 
device = torch.device("cuda" if use_cuda else "cpu")
model = model.to(device)

test_acc, confuse_matrix = check_accuracy(model, test_loader, device, error_analysis=True)