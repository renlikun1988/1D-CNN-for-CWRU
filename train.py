# -*- coding: utf-8 -*-
"""
Created on Thu Apr 19 10:31:08 2018

@author: rlk
"""

from config import Config
from data import create_dataset
from models import create_model
from torch.utils.data import DataLoader
from utils import check_accuracy
import torch
from tensorboardX import SummaryWriter
import copy
import time
import pandas as pd
from torchsummary import summary

opt = Config()

tr_dataset = create_dataset(opt.train_dir, train=True)
tr_loader = DataLoader(tr_dataset, batch_size=opt.batch_size, shuffle=True)
val_dataset = create_dataset(opt.val_dir, train=False)
val_loader = DataLoader(val_dataset, batch_size=opt.batch_size, shuffle=True)
print('#training num = %d' % len(tr_dataset))
print('#val num = %d' % len(val_dataset))

model = create_model(opt.model, opt.model_param)
writer = SummaryWriter(comment=str(opt.model_param['kernel_num1'])+'_'+
                       str(opt.model_param['kernel_num2']))

total_steps = 0
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), 
                           lr=opt.lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, opt.lr_decay_iters, 
                                            opt.lr_decay)  # regulation rate decay
loss_fn = torch.nn.CrossEntropyLoss()

###==============training=================###

use_cuda = torch.cuda.is_available()
if use_cuda:
    print('CUDA is available') 
device = torch.device(opt.device if use_cuda else "cpu")
model = model.to(device)
#summary(model, (1,2048))
# save best_model wrt. val_acc
best_model_wts = copy.deepcopy(model.state_dict())
best_acc = 0.0
# one epoch
for epoch in range(opt.epochs):
    t0 = time.time()
    print('Starting epoch %d / %d' % (epoch + 1, opt.epochs))
    scheduler.step()
    # set train model or val model for BN and Dropout layers
    model.train()
    # one batch
    for t, (x, y) in enumerate(tr_loader):
        # add one dim to fit the requires of conv1d layer        
        x.resize_(x.size()[0], 1, x.size()[1]) 
        x, y = x.float(), y.long()
        x, y = x.to(device), y.to(device)
        # loss and predictions
        scores = model(x)
        loss = loss_fn(scores, y)
        writer.add_scalar('loss', loss.item())
        # print and save loss per 'print_every' times
        if (t + 1) % opt.print_every == 0:
            print('t = %d, loss = %.4f' % (t + 1, loss.item()))
        # parameters update
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()           
    # save epoch loss and acc to train or val history
    train_acc, _= check_accuracy(model, tr_loader, device)
    val_acc, _= check_accuracy(model, val_loader, device)
    # writer acc and weight to tensorboard
    writer.add_scalars('acc', {'train_acc': train_acc, 'val_acc': val_acc}, epoch)
    for name, param in model.named_parameters():
        writer.add_histogram(name, param.clone().cpu().data.numpy(), epoch)
    # save the best model
    if val_acc > best_acc:
        best_acc = val_acc
        best_model_wts = copy.deepcopy(model.state_dict())
    t1 = time.time()
    print(t1-t0)
# print results
print('kernel num1: {}'.format(opt.model_param['kernel_num1']))
print('kernel num2: {}'.format(opt.model_param['kernel_num2']))
print('Best val Acc: {:4f}'.format(best_acc))

# load best model weights
model.load_state_dict(best_model_wts)
val_acc, confuse_matrix = check_accuracy(model, val_loader, device, error_analysis=True)
# write the confuse_matrix to Excel
data_pd = pd.DataFrame(confuse_matrix)
writer = pd.ExcelWriter('results\\confuse_matrix_rate.xlsx')
data_pd.to_excel(writer)
writer.save()
writer.close()
# save model in results dir
model_save_path = 'results\\' + time.strftime('%Y%m%d%H%M_') + str(int(100*best_acc)) + '.pth'
torch.save(model.state_dict(), model_save_path)
print('best model is saved in: ', model_save_path)





