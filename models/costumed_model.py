# -*- coding: utf-8 -*-
'''
20180401
nn architecture for CWRU datasets of 101classification
BY rlk
'''


from torch import nn


class Flatten(nn.Module):
    def forward(self, x):
        N, C, L = x.size()  # read in N, C, L
        z = x.view(N, -1)
#        print(C, L)
        return z  # "flatten" the C * L values into a single vector per image


class CWRUcnn(nn.Module):
    def __init__(self, kernel_num1=81, kernel_num2=27, kernel_size=55, pad=0, ms1=16, ms2=16):
        super(CWRUcnn, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv1d(1, kernel_num1, kernel_size, padding = pad),
            nn.BatchNorm1d(kernel_num1),
            nn.ReLU(),
            nn.MaxPool1d(ms1),
            nn.Conv1d(kernel_num1, kernel_num1, kernel_size, padding = pad),
            nn.BatchNorm1d(kernel_num1),
            nn.ReLU(),
            nn.Dropout(),
            nn.Conv1d(kernel_num1, kernel_num1, kernel_size, padding = pad),
            nn.BatchNorm1d(kernel_num1),
            nn.ReLU(),
            nn.MaxPool1d(ms2),
            nn.Conv1d(kernel_num1, kernel_num2, kernel_size, padding = pad),
            nn.BatchNorm1d(kernel_num2),
            nn.ReLU(),
            nn.Dropout(),
            nn.Conv1d(kernel_num2, kernel_num2, kernel_size, padding = pad),
            nn.BatchNorm1d(kernel_num2),
            nn.ReLU(),
            Flatten(),
            nn.Linear(kernel_num2*8, 101)) #ms1=16,ms2=16
#            nn.Linear(27*14, 101)) #ms1=16,ms2=9
#            nn.Linear(27*25, 101)) #ms1=9,ms2=9
#            nn.Linear(27*75, 101))  #ms1=9,ms2=3

    def forward(self, x):
        return self.layers(x)
