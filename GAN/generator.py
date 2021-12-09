import argparse
import os
import numpy as np
import math
import sys

from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch


class Generator(nn.Module):
    def __init__(self,feature_size):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(feature_size+30,feature_size+30),
            nn.Linear(feature_size+30,64),
            nn.Tanh(),
            nn.Dropout(p=0.5),
            nn.Linear(64, 128),
            nn.Tanh(),
            nn.Dropout(p=0.5),
            nn.Linear(128, 512),
            nn.Tanh(),
            nn.Dropout(p=0.5),
            nn.Linear(512,128),
            nn.Tanh(),
            nn.Dropout(p=0.5),
            nn.Linear(128,feature_size)
        )
    def forward(self,feat_noise):
        generated_feature = self.model(feat_noise)
        return generated_feature

class Discriminator(nn.Module):
    def __init__(self,feature_size):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(feature_size, feature_size),
            nn.Linear(feature_size,1024),
            nn.Sigmoid(),
            nn.Dropout(p=0.5),
            nn.Linear(1024,512),
            nn.Sigmoid(),
            nn.Dropout(p=0.5),
            nn.Linear(512,256),
            nn.Sigmoid(),
            nn.Dropout(p=0.5),
            nn.Linear(256,64),
            nn.Sigmoid(),
            nn.Dropout(p=0.5),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    def forward(self,features):
        return self.model(features)


generator = Generator(100)
# fill your architecture with the trained weights
generator.load_state_dict(torch.load("generator.pt"))
cuda = True if torch.cuda.is_available() else False
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
# dataloader
malicious_feat = np.load('../example/test.npy')
np.random.shuffle(malicious_feat)
malicious_feat = torch.from_numpy(malicious_feat).float()
malicious_feat = Variable(malicious_feat.type(Tensor))
if cuda:
    generator.cuda()
permutation_m = torch.randperm(malicious_feat.size()[0])
noise = Variable(Tensor(np.random.normal(0, 1, (1000,30))))
indices_m = permutation_m[0:1000]
batch_malicious = malicious_feat[indices_m]
z = torch.cat((batch_malicious,noise),dim=1)
mimic_set = generator(z).detach()
print('check before saving:',mimic_set)
np.save('mimic_set_gen_400.npy',mimic_set.cpu().numpy())
