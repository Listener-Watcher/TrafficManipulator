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

# RMSE error
def dist(x,y):
    return torch.sqrt(torch.sum((x-y)**2,dim=1)/x.size(1))
cuda = True if torch.cuda.is_available() else False
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

# dataloader
benign_feat = np.load('../example/train_ben.npy')
np.random.shuffle(benign_feat)
malicious_feat = np.load('../example/test.npy')
np.random.shuffle(malicious_feat)
benign_feat = torch.from_numpy(benign_feat).float()
malicious_feat = torch.from_numpy(malicious_feat).float()
benign_feat = Variable(benign_feat.type(Tensor))
malicious_feat = Variable(malicious_feat.type(Tensor))

# Initialize generator and discriminator
generator = Generator(100)
discriminator = Discriminator(100)
if cuda:
    generator.cuda()
    discriminator.cuda()
optimizer_G = torch.optim.Adam(generator.parameters(), lr=3e-5)
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=3e-5)

batches_done = 0
n_epochs = 1000
batch_size = 1375
n_critic = 1
for epoch in range(n_epochs):
    permutation_m = torch.randperm(malicious_feat.size(0))
    permutation_n = torch.randperm(benign_feat.size(0))
    for i in range(0,40):
        # Train Discriminator
        optimizer_D.zero_grad()
        noise = Variable(Tensor(np.random.normal(0, 1, (250,30))))
        indices_m = permutation_m[i*250:i*250+250]
        indices_n = permutation_n[i*batch_size:i*batch_size+batch_size]
        batch_malicious, batch_benign = malicious_feat[indices_m], benign_feat[indices_n]
        z = torch.cat((batch_malicious,noise),dim=1)
        fake_features = generator(z).detach()
        # Adversarial loss
        loss_D = -torch.mean(torch.log(1-discriminator(batch_benign))) - torch.mean(torch.log(discriminator(fake_features)))
        loss_D.backward()
        optimizer_D.step()
        # Clip weight of discriminator
        # for p in discriminator.parameters():
        #     p.data.clamp_(-opt.clip_value, opt.clip_value)
        # Train the generator every n_critic iterations
        if i % n_critic == 0:

            # -----------------
            #  Train Generator
            # -----------------

            optimizer_G.zero_grad()
            # Generate a batch of malicious features
            noise = Variable(Tensor(np.random.normal(0, 1, (250,30))))
            z = torch.cat((batch_malicious,noise),dim=1)
            fake_features = generator(z)
            # Adversarial loss
            loss_G = torch.mean(torch.log(discriminator(fake_features))+dist(batch_malicious,fake_features))
            loss_G.backward()
            optimizer_G.step()

            print(
                "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
                % (epoch, n_epochs, batches_done % 40, 40, loss_D.item(), loss_G.item())
            )

        # if batches_done % sample_interval == 0:
        #     save_image(gen_imgs.data[:25], "images/%d.png" % batches_done, nrow=5, normalize=True)
        batches_done += 1

# sample the features to form mimic_set for PSO
permutation_m = torch.randperm(malicious_feat.size()[0])
noise = Variable(Tensor(np.random.normal(0, 1, (1000,30))))
indices_m = permutation_m[0:1000]
batch_malicious = malicious_feat[indices_m]
z = torch.cat((batch_malicious,noise),dim=1)
mimic_set = generator(z).detach()
torch.save(generator.state_dict(), 'generator.pt')
torch.save(discriminator.state_dict(),'discriminator.pt')
print("check before saving:",mimic_set)
np.save('mimic_set_generated.npy',mimic_set.numpy())
