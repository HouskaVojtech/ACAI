#!/usr/bin/env python
# coding: utf-8

# In[1]:


import librosa
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
from IPython.display import clear_output
from jupyterplot import ProgressPlot

from time import time
from collections import defaultdict

import torch
from torch import nn, optim
import torch.nn.functional as F
import math
import torchvision
from torch.utils.data import DataLoader
from IPython.display import clear_output

from math import isqrt


# In[2]:




# In[3]:


LR = 0.0001
COLORS = 1
activation = nn.LeakyReLU
EPOCHS = 100
LEAKY_RELU_SLOPE = 0.2
INPUT_SIZE = 32


# In[4]:


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder,self).__init__()
        
        layers = []
        layers.append(nn.Conv2d(in_channels=COLORS, out_channels = 16, kernel_size=1, padding=1))
        layers += [nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, padding=1), activation(LEAKY_RELU_SLOPE)]
        layers += [nn.Conv2d(16, 16, 3, padding=1), activation(LEAKY_RELU_SLOPE)]
        layers.append(nn.AvgPool2d(2))
        
        layers += [nn.Conv2d(in_channels=16, out_channels=8, kernel_size=3, padding=1), activation(LEAKY_RELU_SLOPE)]
        layers += [nn.Conv2d(8, 8, 3, padding=1), activation(LEAKY_RELU_SLOPE)]
        layers.append(nn.AvgPool2d(2))
        
        layers += [nn.Conv2d(in_channels=8, out_channels=4, kernel_size=3, padding=1), activation(LEAKY_RELU_SLOPE)]
        layers += [nn.Conv2d(4, 4, 3, padding=1), activation(LEAKY_RELU_SLOPE)]
        layers.append(nn.AvgPool2d(2))
        
        layers += [nn.Conv2d(in_channels=4, out_channels=4, kernel_size=3, padding=1), activation(LEAKY_RELU_SLOPE)]
        layers += [nn.Conv2d(4, 2, 3, padding=1), activation(LEAKY_RELU_SLOPE)]
        
        self.layer = nn.Sequential(*layers)
        
    def forward (self,x):
        return self.layer(x)
    
    
class Decoder(nn.Module):
    def __init__(self):
        super(Decoder,self).__init__()
        
        layers = []
        
        layers += [nn.Conv2d(2, 4, 3, padding=1), activation(LEAKY_RELU_SLOPE)]
        layers += [nn.Conv2d(4, 4, 3, padding=1), activation(LEAKY_RELU_SLOPE)]
        layers.append(nn.Upsample(scale_factor=2))
        
        layers += [nn.Conv2d(4, 8, 3, padding=1), activation(LEAKY_RELU_SLOPE)]
        layers += [nn.Conv2d(8, 8, 3, padding=1), activation(LEAKY_RELU_SLOPE)]
        layers.append(nn.Upsample(scale_factor=2))
        
        layers += [nn.Conv2d(8, 16, 3, padding=1), activation(LEAKY_RELU_SLOPE)]
        layers += [nn.Conv2d(16, 16, 3, padding=1), activation(LEAKY_RELU_SLOPE)]
        layers.append(nn.Upsample(scale_factor=2))
        
        layers += [nn.Conv2d(16, 16, 3, padding=1), activation(LEAKY_RELU_SLOPE)]
        layers.append(nn.Conv2d(16, COLORS, 3, padding=1))
        
        self.layer = nn.Sequential(*layers)
        
    def forward (self,x):
        return self.layer(x)
    
    
class Critic(nn.Module):
    def __init__(self):
        super(Critic, self).__init__()
        self.encoder = Encoder()
        
    def forward(self, x):
        x = self.encoder(x)
        x = x.reshape(x.shape[0], -1)
        x = torch.mean(x, -1)
        return x


# In[5]:


def squared_l2(x):
    return torch.norm(x)**2


# In[6]:


def swap(x):
    a, b = x.split(math.ceil(x.shape[0]/2))
    return torch.cat([b, a])


# In[7]:


def mean_of_squares ( x ):
    return torch.mean(x**2)




# In[10]:


def lerp(start, end, weights):
    return start + weights * (end - start)


# In[11]:


def create_mosaic(torch_array):
    n_imgs = torch_array.shape[0]
    n_channels = torch_array.shape[1]
    height = torch_array.shape[2]
    width = torch_array.shape[3]
    
    sqrt = isqrt(n_imgs)
    if sqrt**2 != n_imgs:
        raise ValueError("Number of images has to have integer square root!")
        
    if len(torch_array.shape) != 4:
        raise ValueError("Number of dimensions needs to be 4: N_imgs, height, width, n_channels")
        
    im = torch_array.reshape(n_imgs*height,width,n_channels) 
    return torch.cat(torch.split(im,height*sqrt),axis=1).detach().numpy()
    
def reconstruct(x):
    return decoder(encoder(x)).cpu()


# In[12]:


def interpolate_2(x):
    n_imgs = x.shape[0] 
    split = n_imgs // 2 
    if n_imgs % 2 != 0:
        raise ValuError('The number of images has to be divisible by 2')
    
    z = encoder(x)
    # split z into two vectors 
    # interpolate between the latent vectors then decoder(latent)
    
    alpha = torch.linspace(0,1,split).reshape(-1,1,1,1).to('cuda')
    
    a = z[split:]
    b = z[:split]
    
    interpolations = [ a * (1 - al ) + b * al for al in alpha ]
    
    res = [decoder(z) for z in interpolations]
    
    return torch.cat(res).cpu()
    

if __name__ == "__main__":

    encoder = Encoder().cuda() 
    decoder = Decoder().cuda()
    critic = Critic().cuda()

    ae_optim = torch.optim.Adam(list(encoder.parameters()) + list(decoder.parameters()),lr=LR)
    critic_optim = torch.optim.Adam(critic.parameters())




    test_loader = DataLoader(dataset, batch_size=60000)
    for i in test_loader:
        test_x = i[0].to('cuda')
        test_y = i[1]


    # In[13]:

    dataset = torchvision.datasets.MNIST('/home/volta/Projects/Bakalarka/data',
                                         transform=torchvision.transforms.Compose([
                                             torchvision.transforms.ToTensor(),
                                             torchvision.transforms.Resize(32),
                                             torchvision.transforms.Normalize((0.1307,), (0.3081,))
                                             ])
                                         )


    dataloader = DataLoader(dataset, batch_size=256, num_workers=12)

    reconstruction_sample = next(iter(dataloader))[0][:8**2].to('cuda')
    interpolation_sample = next(iter(dataloader))[0][:16].to('cuda')

    pp = ProgressPlot(line_names=[ "std of criti", "critic_loss", "ae_loss"])

    start = time()

    l_param = 0.5
    g_param = 0.2


    for epoch in range(EPOCHS):
        
        for batch in dataloader:
            
            ae_optim.zero_grad()
            critic_optim.zero_grad()
            
            batch = batch[0].to('cuda')
            
            z1 = encoder(batch)
            ae_out = decoder(z1)
            
            alpha = torch.FloatTensor(batch.shape[0],1,1,1).uniform_(0,0.5).to('cuda')
            
            
            x_alpha = decoder(torch.lerp(swap(z1),z1,alpha))
            
            cr_x_alpha = critic(x_alpha)
            
            ae_loss = F.mse_loss(ae_out, batch) + l_param * mean_of_squares(cr_x_alpha)
            
            critic_loss = F.mse_loss(cr_x_alpha, alpha.reshape(-1)) + mean_of_squares(critic(torch.lerp(ae_out,batch, g_param)))
            
            pp.update([[torch.std(cr_x_alpha).item(),critic_loss,ae_loss]])
                                                                           
            ae_loss.backward(retain_graph=True)
            critic_loss.backward()
            
            ae_optim.step()
            critic_optim.step()        
            
        #interp_mosaic = create_mosaic(interpolate_2(interpolation_sample))
        #reconstruct_mosaic = create_mosaic(reconstruct(reconstruction_sample))
        #
        #clear_output(wait=True)
        #
        #fig,ax = plt.subplots(1,2)
        #ax[0].imshow(interp_mosaic)
        #ax[1].imshow(reconstruct_mosaic)
        #
        #plt.show()
            
            
    end = time()
    print(end-start)


    # In[14]:


    encoder.eval()
    decoder.eval()
    critic.eval()


