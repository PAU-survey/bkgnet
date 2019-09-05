#!/usr/bin/env python
# encoding: UTF8

from IPython.core import debugger as ipdb
import numpy as np
import pandas as pd

from astropy.io import fits
from astropy.io.fits.hdu.image import PrimaryHDU
import random
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import datasets, models, transforms
import time
import os
import copy
from sklearn.model_selection import train_test_split
from math import sqrt
import torch.nn.functional as F

from torch.utils.data.dataset import random_split
from torch.utils.data import TensorDataset, DataLoader
from network import CNN_model
from astropy.modeling.functional_models import Sersic2D
import re


print('packages uploaded')


class BKGnet_train:
    """Training BKGnet"""

    def __init__(self, batch_size=50):

        # Load the model.
        cnn = CNN_model() 

        self.batch_size = batch_size
        self.cnn = cnn

    def _internal_naming(self, filename):
        """Converting to internal band numbering."""
        
        NB = np.asarray(re.findall("NB(\d+).", filename))
        CCD = np.asarray(re.findall("std.(\d+).", filename))
        join,nbs = '_', 'NB'
        exp_num = re.findall("paucam.(\d+).", filename)
        exp_num_0 = np.asarray(exp_num,dtype=np.int)

        if exp_num_0[0] >= 7582: 
            interv = "_1"
        if exp_num_0[0] < 7582:
            interv = "_0" 

    
        code = nbs+ NB[0]+join+ nbs+ NB[1]+ join + CCD[0] + interv

        D = {'NB455_NB525_01_0':0,'NB455_NB525_02_0':1,'NB455_NB525_03_0':2,'NB455_NB525_04_0':3,'NB455_NB525_05_0':4,'NB455_NB525_06_0':5,'NB455_NB525_07_0':6,'NB455_NB525_08_0':7,'NB535_NB605_01_0':8,'NB535_NB605_02_0':9,'NB535_NB605_03_0':10,'NB535_NB605_04_0':11,'NB535_NB605_05_0':13,'NB535_NB605_06_0':12,'NB535_NB605_07_0':15,'NB535_NB605_08_0':14,'NB615_NB685_01_0':23,'NB615_NB685_02_0':22,'NB615_NB685_03_0':21,'NB615_NB685_04_0':20,'NB615_NB685_05_0':19,'NB615_NB685_06_0':18,'NB615_NB685_07_0':17,'NB615_NB685_08_0':16,'NB695_NB765_01_0':31,'NB695_NB765_02_0':30,'NB695_NB765_03_0':29,'NB695_NB765_04_0':28,'NB695_NB765_05_0':27,'NB695_NB765_06_0':26,'NB695_NB765_07_0':25,'NB695_NB765_08_0':24,'NB775_NB845_01_0':39,'NB775_NB845_02_0':38,'NB775_NB845_03_0':37,'NB775_NB845_04_0':36,'NB775_NB845_05_0':35,'NB775_NB845_06_0':34,'NB775_NB845_07_0':33,'NB775_NB845_08_0':32,'NB455_NB525_01_1':40,'NB455_NB525_02_1':41,'NB455_NB525_03_1':42,'NB455_NB525_04_1':43,'NB455_NB525_05_1':44,'NB455_NB525_06_1':45,'NB455_NB525_07_1':46,'NB455_NB525_08_1':47,'NB535_NB605_01_1':48,'NB535_NB605_02_1':49,'NB535_NB605_03_1':50,'NB535_NB605_04_1':51,'NB535_NB605_05_1':53,'NB535_NB605_06_1':52,'NB535_NB605_07_1':55,'NB535_NB605_08_1':54,'NB615_NB685_01_1':63,'NB615_NB685_02_1':62,'NB615_NB685_03_1':61,'NB615_NB685_04_1':60,'NB615_NB685_05_1':59,'NB615_NB685_06_1':58,'NB615_NB685_07_1':57,'NB615_NB685_08_1':56,'NB695_NB765_01_1':71,'NB695_NB765_02_1':70,'NB695_NB765_03_1':69,'NB695_NB765_04_1':68,'NB695_NB765_05_1':67,'NB695_NB765_06_1':66,'NB695_NB765_07_1':65,'NB695_NB765_08_1':64,'NB775_NB845_01_1':79,'NB775_NB845_02_1':78,'NB775_NB845_03_1':77,'NB775_NB845_04_1':76,'NB775_NB845_05_1':75,'NB775_NB845_06_1':74,'NB775_NB845_07_1':73,'NB775_NB845_08_1':72}     

        
        nr = D[code] - 1

        return nr
    
    def circular_mask(self):
        maskCircular = np.ones(shape = (120,120),dtype=bool)
        for k in range(-60,60):
            for j in range(-60,60):
                r  = sqrt(k**2+ j**2)    
                if r < 8:
                    maskCircular[k+60,j+60] = False 
        return maskCircular
  

    def create_stamps(self,directory):
        """Create the postage stamps from positions given in pixels."""

        fluxes = pd.read_csv('fluxes_sim_gal.csv', sep = ',',  header = 0)
        stamps = np.empty(shape = (0,120,120))
        stamps_df = pd.DataFrame()
        xgrid,ygrid = np.meshgrid(np.arange(120), np.arange(120))
        npos = 100

        for filename in os.listdir(directory):
            
            hdul =  fits.open(os.path.join(directory, filename))
            img = hdul[0].data
 

            
            coord_pix = pd.DataFrame(np.c_[np.random.randint(68,4096,npos), np.random.randint(68,2040, npos)], columns = ['x', 'y']) 
            postages = np.zeros(shape=(npos, 120, 120), dtype = np.float32)

            # Padding to allow stamps on the image edges.
            img_pad = np.pad(img, pad_width= 60 , mode='constant', constant_values=0)

            L = [] 
            for i, (ind, sub) in enumerate(coord_pix.iterrows()):
                # Remember to make a copy of the array.
                postage = img_pad[sub.x:sub.x+120, sub.y:sub.y+120].copy()
                postage = postage.astype(np.float32, copy = False)
     

                mask_central = self.circular_mask()
                masked_array_circular = np.ma.masked_array(postage, mask = mask_central)
                masked_array_circular = np.ma.MaskedArray.compressed(masked_array_circular)
                label = masked_array_circular.mean()
            
                index = np.random.choice(len(fluxes))
                r50 = fluxes.loc[index, 'r50']
                r50 = r50 * 0.03 / 0.26
                I50 =  fluxes.loc[index, 'flux50']
                ser = fluxes.loc[index, 'sersic_n_gim2d']
                mag = fluxes.loc[index, 'I_auto']

                band  = self._internal_naming(filename)

                mod = Sersic2D(amplitude = I50, r_eff =r50, n=ser, x_0=60, y_0=60)
                galaxy = mod(xgrid, ygrid)

                postage = postage + galaxy

                postage[60-8:60 + 8, 60-8:60+8] = np.nan
                postages[i] = postage
         
                S = pd.Series()
                S['labels'] = label
                S['Iauto'] = mag 
                S['band'] =  band
                L.append(S)


            df = pd.concat(L, axis=1).T
            df = df.set_index(coord_pix.index)
            df['x'] = coord_pix.x + 60
            df['y'] = coord_pix.y + 60

      
        stamps = np.concatenate((stamps,postages), axis = 0)
        stamps_df = pd.concat((stamps_df,df))
        return postages, df


    def create_loader(self,stamps, df):

        data = TensorDataset(torch.Tensor(stamps),torch.Tensor(df.labels).unsqueeze(1),torch.Tensor(df.x).unsqueeze(1),torch.Tensor(df.y).unsqueeze(1), torch.LongTensor(df.band).unsqueeze(1), torch.Tensor(df.Iauto).unsqueeze(1))
        
        
        train_size = int(0.9 * len(data))
        val_size = len(data) - train_size
        train_dataset, val_dataset = random_split(data, [train_size, val_size])
        train_loader,val_loader= DataLoader(train_dataset, batch_size=100,shuffle=True),DataLoader(val_dataset, batch_size=100, shuffle=True)
        return train_loader, val_loader, train_size, val_size


    def train_phase(self,train_loader,val_loader,train_size, val_size):

        # TRAININ
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.cnn.parameters(), lr=0.00001)
        #exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size= 20, gamma=0.1)
        self.cnn.train()
        self.cnn.cuda()
        running_loss_train = 0.0 
        

        #exp_lr_scheduler.step()

        for k in range(80):
            print ('epoch',k)

            for bstamp, labels, x, y, band, Iauto, in train_loader:      
  
                optimizer.zero_grad()
    
                flat = bstamp.view(len(bstamp), -1)
                mean = torch.tensor(np.nanmean(flat, 1))
                std = torch.tensor(np.nanstd(flat, 1))

                bstamp = (bstamp - mean[:, None, None]) / \
                     std[:, None, None]
            
                # Removing the central region, as done in the training.
                bstamp[:, 60-8:60+8, 60-8:60+8] = 0
                bstamp = bstamp.unsqueeze(1)
             

                outputs_train = self.cnn(bstamp.cuda(),x.cuda(),y.cuda(),Iauto.cuda(),band.cuda())
                #print(outputs_train)

                
                sigma = torch.exp(outputs_train[:,1])
                bkg_pred = outputs_train[:,0]
                
                rerr = (bkg_pred - labels.squeeze(1).cuda())/sigma.cuda() 
                print(bkg_pred.shape, labels.shape,sigma.shape)
                print(rerr.shape)


                training_loss = rerr.pow(2) + 2*torch.log(sigma)
 
                training_loss = training_loss.mean()
 
        
                training_loss.backward()
                optimizer.step()

                running_loss_train += training_loss

                bstamp.cpu(), labels.cpu(), x.cpu(), y.cpu(),band.cpu(), Iauto.cpu()

            
            running_loss_train = running_loss_train/(train_size//len(bstamp))

            print('Training loss: {:.7f}'.format( running_loss_train))

            self.cnn.eval()

            running_loss_val = 0.0
      
            with torch.no_grad():
                for bstamp, labels, x, y, band, Iauto  in val_loader:
                    
                    flat = bstamp.view(len(bstamp), -1)
                    mean = torch.tensor(np.nanmean(flat, 1))
                    std = torch.tensor(np.nanstd(flat, 1))

                    bstamp = (bstamp - mean[:, None, None]) / \
                     std[:, None, None]
            
                    # Removing the central region, as done in the training.
                    bstamp[:, 60-8:60+8, 60-8:60+8] = 0
                    bstamp = bstamp.unsqueeze(1)
                    
 
                    outputs_val = self.cnn(bstamp.cuda(),x.cuda(),y.cuda(),Iauto.cuda(),band.cuda())
                    
                    sigma = torch.exp(outputs_val[:,1])
                    bkg_pred = outputs_val[:,0] 
                    print(bkg_pred.shape)
                    rerr = (bkg_pred - labels.squeeze(1).cuda())/sigma.cuda()
                    relerr = (bkg_pred.cpu() - labels.squeeze(1))/labels.squeeze(1) 

                    loss_val = rerr.pow(2) + 2*torch.log(sigma)	
                    loss_val = loss_val.mean() 

                    running_loss_val += loss_val
                       
                    bstamp.cpu(), labels.cpu(), x.cpu(), y.cpu(), band.cpu(), Iauto.cpu()
                    
                #print(rerr)
                print(rerr.detach().cpu().shape)
                running_loss_val = running_loss_val/(val_size//len(bstamp))
                print('Validation loss: {:.7f}'.format( running_loss_val))
                print('sig68:', self.sigma68(rerr.detach().cpu()))  
                print('sig68_rel', self.sigma68(relerr.detach().cpu())) 
     
                torch.save(self.cnn.state_dict(),os.path.join('models','model_COSMOS_%s.pt'%k))

        return 


    def train(self, directory):
            
        epochs = 50
        stamps, stamps_info = self.create_stamps(directory)
        train_loader,val_loader, train_size, val_size = self.create_loader(stamps, stamps_info)
        self.train_phase(train_loader,val_loader,train_size, val_size) 
        return 


    def diff(self, pred, true): return (pred-true)/true


    def sigma68(self,data): return 0.5*(pd.Series(data).quantile(q = 0.84) - pd.Series(data).quantile(q = 0.16))

