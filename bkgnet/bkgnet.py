#!/usr/bin/env python
# encoding: UTF8

# Force the CPU version to only use one thread. Needed for running
# at PIC, but also useful locally. There one can instead run multiple
# jobs in parallell.

import os
os.environ["MKL_NUM_THREADS"] = "1" 
os.environ["NUMEXPR_NUM_THREADS"] = "1" 
os.environ["OMP_NUM_THREADS"] = "1" 

import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader

# To be changed when knowing if the new model performs better.
from .new_network import CNN_model

class BKGnet:
    """Interface for background prection using neural networks."""
    
    # Here we estimate the background on CPUs. This should be much
    # simpler to integrate and sufficiently fast.
    def __init__(self, model_path, batch_size=50):
        
        # Load the model.
        cnn = CNN_model()
        cnn.load_state_dict(torch.load(model_path, map_location='cpu'))
        cnn.eval()
       
        self.batch_size = batch_size
        self.cnn = cnn
   
    def _internal_naming(self, band, intervention):
        """Converting to internal band numbering."""
        band = band + '_' + str(intervention)
        # Convention based on how the bands are laid out in the trays.
        D = {'NB455_0': 1,'NB465_0': 2,'NB475_0': 3,'NB485_0': 4, 'NB495_0': 5, 'NB505_0': 6, 'NB515_0': 7, 'NB525_0': 8, \
             'NB535_0': 9, 'NB545_0': 10, 'NB555_0': 11, 'NB565_0': 12, 'NB575_0': 13, 'NB585_0': 14, 'NB595_0': 15, \
             'NB605_0': 16, 'NB615_0': 24, 'NB625_0': 23, 'NB635_0': 22, 'NB645_0': 21, 'NB655_0': 20, 'NB660_0': 19, \
             'NB675_0': 18, 'NB685_0': 17, 'NB695_0': 32, 'NB705_0': 31, 'NB715_0': 30, 'NB725_0': 29, 'NB735_0': 28, \
             'NB745_0': 27,'NB755_0': 26, 'NB765_0': 25, 'NB775_0': 40, 'NB785_0': 39, 'NB795_0': 38, 'NB805_0': 37, \
             'NB815_0': 36, 'NB825_0': 35, 'NB835_0': 34, 'NB845_0': 33, \
             'NB455_0': 41,'NB465_0': 42,'NB475_0': 43,'NB485_0': 44, 'NB495_0': 45, 'NB505_0': 46, 'NB515_0': 47, 'NB525_0': 48, \
             'NB535_0': 49, 'NB545_0': 50, 'NB555_0': 51, 'NB565_0': 52, 'NB575_0': 53, 'NB585_0': 54, 'NB595_0': 55, \
             'NB605_0': 56, 'NB615_0': 64, 'NB625_0': 63, 'NB635_0': 62, 'NB645_0': 61, 'NB655_0': 60, 'NB660_0': 59, \
             'NB675_0': 58, 'NB685_0': 57, 'NB695_0': 72, 'NB705_0': 71, 'NB715_0': 70, 'NB725_0': 69, 'NB735_0': 68, \
             'NB745_0': 67,'NB755_0': 66, 'NB765_0': 65, 'NB775_0': 80, 'NB785_0': 79, 'NB795_0': 78, 'NB805_0': 77, \
             'NB815_0': 76, 'NB825_0': 75, 'NB835_0': 74, 'NB845_0': 73}

        # Just to avoid changing the dictionary.
        nr = D[band] - 1    

        return nr

    def create_stamps(self, img, coord_pix):
        """Create the postage stamps from positions given in pixels."""

        npos = len(coord_pix)
        postages = np.zeros(shape=(npos, 120, 120), dtype = np.float32)

        # Padding to allow stamps on the image edges.
        img_pad = np.pad(img, pad_width= 60 , mode='constant', constant_values=0)

        L = []
        for i, (ind, sub) in enumerate(coord_pix.iterrows()):
            # Remember to make a copy of the array.
            postage = img_pad[sub.x:sub.x+120, sub.y:sub.y+120].copy()
            postage = postage.astype(np.float32, copy = False)
            postage[60-8:60 + 8, 60-8:60+8] = np.nan
            postages[i] = postage

            S = pd.Series()
          
            L.append(S)
            
            
        df = pd.concat(L, axis=1).T
        df = df.set_index(coord_pix.index)
        df['x'] = coord_pix.x + 60
        df['y'] = coord_pix.y + 60
        
        return postages, df
    
    def _asdataset(self, postage_stamps, ps_info):
        """Convert to a dataset."""

        postage_stamps = torch.tensor(postage_stamps)
                
        ps_info = ps_info.astype(np.float32, copy=False)
        x = torch.tensor(ps_info.x.values).unsqueeze(1)
        y = torch.tensor(ps_info.y.values).unsqueeze(1) 
        I_auto = torch.tensor(ps_info.I_auto.values).unsqueeze(1)
     

        band = torch.tensor(ps_info.band.values).unsqueeze(1).type(torch.long)
        dset = TensorDataset(postage_stamps, x, y, I_auto, band)

        
        return dset
    
    def _background_stamps(self, postage_stamps, ps_info):
        """Determine the bakground for the postage stamps."""
        
        dset  = self._asdataset(postage_stamps, ps_info)
        loader = DataLoader(dset, batch_size=self.batch_size, \
                            shuffle=False)

        pred = []
        #for bstamp, bx, by, bmax_flux, bband in loader:
        for bstamp, bx, by, bIauto, bband in loader:
            # Normalizing postage-stamp by postage stamp.
            flat = bstamp.view(len(bstamp), -1)
            mean = torch.tensor(np.nanmean(flat, 1))
            std = torch.tensor(np.nanstd(flat, 1))

            bstamp = (bstamp - mean[:, None, None]) / \
                     std[:, None, None]
            
            # Removing the central region, as done in the training.
            bstamp[:, 60-8:60+8, 60-8:60+8] = 0
            bstamp = bstamp.unsqueeze(1)
        
            with torch.no_grad():
                outputs = self.cnn(bstamp, bx, by, br50, bIauto, bband, interv)
                #outputs = self.cnn(bstamp, bx, by, bmax_flux, bband, std)

            # The network gives the error in log(error)
            outputs[:,0] = std*outputs[:,0] + mean
            outputs[:,1] = std*torch.exp(outputs[:,1]) 

            pred.append(outputs) #.squeeze())
        
        pred = pd.DataFrame(torch.cat(pred).detach().numpy(), \
                            index=ps_info.index, columns=['bkg', 'bkg_error'])

        return pred

    def background_img(self, img, coords_pix, I_auto, band):
        """Predict background using BKGnet."""

        stamps, ps_info = self.create_stamps(img, coords_pix)
        if ps_info.exp_num >= 7582:
            interv = 1
        if ps_info.exp_num < 7582:
            interv = 0
        ps_info['band'] = self._internal_naming(ps_info.band, interv) 
        ps_info['I_auto'] = I_auto
        pred = self._background_stamps(stamps, ps_info)

        return pred
