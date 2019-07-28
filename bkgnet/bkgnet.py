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
   
    def _internal_naming(self, band):
        """Converting to internal band numbering."""
       
        # Convention based on how the bands are laid out in the trays.
        D = {'NB455': 1,'NB465': 2,'NB475': 3,'NB485': 4, 'NB495': 5, 'NB505': 6, 'NB515': 7, 'NB525': 8, \
             'NB535': 9, 'NB545': 10, 'NB555': 11, 'NB565': 12, 'NB575': 13, 'NB585': 14, 'NB595': 15, \
             'NB605': 16, 'NB615': 24, 'NB625': 23, 'NB635': 22, 'NB645': 21, 'NB655': 20, 'NB665': 19, \
             'NB675': 18, 'NB685': 17, 'NB695': 32, 'NB705': 31, 'NB715': 30, 'NB725': 29, 'NB735': 28, \
             'NB745': 27,'NB755': 26, 'NB765': 25, 'NB775': 40, 'NB785': 39, 'NB795': 38, 'NB805': 37, \
             'NB815': 36, 'NB825': 35, 'NB835': 34, 'NB845': 33}

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
            S['max_flux'] = np.nanmax(postage)
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
        r50 = torch.tensor(ps_info.r50.values).unsqueeze(1)
        I_auto = torch.tensor(ps_info.I_auto.values).unsqueeze(1)

        band = torch.tensor(ps_info.band.values).unsqueeze(1).type(torch.long)
        interv = torch.tensor(ps_info.interv.values).unsqueeze(1)
        
#        max_flux = torch.tensor(ps_info.max_flux.values).unsqueeze(1)
        dset = TensorDataset(postage_stamps, x, y, r50, I_auto, band, interv)

#        for bstamp, bx, by, br50, bIauto, bband, interv in loader:
#        dset = TensorDataset(postage_stamps, x, y, max_flux, band)
        
        return dset
    
    def _background_stamps(self, postage_stamps, ps_info):
        """Determine the bakground for the postage stamps."""
        
        dset = self._asdataset(postage_stamps, ps_info)
        loader = DataLoader(dset, batch_size=self.batch_size, \
                            shuffle=False)

        pred = []
        #for bstamp, bx, by, bmax_flux, bband in loader:
        for bstamp, bx, by, br50, bIauto, bband, interv in loader:
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

    def background_img(self, img, coords_pix, r50, I_auto, band, interv):
        """Predict background using BKGnet."""

        stamps, ps_info = self.create_stamps(img, coords_pix)
        ps_info['band'] = self._internal_naming(band)
        ps_info['interv'] = interv
        ps_info['r50'] = r50
        ps_info['I_auto'] = I_auto
        pred = self._background_stamps(stamps, ps_info)

        return pred
