# Copyright (C) 2019 Laura Cabayol, Martin B. Eriksen
# This file is part of BKGnet <https://github.com/PAU-survey/bkgnet>.
#
# BKGnet is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# BKGnet is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with BKGnet.  If not, see <http://www.gnu.org/licenses/>.
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

from .network import CNN_model

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
             'NB455_1': 41,'NB465_1': 42,'NB475_1': 43,'NB485_1': 44, 'NB495_1': 45, 'NB505_1': 46, 'NB515_1': 47, 'NB525_1': 48, \
             'NB535_1': 49, 'NB545_1': 50, 'NB555_1': 51, 'NB565_1': 52, 'NB575_1': 53, 'NB585_1': 54, 'NB595_1': 55, \
             'NB605_1': 56, 'NB615_1': 64, 'NB625_1': 63, 'NB635_1': 62, 'NB645_1': 61, 'NB655_1': 60, 'NB660_1': 59, \
             'NB675_1': 58, 'NB685_1': 57, 'NB695_1': 72, 'NB705_1': 71, 'NB715_1': 70, 'NB725_1': 69, 'NB735_1': 68, \
             'NB745_1': 67,'NB755_1': 66, 'NB765_1': 65, 'NB775_1': 80, 'NB785_1': 79, 'NB795_1': 78, 'NB805_1': 77, \
             'NB815_1': 76, 'NB825_1': 75, 'NB835_1': 74, 'NB845_1': 73}

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
            postage = img_pad[int(sub.x):int(sub.x+120), int(sub.y):int(sub.y+120)].copy()
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
                outputs = self.cnn(bstamp, bx, by, bIauto, bband)
                #outputs = self.cnn(bstamp, bx, by, bmax_flux, bband, std)

            # The network gives the error in log(error)
            outputs[:,0] = std*outputs[:,0] + mean
            outputs[:,1] = std*torch.exp(outputs[:,1]) 

            pred.append(outputs) #.squeeze())
        
        pred = pd.DataFrame(torch.cat(pred).detach().numpy(), \
                            index=ps_info.index, columns=['bkg', 'bkg_error'])

        return pred

    def background_img(self, img, coords_pix, I_auto, band,exp_num):
        """Predict background using BKGnet."""

        stamps, ps_info = self.create_stamps(img, coords_pix)
    
        interv = '1' if exp_num >= 7582 else '0'            
         
        ps_info['band'] = self._internal_naming(band, interv) 
        ps_info['I_auto'] = I_auto
        pred = self._background_stamps(stamps, ps_info)

        return pred

