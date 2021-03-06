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
# Not to be integrated! Only for displaying the integration
# example.
import os
import pandas as pd
import numpy as np
import re
from astropy import wcs

def get_band(filename):
    """Mapping from file name to band name."""
    
    # Don't use this. Provide your own. Please.
    bands = pd.read_table('bands.csv', sep = ',', header = 0)
    NB = np.asarray(re.findall("NB(\d+).", filename))
    CCD = np.asarray(re.findall("std.(\d+).", filename))
    join,nbs = '_', 'NB'

    code = nbs+ NB[0]+join+ nbs+ NB[1]+ join + CCD[0]
    for k in range(bands.shape[0]):
        if bands.loc[k,'id'] == code:
            band = bands.loc[k,'band']
        
    return band

def get_expnum(filename):
    """Extract exposure number from file name."""
    
    fname = os.path.basename(filename)
    nr = int(fname.split('.')[1])
    
    return nr

def load_cosmos(cosmos_path):
    """Load the cosmos catalogue."""
    
    # Selecting bright objects for a small test
    cat = pd.read_table(cosmos_path, delimiter = ',', \
                        header = 0, comment = '#')
    
    #cat = cat[cat.I_auto < 19]
    coords = cat[['ref_id', 'ra', 'dec','I_auto','exp_num']]
    #coords = coords.rename(columns={'paudm_id': 'ref_id'})
    coords = coords.set_index('ref_id')
    
    return coords

def get_pixelpos(coords, header):
    """Get the pixel position."""
    
    # @Santi. It would be better if you can provide exactly the same
    # positions as used for the photometry. 
    w = wcs.WCS(header)
    footprint = w.calc_footprint(header)
    coords_inside = coords[(coords.ra>np.amin(footprint[:,0]))&(coords.ra<np.amax(footprint[:,0]))&\
                          (coords.dec>np.amin(footprint[:,1]))&(coords.dec<np.amax(footprint[:,1]))]
    
        
    pix_coords = w.wcs_world2pix(coords_inside.loc[:, ['ra','dec']],0)
    
    # @Santi, do you keep coordinates as floats or ints?
    pix_coords = pix_coords.astype(np.int)
    pix_coords = pd.DataFrame(pix_coords, columns=['y', 'x'], index=coords_inside.index)
    
    return pix_coords
