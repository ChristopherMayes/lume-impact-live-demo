#!/usr/bin/env python
# coding: utf-8

# # Live VCC image data -> distgen xy_dist file
# 
# See https://github.com/slaclab/lcls-lattice/blob/master/distgen/models/cu_inj/vcc_image/vcc_image.ipynb for a better explanation

# In[1]:


import epics
import numpy as np
import os

import h5py

from lcls_live.tools import isotime

import matplotlib.pyplot as plt

import logging
logger = logging.getLogger(__name__)


# In[2]:


# Nicer plotting
import matplotlib.pyplot as plt
get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'retina'")


# # EPICS tools

# In[3]:


def caget_dict(names):
    return dict(zip(names, epics.caget_many(names)))

def save_pvdata(filename, pvdata, attrs=None):
    
    logger.info(f"Writing {filename}")
    
    with h5py.File(filename, 'w') as h5:
        if attrs:
            for k, v in attrs.items():
                h5.attrs[k] = v
        for k, v in pvdata.items():
            h5[k] = v
            


# # Image tools

# In[4]:


from skimage.filters import sobel
from skimage.util import img_as_ubyte
from skimage.segmentation import watershed
from skimage.filters.rank import median
from skimage.morphology import disk

def isolate_image(img, fclip=0.08):
    """
    Uses a masking technique to isolate the VCC image
    """
    img=img.copy()
    
    # Clip lowest fclip fraction
    img[img < np.max(img)* fclip] = 0
    
    
    # Filter out hot pixels to use aas a mask
    # https://scikit-image.org/docs/0.12.x/auto_examples/xx_applications/plot_rank_filters.html
    img2 = median(img_as_ubyte(img), disk(2))
    
    elevation_map = sobel(img2)
    markers = np.zeros_like(img2)
    
    # TODO: tweak these numbers
    markers[img2 < .1] = 1
    markers[img2 > .2] = 2

    # Wateshed
    segmentation = watershed(elevation_map, markers)
    
    # Set to zero in original image
    img[np.where(segmentation != 2)]  = 0 
    
    # 
    ixnonzero0 = np.nonzero(np.sum(img2, axis=1))[0]
    ixnonzero1 = np.nonzero(np.sum(img2, axis=0))[0]
    
    i0, i1, j0, j1 = ixnonzero0[0], ixnonzero0[-1], ixnonzero1[0], ixnonzero1[-1]
    cutimg = img[i0:i1,j0:j1]
    
    return cutimg


# # PVs

# In[5]:


LCLS_VCC_PV = {
    
    'array':  'CAMR:IN20:186:IMAGE',
    'size_x': 'CAMR:IN20:186:N_OF_COL',
    'size_y': 'CAMR:IN20:186:N_OF_ROW',
    'resolution': 'CAMR:IN20:186:RESOLUTION',
    'resolution_units': 'CAMR:IN20:186:RESOLUTION.EGU'
    
    
}
#epics.caget_many(LCLS_VCC_PV.values())


# In[6]:


LCLS2_VCC_PV = {
    
    'array':  'CAMR:LGUN:950:Image:ArrayData',
    'size_x': 'CAMR:LGUN:950:Image:ArraySize0_RBV',
    'size_y': 'CAMR:LGUN:950:Image:ArraySize1_RBV',
    'resolution': 'CAMR:LGUN:950:RESOLUTION',
    'resolution_units': 'CAMR:LGUN:950:RESOLUTION.EGU'
}
#epics.caget_many(LCLS_VCC_PV.values())


# In[7]:


FACET_VCC_PV = {
    
    'array': 'CAMR:LT10:900:Image:ArrayData',
    'size_x': 'CAMR:LT10:900:ArraySizeX_RBV',
    'size_y': 'CAMR:LT10:900:ArraySizeY_RBV',
    'resolution': 'CAMR:LT10:900:RESOLUTION',
    'resolution_units': 'CAMR:LT10:900:RESOLUTION.EGU'
    
    
}

# Master dict
VCC_DEVICE_PV = {
    'CAMR:LGUN:950':LCLS2_VCC_PV,
    'CAMR:IN20:186':LCLS_VCC_PV,
    'CAMR:LT10:900':FACET_VCC_PV 
}

#epics.caget_many(FACET_VCC_PV.values())


# # Get

# In[8]:


def get_epics_vcc_data(epics, vcc_device, wait_for_good=True, good_std=4):
    """
     epics,
    wait_for_good: bool, default True
        will repeat epics.caget_many until the array data
        seems like an image
    """
    # Get actual PVs
    d = VCC_DEVICE_PV[vcc_device].copy()
    
    trials = 0
    
    if wait_for_good:  
        array_pvname = d.pop('array')

        found = False
        m = epics.PV(array_pvname)
        ii = 0
        while not found:
            ii += 1
            if ii % 10 == 0:
                print(f"Waited {ii} times for good {array_pvname}")
            trials += 1
            a = m.get()
            if a is None:
                continue
            if a.std() > good_std:
                found = True
                # Get regular pvs
                pvdata = caget_dict(d.values())                
                isotime_found = isotime()
                pvdata[array_pvname] = a
    else:
        pvdata = caget_dict(d.values())
        isotime_found = isotime()
        
    #out = {'pvdata': pvdata, 'isotime': isotime_found}

    return pvdata, isotime_found

#res = get_epics_vcc_data(epics, 'CAMR:LGUN:950', wait_for_good=True)
#res


# In[9]:


def vcc_image_data_from_pvdata(pvdata, vcc_device):
    """
    Process raw pvdata dict into image data
    """
    d = VCC_DEVICE_PV[vcc_device]
    
    image_data = {}
    for k, pvname in d.items():
        image_data[k] = pvdata[pvname]
        
    # Make consistent units
    if image_data['resolution_units'] == 'um/px':
        image_data['resolution_units'] = 'um'  
        
    a = image_data.pop('array')
    n = len(a)
    print(n)
    
    # Try to guess shape, because PVs are sometimes bad (None)
    if n == 1040 *  1392:
        shape = (1040 , 1392)
    elif n == 1024 * 1024:
        shape = (1024 , 1024)        
    else:
        shape = (image_data['size_y'], image_data['size_x'])
    
    
    
    image_data['image'] = a.reshape(shape)           
        
    return image_data


#vcc_image_data_from_pvdata(res[0],     'CAMR:LGUN:950')


# In[10]:


def get_vcc_data(epics, vcc_device, pvdata=None, wait_for_good=True, good_std=4, save_path=None):
    """
    
    wait_for_good: bool, default True
        will repeat epics.caget_many until the array data
        seems like an image
    """
    
    pvdata, isotime_found = get_epics_vcc_data(epics, vcc_device, wait_for_good=wait_for_good, good_std=good_std) 
    
    if save_path:
        assert os.path.exists(save_path)
        fname = os.path.join(save_path,  f"pvdata_{vcc_device}_{isotime_found}.h5")
        save_pvdata(fname, pvdata, attrs={'isotime':isotime_found})
    
    image_data = vcc_image_data_from_pvdata(pvdata, vcc_device)
    
    return image_data

#out = get_vcc_data(epics, 'CAMR:LGUN:950', save_path='vcc_archive')
#plt.imshow(out['image'])


# In[11]:


def write_distgen_xy_dist(filename, image, resolution, resolution_units='m'):
    """
    Writes image data in distgen's xy_dist format
    
    Returns the absolute path to the file written
    
    """
    
    # Get width of each dimension
    widths = resolution * np.array(image.shape)
    
    center_y = 0
    center_x = 0
    
    # Form header
    header = f"""x {widths[1]} {center_x} [{resolution_units}]
y {widths[0]} {center_y}  [{resolution_units}]"""
    
    # Save with the correct orientation
    np.savetxt(filename, np.flip(image, axis=0), header=header, comments='')
    
    return os.path.abspath(filename)


# In[12]:


def get_live_distgen_xy_dist(filename='test.txt', vcc_device='CAMR:IN20:186', pvdata=None, fclip=0.08):
    
    # Get data
    image_data = get_vcc_data(epics, vcc_device, pvdata)
    image = image_data['image']
    
    cutimg = isolate_image(image, fclip=fclip)
    
    assert cutimg.ptp() > 0
        
    fout = write_distgen_xy_dist(filename, cutimg,
                                 image_data['resolution'],
                                 resolution_units=image_data['resolution_units'])
    
    return fout, image, cutimg
    
# import matplotlib.pyplot as plt
# %config InlineBackend.figure_format = 'retina'
# fout, i1, i2 = get_live_distgen_xy_dist(vcc_device='CAMR:LGUN:950', fclip=0.08)
# # #fout, i1, i2 = get_live_distgen_xy_dist(vcc_device='CAMR:LT10:900', fclip=0.08)
# # 
# plt.imshow(i2)
# fout
# # 


# In[13]:


#!cp test.txt $LCLS_LATTICE/distgen/models/sc_inj/vcc_image/laser_image.txt


# In[14]:


# #gfile = os.path.expandvars('$FACET2_LATTICE/distgen/models/f2e_inj/vcc_image/distgen.yaml')
# gfile = os.path.expandvars('$LCLS_LATTICE/distgen/models/sc_inj/vcc_image/distgen.yaml')
# from distgen import Generator
# 
# G = Generator(gfile)
# G['xy_dist:file'] = fout
# G['n_particle'] = 100000
# G.run()
# G.particles.plot('x', 'y', bins=100, figsize=(5,5))


# In[15]:


# PVDATA = dict(zip(FACET_VCC_PV.values(), epics.caget_many(FACET_VCC_PV.values())))
# PVDATA


# In[16]:


# fout, i1, i2 = get_live_distgen_xy_dist(vcc_device='CAMR:LT10:900', pvdata=PVDATA)
# plt.imshow(i2)

