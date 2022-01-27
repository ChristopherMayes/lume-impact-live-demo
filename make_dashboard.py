#!/usr/bin/env python
# coding: utf-8

# # Dashboard creation routines

# In[1]:


from impact import Impact
from distgen import Generator
import os
import json
import numpy as np

from pathlib import Path

import matplotlib as mpl
#mpl.use('Agg')
import matplotlib.pyplot as plt
plt.style.use('dark_background')


# In[2]:


from PIL import Image, ImageOps, ImageEnhance 

def fig2data ( fig ):
    """
    @brief Convert a Matplotlib figure to a 4D numpy array with RGBA channels and return it
    @param fig a matplotlib figure
    @return a numpy 3D array of RGBA values
    """
    # draw the renderer
    fig.canvas.draw ( )
    # Get the RGBA buffer from the figure
    w,h = fig.canvas.get_width_height()
    buf = np.frombuffer ( fig.canvas.tostring_argb(), dtype=np.uint8 )
    buf.shape = ( w, h, 4 )
    # canvas.tostring_argb give pixmap in ARGB mode. Roll the ALPHA channel to have it in RGBA mode
    buf = np.roll ( buf, 3, axis = 2 )
    return buf

def fig2img ( fig ):
    """
    @brief Convert a Matplotlib figure to a PIL Image in RGBA format and return it
    @param fig a matplotlib figure
    @return a Python Imaging Library ( PIL ) image
    """
    # put the figure pixmap into a numpy array
    buf = fig2data ( fig )
    w, h, d = buf.shape
    return Image.frombytes( "RGBA", ( w ,h ), buf.tobytes( ) )


# In[3]:


def iscreen(impact_object, screen='OTR2', k1='x', k2='y', dpi=72, title=None):
    fig = impact_object.particles[screen].plot(k1, k2, return_figure=True, figsize=(5,4))
    fig.dpi=dpi
    
    if not title:
        title = screen
    fig.axes[2].set_title(title)
    fig.tight_layout()
    return fig2img(fig)


# In[4]:


def info_str(impact_object, name=''):
    I = impact_object
    H = impact_object.header
    P = I.particles['initial_particles']
    
    timestep = I.ele['change_timestep_1']
    dt1 = timestep['dt']
    s1 = timestep['s']
    run_time = I.output['run_info']['run_time'] 

    
    summary=f"""{name} 
    
LUME-Impact running Impact-T 
Distgen created particles at the cathode
Particles in openPMD-beamphysics format 

{H['Np']:,} macroparticles
{H['Nbunch']} bunch of {I['species']}s
total charge: {I['total_charge']*1e12:.1f} pC
Processor domain: {H['Nprow']} x {H['Npcol']} = {H['Nprow']*H['Npcol']} CPUs
Space charge grid: {H['Nx']} x {H['Ny']} x {H['Nz']}
Timestep: {H['Dt']*1e12} ps to {s1} m, 
          then {dt1*1e12} ps until the end
                  
Run time: {run_time/60:.1f} min

"""
    return summary

def itext(impact_object, dpi=72, name=''):
    text = info_str(impact_object, name=name)
    fig, ax = plt.subplots(figsize=(5,4))
    fig.dpi=dpi
    fig.tight_layout()
    ax.set_axis_off()
    ax.text(0.1, 0.5, text, fontsize=13, horizontalalignment='left', verticalalignment='center', transform=ax.transAxes)
    return fig2img(fig)


# In[5]:


def make_dashboard(impact_object=None,
                   dat=None,
                   itime=None,
                   outpath='test/',
                   screen1='YAG02',
                   screen2='YAG03',
                   screen3='OTR2',
                   ylim=(0,2e-6),
                   ylim2=(0,2e-3),
                   name='lume-impact-live-demo'
                  ):
    """
    Makes a composite dashboard image from data dict
    
    Returns the path to the figure written
    """
    if impact_object:
        I = impact_object   
    else:
        itime = dat['isotime']
        I = Impact.from_archive(dat['outputs']['archive'])
        #G = Generator()
        #G.load_archive(dat['archive'])
    #return I # Debug
    
    run_time = I.output['run_info']['run_time'] 
    # Main figure
    FIG0 = I.plot(['norm_emit_x','norm_emit_y'], 
       y2=['sigma_x', 'sigma_y', 'sigma_z'],
       
       ylim=ylim, ylim2=ylim2,
        figsize=(16,8), return_figure=True)
    
    
    n_particle = I.particles['final_particles'].n_particle
    
    title=f'Acquired settings at {itime}, simulation run time: {run_time/60:5.1f} min'
    
    FIG0.tight_layout()
    FIG0.axes[0].set_title(title)
    
    DPI = 150 # test
    FIG0.dpi=DPI
    im0 = fig2img(FIG0)
    
    
    # For short debugging runs
    if screen1 not in I.particles:
        screen1='initial_particles'
        screen2='initial_particles'
        screen3='final_particles'

        
    # info text
    #imtext = ImageOps.invert(itext(I, dpi=DPI).convert('RGB'))     
    imtext =itext(I, dpi=DPI, name=name)

    im1 = iscreen(I, screen=screen1, k1='x', k2='y', dpi=DPI)
    im2 = iscreen(I, screen=screen2, k1='x', k2='y', dpi=DPI)
    #im3 = iscreen(I, screen=screen3, k1='x', k2='y', dpi=DPI)
    im3 = imtext
    im4 = iscreen(I, screen=screen3, k1='delta_z', k2='delta_energy', dpi=DPI)
    im5 = iscreen(I, screen=screen3, k1='x', k2='y', dpi=DPI)
    
    im99 = iscreen(I, screen='initial_particles', k1='x', k2='y', dpi=DPI, title='cathode')
    
    SIZE =  (im0.width + im1.width, im1.height+im2.height+im3.height)
    ii = Image.new('RGB', SIZE)
    
    invim0 = ImageOps.invert(im0.convert('RGB'))
    ii.paste(im0, (0, 10))
    
    ii.paste(im99, (0, im0.height))
    ii.paste(im1, (im99.width, im0.height))
    ii.paste(im2, (im99.width+im1.width,im0.height))
    ii.paste(im3, (im0.width,0))
    ii.paste(im4, (im0.width,im3.height))
    ii.paste(im5, (im0.width,im4.height+im3.height))
    
    fname = f'{name}-{itime}-dashboard.png'
    fout = os.path.join(outpath, fname)
    
    # Enhance contrast
    #enhancer = ImageEnhance.Brightness(ii) 
    enhancer = ImageEnhance.Contrast(ii) 
    iout = enhancer.enhance(1.2)
    iout.save(fout)
    
    return fout
    


# In[7]:


#%%capture
#I0 = make_dashboard(dat=json.load(open('output/lume-impact-live-demo-2021-04-05T19:13:18-07:00.json')))

