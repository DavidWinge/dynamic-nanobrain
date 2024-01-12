#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  7 11:25:20 2021

@author: dwinge
"""

import numpy as np

# %% Setup the analyze function that runs N trials for each of the args specified in 
# kwargs, effectively iteration through a list of arguments. 

# Note on relative imports, we have to load these things on the main package 
# level otherwise the relative imports won't work
import dynamicnanobrain.beesim.trialflight as trials 
import dynamicnanobrain.beesim.beeplotter as beeplotter

#%%
#my_nw = trials.setup_network(memupdate=0.05,memscale=10000.,pon_cpu1_m=0.0) 
#my_nw = trials.setup_network(memupdate=0.75,memscale=160000.,Vt_noise=0.05)
my_nw = trials.setup_network(memupdate=0.75,memscale=160000.)
#out_res, inb_res, out_travel, inb_travel = trials.run_trial(my_nw,3000,3000,a=0.1,hupdate=4e-4, noise=0.05)                                                         
#out_res, inb_res, out_travel, inb_travel = trials.run_trial(my_nw,1000,2500,a=0.08,straight_route=True, noise=0.05,hupdate=2e-4, bias_scaling=0.1, mem_init_c=0.3) 
scale=0.3
out_res, inb_res, out_travel, inb_travel = trials.run_trial(my_nw,20,40,noise=0.2,printtime=True,tn2scaling=scale,tb1scaling=scale) 

