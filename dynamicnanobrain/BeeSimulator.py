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

def decode_position(cpu4_reshaped, cpu4_mem_gain):
    """Decode position from sinusoid in to polar coordinates.
    Amplitude is distance, Angle is angle from nest outwards.
    Without offset angle gives the home vector.
    Input must have shape of (2, -1)"""
    signal = np.sum(cpu4_reshaped, axis=0)
    # coefficient c1 for the fundamental frequency
    fund_freq = np.fft.fft(signal)[1]
    #angle = -np.angle(np.conj(fund_freq))
    # add pi to account for TB1_1 being at np.pi
    angle = np.pi - np.angle(fund_freq)
    
    distance = np.absolute(fund_freq) / cpu4_mem_gain
    return angle, distance

def decode_cpu4(cpu4, cpu4_mem_gain):
    """Shifts both CPU4 by +1 and -1 column to cancel 45 degree flow
    preference. When summed single sinusoid should point home."""
    cpu4_reshaped = cpu4.reshape(2, -1)
    cpu4_shifted = np.vstack([np.roll(cpu4_reshaped[0], 1),
                              np.roll(cpu4_reshaped[1], -1)])
    
    return decode_position(cpu4_shifted, cpu4_mem_gain)

def shift_cpu4(cpu4):
    """Shifts both CPU4 by +1 and -1 column to cancel 45 degree flow
    preference. When summed single sinusoid should point home."""
    cpu4_reshaped = cpu4.reshape(2, -1)
    cpu4_shifted = np.vstack([np.roll(cpu4_reshaped[0], 1),
                              np.roll(cpu4_reshaped[1], -1)])
    
    return np.sum(cpu4_shifted/2,axis=0)

#%%
def analyse(N, param_dict, radius=20):
    """Iterates through param dictionary, running batches of trials according to the param dictionary"""
    min_dists =[]
    min_dist_stds = []
    search_dists=[]
    search_dist_stds=[]
    
    for i in range(param_dict['n']):
        kwargs = {}
        for k, v in param_dict.items():
            if k != 'n' and v[i] != None:
                kwargs[k] = v[i]
                        
        OUT, INB = trials.generate_dataset(N=N, **kwargs)
        
        if 'T_outbound' in kwargs:
            T_outbound = kwargs['T_outbound']
        else:
            T_outbound = 1500
        if 'T_inbound' in kwargs:
            T_inbound = kwargs['T_inbound']
        else:
            T_inbound = 1500

        # Closest position to nest within return part. Use my new one 
        min_d, min_d_sigma, search_d, search_d_sigma= trials.analyze_inbound(INB, T_outbound,T_inbound)

        # Check also disk leaving angle relative to "true" angle
        disc_leaving_angles = trials.analyze_angle(INB,radius)
        
        min_dists.append(min_d)
        min_dist_stds.append(min_d_sigma)
        search_dists.append(search_d)
        search_dist_stds.append(search_d_sigma)
    
    return min_dists, min_dist_stds, search_dists, search_dist_stds, disc_leaving_angles

def analyse2(N, param_dict, radius=20):
    """Iterates through param dictionary, running batches of trials according to the param dictionary"""
    d_angles =[]
    d_lens = []
    path_straightness=[]
    
    for i in range(param_dict['n']):
        kwargs = {}
        for k, v in param_dict.items():
            if k != 'n' and v[i] != None:
                kwargs[k] = v[i]
                        
        OUT, INB = trials.generate_dataset(N=N, **kwargs)

        # Compare CPU4 memory with turning point
        # Turning point
        x_turning_point = INB['x'].loc[:,0]
        y_turning_point = INB['y'].loc[:,0]
        tp_len = np.sqrt(x_turning_point**2 + y_turning_point**2)
        tp_angle = np.arctan2(y_turning_point,x_turning_point)
        # From CPU4
        #cpu4 = trials.get_cpu4activity(OUT)
        #cpu4_angle, cpu4_len = trials.decode_cpu4(cpu4,0.0005)
        # Differences
        d_angle = tp_angle #- cpu4_angle
        d_len = tp_len #- cpu4_len
    
        # Calculate tortuosity
        straight = trials.compute_path_straightness(INB)

        # Check also disk leaving angle relative to "true" angle
        disc_leaving_angles = trials.analyze_angle(INB,radius)
        
        d_angles.append(d_angle)
        d_lens.append(d_len)
        path_straightness.append(straight)
    
    return d_angles, d_lens, path_straightness, disc_leaving_angles
#%% Here we specify the arguments of interest and generate the bulk of data
# needed for the analysis. In addition to saving the minimal distance to the 
# nest during the inbound flight, we also keep track of the size of the search
# pattern, perhaps this will prove interesting. 

N = 10 # number of trials for each parameter to test
distances = [200, 400, 750, 1500, 3000, 6000]
#distances = np.round(10 ** np.linspace(2, 4, N_dists)).astype('int')
N_dists= len(distances)

# List of parameters that have been studied (keeping for reference)
memupdate_vals = [0.0005,0.001,0.002,0.004]#, 0.005, 0.0025, 0.0050, 0.01]
inputscale_vals = [0.7, 0.8, 0.9, 1.0]
meminitc_vals = [0.0,0.25,0.5]
cpushift_vals = [0.0, -0.2, -0.4, -0.6]
memupdate_vals = [0.00025, 0.0005, 0.001, 0.002]
Vt_noise_vals = [0.01, 0.02, 0.05]
reduced_a = 0.07

# Specify the dict of parameters
param_dicts = [{'n':N_dists,'T_outbound': distances, 'T_inbound': distances}]

# List of dictionaries
#param_dicts = [{'n':N_dists, 'a':[reduced_a]*N_dists, 'Vt_noise': [noise]*N_dists, 'T_outbound': distances, 'T_inbound': distances} for noise in Vt_noise_vals]
#
# This dictionary specifies an earlier run that I kept the data from
param_dict_ref = {'n':N_dists,'T_outbound': distances, 'T_inbound': distances}

min_dists_l = []
min_dist_stds_l = []
search_dists_l=[]
search_dist_stds=[]
    
for param_dict in param_dicts:
    min_dists, min_dist_stds , search_dist, search_dist_std, _ = analyse(N, param_dict)
    min_dists_l.append(min_dists)
    min_dist_stds_l.append(min_dist_stds)
    search_dists_l.append(search_dist)
    search_dist_stds.append(search_dist_std)
    
#%% Add reference dict to lists
min_dists, min_dist_stds , search_dist, search_dist_std, _ = analyse(100, param_dict_ref)
min_dists_l.insert(0,min_dists)
min_dist_stds_l.insert(0,min_dist_stds)
search_dists_l.insert(0,search_dist)
search_dist_stds.insert(0,search_dist_std)

#%% Use old data
N_dists=1
reduced_a = 0.07
distances=[1500]*N_dists
param_dict = {'n':N_dists, 'a':[reduced_a]*N_dists, 'turn_noise':[0.2]*N_dists, 'T_outbound': distances, 'T_inbound': distances, 'tn2_scaling':[2.0]*N_dists}
d_angles, d_lens, path_straightness, disc_leaving_angles = analyse2(100, param_dict)

#%% Generate data set to test noise in inbound flight
N_dists=1
reduced_a = 0.07
distances=[1500]*N_dists
param_dict = {'n':N_dists, 'a':[reduced_a]*N_dists, 'memupdate':[0.0005]*N_dists, 'turn_noise':[0.2]*N_dists, 'T_outbound': [2]*N_dists, 'T_inbound': distances,
              'tn2scaling':[2.0]*N_dists}
param_dict = {'n':N_dists, 'a':[reduced_a]*N_dists, 'turn_noise':[0.2]*N_dists, 'T_outbound': distances, 'T_inbound': distances, 'tn2scaling':[2.0]*N_dists}

min_dists, min_dist_stds , search_dist, search_dist_std, _ = analyse(100, param_dict)

#%% Calcualte tortuosity as a figure of merit
cum_min_dist = path_straightness[0]
tortuosity  = trials.compute_tortuosity(path_straightness[0])

mu = np.nanmean(cum_min_dist, axis=1)
std = np.nanstd(cum_min_dist, axis=1)

import matplotlib.pyplot as plt
plt.plot(mu)
plt.plot(mu+std,'k--')
plt.plot(mu-std,'k--')

plt.show()

#%%
disc_leaving_angles = []
for param_dict in param_dicts:
    _ , _ , _ , _ , disc_leaving_angle = analyse(N, param_dict)
    disc_leaving_angles.append(disc_leaving_angle)
    
_, _, _, _, disc_leaving_angle = analyse(100, param_dict_ref)  
disc_leaving_angles.insert(0,disc_leaving_angle)

fig, ax = beeplotter.plot_angular_distances([0.0]+Vt_noise_vals,disc_leaving_angles,scale=[0.5,1.0,1.0,1.0])
fig.show()

#%% Produce a plot of the success of the specific parameters over distance. 
# A label for the parameter can be sent just after the parameter values
fig, ax = beeplotter.plot_distance_v_param(min_dists_l, min_dist_stds_l, distances, memupdate_vals, 'Memory update',ymax=150,xmin=100,xmax=4000)
fig, ax = beeplotter.plot_distance_v_param(min_dists_l, min_dist_stds_l, distances, meminitc_vals, 'Memory init.',ymax=150,xmin=100,xmax=4000)
fig, ax = beeplotter.plot_distance_v_param(min_dists_l, min_dist_stds_l, distances, inputscale_vals, 'Inputscaling',ymax=150,xmin=100,xmax=4000)
fig, ax = beeplotter.plot_distance_v_param(min_dists_l, min_dist_stds_l, distances, cpushift_vals, 'Cpu1/4 shift',ymax=150,xmin=100,xmax=8000)
fig, ax = beeplotter.plot_distance_v_param(min_dists_l, min_dist_stds_l, distances, memupdate_vals, 'Memory update',ymax=150,xmin=100,xmax=8000)
fig, ax = beeplotter.plot_distance_v_param(min_dists_l, min_dist_stds_l, distances, [0.0]+Vt_noise_vals, 'Vt noise',ymax=200,xmin=100,xmax=10000)
fig.show()

#%% Produce a plot showing the search pattern width. Here we adjust the 
# ylabel using an optional variable
fig, ax = beeplotter.plot_distance_v_param(search_dists_l, search_dist_stds, distances, memupdate_vals, 'Memory update', xmin=100, xmax=4000, ylabel='Search pattern width')
fig, ax = beeplotter.plot_distance_v_param(search_dists_l, search_dist_stds, distances, meminitc_vals, 'Memory init.', xmin=100, xmax=4000, ylabel='Search pattern width')
fig, ax = beeplotter.plot_distance_v_param(search_dists_l, search_dist_stds, distances, inputscale_vals, 'Inputscaling', xmin=100, xmax=4000, ylabel='Search pattern width')
fig, ax = beeplotter.plot_distance_v_param(search_dists_l, search_dist_stds, distances, cpushift_vals, 'Cpu1/4 shift', xmin=100, xmax=8000, ylabel='Search pattern width')
fig, ax = beeplotter.plot_distance_v_param(search_dists_l, search_dist_stds, distances, memupdate_vals, 'Memory update', xmin=100, xmax=8000, ylabel='Search pattern width')

fig.show()

#%% Single flight can be generated like this
T=100
OUT, INB = trials.generate_dataset(T,T,1)

# Output is after statistical analysis (mean and std)
min_dist, min_dist_std, search_dist, search_dist_std = trials.analyze_inbound(INB,T,T)

#%% Or single flight can be generated like this to get the network instance
my_nw = trials.setup_network(memupdate=0.001) 
out_res, inb_res, out_travel, inb_travel, overshoots = trials.run_trial(my_nw,500,1000,a=0.07,turn_noise=0.2,straight_route=True,offset=np.pi/4)                                                         

#%%
my_nw = trials.setup_network(memupdate=0.001) 
out_res, inb_res, out_travel, inb_travel, overshoots = trials.run_trial(my_nw,750,1500,a=0.1,mem_init_c=0.15)                                                         

#%% Visualize devices and weights
plt.ion()
my_nw.show_weights()
my_nw.show_devices(Vleak_dict={})

#%%
import pandas as pd
res = pd.concat([out_res,inb_res],ignore_index=True)
# Costruct a pseudo-layer
columns = {}
attr = 'Pout'
for layer in ['CPU4','Pontine'] :
    columns[layer] = [name for name in res.columns if (attr in name) and (layer in name)]

res[columns['CPU4']].shape

### Plotting also the positive input to the CPU1 layer
# Allocate
toCPU1 = np.zeros(res[columns['CPU4']].shape)
# Find weights
W_CPU4_CPU1a = my_nw.weights['CPU4->CPU1a'].W
W_Pontin_CPU1a = my_nw.weights['Pontine->CPU1a'].W
W_CPU4_CPU1b = my_nw.weights['CPU4->CPU1b'].W
W_Pontin_CPU1b = my_nw.weights['Pontine->CPU1b'].W

toCPU1[:,:14] +=  res[columns['CPU4']] @ W_CPU4_CPU1a.T
toCPU1[:,14:] +=  res[columns['CPU4']] @ W_CPU4_CPU1b.T

toCPU1[:,:14] -=  res[columns['Pontine']] @ W_Pontin_CPU1a.T
toCPU1[:,14:] -=  res[columns['Pontine']] @ W_Pontin_CPU1b.T
column_names=[f'CPUin_{a}-Pout' for a in range(0,16)]

df = pd.DataFrame(toCPU1,columns=column_names)
res = pd.concat([res,df],axis=1)

# Add also the motor
motor = np.zeros((res[columns['CPU4']].shape[0],2)) # left, right

columns = {}
attr = 'Pout'
for layer in ['CPU1a','CPU1b'] :
    columns[layer] = [name for name in res.columns if (attr in name) and (layer in name)]

W_CPU1a_motor = np.array([
                [1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1]])
W_CPU1b_motor = np.array([[0, 1],
                          [1, 0]])

motor += res[columns['CPU1a']] @ W_CPU1a_motor.T
motor += res[columns['CPU1b']] @ W_CPU1b_motor.T

scale_factor = 0.0005
motor_out = (motor[0]-motor[1]) + np.random.normal(scale=0.1*200, size=len(motor[0]))

column_names = ['motor-R', 'motor-L']
motor.columns = column_names
#df = pd.DataFrame(motor,columns=column_names)
res = pd.concat([res,motor],axis=1)
#r_right = sum(layers['CPU1a'].P[:7]) + layers['CPU1b'].P[1]
#r_left  = sum(layers['CPU1a'].P[7:]) + layers['CPU1b'].P[0]

# Now let's plot this guy
#fig,_ = beeplotter.plot_traces(res, layers=['CL1','TB1','TN2','CPU4','CPUin','CPU1'],attr='Pout',titles=True)
fig,_ = beeplotter.plot_traces(res, layers=['CL1','TB1','TN2','Rectifier','CPU4','CPU1','motor'],attr='Pout',titles=True)

trials.one_flight_results(out_res, inb_res, out_travel, inb_travel,'test',interactive=True,cpu4_mem_gain=0.0005,radius=20,show_headings=True)

#%%
cum_min_dist = analysis.compute_path_straightness(V, T_outbound)

fig, ax = plotter.plot_route_straightness(cum_min_dist)
if save_figs:
    plotter.save_plot(fig, 'path_straightness')
    
tort = analysis.compute_tortuosity(cum_min_dist)
print("Tortoisity for mean homebound path up to 1 route length of steps is", tort)

# %% [markdown]
# We can calculate the tortuosity using the most basic formula: $$\tau = L / C$$

#%% Plot speed signals

def absvel(df) :
    vx = df['vx']
    vy = df['vy']
    v = np.sqrt(vx**2 + vy**2)
    return v

v_out = absvel(out_travel)
v_in = absvel(inb_travel)
import matplotlib.pyplot as plt
fig, ax = plt.subplots()
ax.plot(out_travel['Time'],v_out)
ax.plot(inb_travel['Time'],v_in)

print('Average velocity (outbound):',v_out.mean())



#%% Plot CPU4 memory bump
cpu4 = trials.get_cpu4activity(out_res)

plt.ion()
beeplotter.plot_homing_dist(cpu4, shift_cpu4(cpu4))
fig, ax = plt.subplots()
ax.plot(cpu4[:8],label='[:8]',color='red')
ax.plot(cpu4[8:],label='[8:]',color='orange')
#ax.plot(np.roll(cpu4[:8],1),'--',label='[:8]',color='red')
#ax.plot(np.roll(cpu4[8:],-1),'--',label='[8:]',color='orange')
#ax.plot(0.5*(np.roll(cpu4[:8],1)+np.roll(cpu4[8:],-1)),'.-',color='black')
#ax.plot(shift_cpu4(cpu4),label='summed+shift')
ax.legend()

angle, distance = decode_cpu4(cpu4, my_nw.mem_update_h)

#%% Extra diagnostics to check out the CPU1 neurons
# When plotting in a notebook, use the onecolumn flag as below.
import dynamicnanobrain.core.plotter as plotter

TB1_list = [f'TB1_{idx}' for idx in range(0,8)]
plotter.plot_nodes(out_res, nodes=TB1_list, onecolumn=True)

CPU4_list = [f'CPU4_{idx}' for idx in range(0,16)]
plotter.plot_nodes(out_res, nodes=CPU4_list)

Pontine_list = [f'Pontine_{idx}' for idx in range(0,16)]
plotter.plot_nodes(inb_res,nodes=Pontine_list)

CPU1a_list = [f'CPU1a_{idx}' for idx in range(0,14)]
CPU1b_list = ['CPU1b_0','CPU1b_1']
CPU1_list = CPU1a_list + CPU1b_list


plotter.plot_nodes(inb_res, nodes=CPU1_list)
