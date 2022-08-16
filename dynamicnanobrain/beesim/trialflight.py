#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 10 13:20:21 2022

Many of the methods have been adapted from Stone et al. 2017

@author: dwinge
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os
import pickle
from scipy.interpolate import interp1d
from pathlib import Path

from . import pathtrials
from . import beeplotter
from . import stone
from ..core import physics
from ..core import plotter

# These paths are a bit tricky, as they will be relative to the importing file
# In the current case, they become relative to BeeSimulator one folder above
DATA_PATH= Path('../data/beesim/')
PLOT_PATH= Path('../plots/beesim/')
default_device_file = Path('../parameters/device_parameters_1ns.txt')

def compute_mean_tortuosity(cum_min_dist):
    """Computed with tau = L / C."""
    mu = np.nanmean(cum_min_dist, axis=1)
    tortuosity = 1.0 / (1.0 - mu[len(mu)//2])
    return tortuosity

def compute_tortuosity(cum_min_dist):
    """Computed with tau = L / C."""
    L_index = len(cum_min_dist)//2
    tortuosity = 1.0 / (1.0 - cum_min_dist[L_index])
    return tortuosity

def compute_path_straightness(INB):

    # Loop over each sample
    N = INB['x'].index[-1][0]+1 # should choose last index
    maxlen = INB['x'].index.max()[1]+1
    dist_from_nest = np.zeros((maxlen,N))
    for k in range(0,N) :
        
        dist_from_nest[:,k] = np.sqrt((INB['x'].loc[k])**2 +
                                      (INB['y'].loc[k])**2)

    turn_dists = dist_from_nest[0]
    # Get shortest distance so far to nest at each time step
    # We make the y axis equal, by measuring in terms of proportion of
    # route distance.
    cum_min_dist = np.minimum.accumulate(dist_from_nest / turn_dists)

    # Get cumulative speed
    #cum_speed = np.cumsum(np.sqrt((V[:, T_outbound:, 0]**2 + V[:, T_outbound:, 1]**2)), axis=1)
    cum_speed = np.zeros((maxlen,N))
    for k in range(0,N) :
        cum_speed[:,k] = np.cumsum(np.sqrt((INB['vx'].loc[k])**2 +
                                           (INB['vy'].loc[k])**2))

    # Now we also make the x axis equal in terms of proportion of distance
    # Time is stretched to compensate for longer/shorter routes
    cum_min_dist_norm = []
    for i in np.arange(N):
        t = cum_speed[:,i]
        xs = np.linspace(0, turn_dists[i]*2, 500, endpoint=False)
        cum_min_dist_norm.append(np.interp(xs,
                                           t,
                                           cum_min_dist[:,i]))
    return np.array(cum_min_dist_norm).T

def angular_distance(a, b):
    return (a - b + np.pi) % (2 * np.pi) - np.pi

def analyze_angle(INB, radius=20):

    # Turning point is first logged point in each inbound travel
    x_turning_point = INB['x'].loc[:,0]
    y_turning_point = INB['y'].loc[:,0]
    
    # Loop over each sample
    N = INB['x'].index[-1][0]+1 # should choose last index
    maxlen = INB['x'].index.max()[1]+1
    dist_from_tp = np.zeros((maxlen,N))
    for k in range(0,N) :
        
        dist_from_tp[:,k] = np.sqrt((INB['x'].loc[k] - x_turning_point[k])**2 +
                                    (INB['y'].loc[k] - y_turning_point[k])**2)
        
    # Find the first point where we are distance of radius from turning point
    # argmax gives first occurence when searching a boolean array
    leaving_point = np.argmax(dist_from_tp > radius, axis=0) 
    nest_angles = np.arctan2(-x_turning_point, - y_turning_point)
    return_angles=np.zeros(N)
    
    for k in range(0,N) :
        
        return_angles[k] = np.arctan2(INB['x'].loc[k,leaving_point[k]] - x_turning_point[k],
                                   INB['y'].loc[k,leaving_point[k]] - y_turning_point[k])
        
    return angular_distance(nest_angles, return_angles)

def analyze_inbound(df, Tout, Tinb, search_fraction=0.4) :
    
    # 1. Closest distance to nest when homing
    dist = np.sqrt(np.power(df['x'],2)+np.power(df['y'],2))
    try :
        closest_dist_l = [dist[k].min() for k in range(0,dist.index[-1][0]+1)]
    except TypeError : # in case df is a simple DataFrame we revert to old code
        closest_dist_l = dist.min()
    
    # 2. Estimate size of search pattern
    def searchwidth(df,search_fraction, Tout, Tinb) :
        
        search_time = Tout+Tinb*(1-search_fraction)
        search_res = df[df['Time']>search_time]
        
        min_x = search_res['x'].min()
        max_x = search_res['x'].max()    
        min_y = search_res['y'].min()
        max_y = search_res['y'].max()
        
        return (max_x-min_x+max_y-min_y)/2

    try :
        search_dist_l = [searchwidth(df.loc[k],search_fraction,Tout,Tinb) for k in range(0,dist.index[-1][0]+1)]
    except TypeError : # in case df is a simple DataFrame we revert to old code
        search_dist_l = searchwidth(df, search_fraction, Tout, Tinb)
        
    return np.mean(closest_dist_l),np.std(closest_dist_l),np.mean(search_dist_l),np.std(search_dist_l)


# Calculate the position of the agent
def integrate_to_position(headings, velocity) :
    x = np.array([0.,0.]) # starting position
    for k,_ in enumerate(headings) :
        x += velocity[k]
        
    return x

def construct_out_travel(Tout,headings,velocity) :
    # Package this into our DataFrame format
    out_time=np.arange(0,Tout)
    out_travel = pd.DataFrame()
    out_travel['Time'] = out_time
    x = np.zeros((Tout,2))
    x[0] = np.array([0.,0.]) # starting position
    for k in range(1,Tout):
        x[k] = x[k-1] + velocity[k]

    out_travel[['x','y']] = x
    out_travel[['vx','vy']] = velocity
    out_travel['heading']=headings
    
    return out_travel

def generate_straight_route(Tout, heading, velocity, offset=0.0) :
    vx = velocity*np.sin(heading+offset)
    vy = velocity*np.cos(heading+offset)
    v_vec = np.ones((Tout,2))
    v_vec[:,0] *=vx
    v_vec[:,1] *=vy    
    h_vec = np.ones(Tout)*heading
    return h_vec,v_vec
    
def run_trial(trial_nw,Tout=1000,Tinb=1500,
              tn2scaling=0.9,noise=0.1, savestep=1.0,tb1scaling=0.9, 
              bias_scaling=0.1,
              straight_route=False, fix_heading=0.0, fix_velocity=0.5, offset=0.0,
              **kwargs) :
    
    if straight_route :
        headings, velocity = generate_straight_route(Tout, fix_heading, fix_velocity, offset)
    else :
        # Generate route as Tom Stone did
        headings, velocity = pathtrials.generate_route(T=Tout, vary_speed=True)
    outbound_end_position = integrate_to_position(headings,velocity)

    out_travel = construct_out_travel(Tout, headings, velocity)
        
    # Generate signals cl1 and tn2 for all Tout time steps
    tb1 = np.zeros(stone.N_TB1)
    memory = np.zeros(stone.N_CPU4)
    cl1 = np.zeros((Tout,stone.N_CL1))
    tn2 = np.zeros((Tout,stone.N_TN2))
    
    cx_pontine = pathtrials.get_cx_instance(noise)
    for k in range(Tout) :
        _, cl1[k], tb1, _, tn2[k], memory, _, _, _ = pathtrials.update_cells(headings[k], velocity[k], tb1, memory, cx_pontine)

    # Define interpolation functions to feed some pre-produced values into the network
    def cl1_input(scaling) :
        return interp1d(range(Tout), cl1*scaling, axis=0)

    def tn2_input(scaling) :
        return interp1d(range(Tout), tn2*scaling, axis=0)
        
    def bias_input(scaling) :
        bias_func = lambda x : [scaling]
        return bias_func
    
    # Setup the inputs for CL1 and TN1
    trial_nw.specify_inputs('CL1', cl1_input, tb1scaling)
    trial_nw.specify_inputs('TN2', tn2_input, tn2scaling)
    trial_nw.specify_inputs('Bias', bias_input, bias_scaling)

    # Feed the network the correct input signals for the outbound travel
    out_res = trial_nw.evolve(T=Tout,savestep=savestep,
                              tn2scaling=tn2scaling,
                              tb1scaling=tb1scaling,
                              noise=noise,
                              **kwargs)
    
    # Let the network navigate the inbound journey
    inb_res, inb_travel = trial_nw.evolve(T=Tout+Tinb,reset=False,t0=Tout,inbound=True,
                                       initial_heading=headings[-1],
                                       initial_pos=outbound_end_position,
                                       initial_vel=velocity[-1],
                                       tn2scaling=tn2scaling,
                                       tb1scaling=tb1scaling,
                                       noise=noise,
                                       **kwargs)

    return out_res, inb_res, out_travel, inb_travel

# TODO: Use a memscale option to set the memory
def setup_network(memscale=1.0, memupdate=0.001, manipulate_shift=True, onset_shift=0.0,
                  cpu_shift=-0.2,Vt_noise=0.0, device_file=default_device_file, **kwargs) :
    
    setup_nw = stone.StoneNetwork(mem_update_h=memupdate,**kwargs) 
    # Setup the internal devices
    devices = {}
    layers = ['TB1','Rectifier','CPU4','CPU1a','CPU1b','Pontine']
    for layer in layers :
        devices[layer]=physics.Device(device_file)
    
    # Scale the memory unit explicitly
    R_ref = devices['CPU4'].p_dict['Rstore']
    C_ref = devices['CPU4'].p_dict['Cstore']
    devices['CPU4'].set_parameter('Rstore',R_ref*(np.sqrt(memscale)))
    devices['CPU4'].set_parameter('Cstore',C_ref*(np.sqrt(memscale)))
    
    # Some diagnostics
    devices['CPU4'].print_parameter('Cstore')
    devices['CPU4'].print_parameter('Rstore')
    print(f'Calculate tau_gate={devices["CPU4"].calc_tau_gate()} ns')
    #setup_nw.weights['Rectifier->CPU4'].print_W()
    
    # Tune the activation functions of the CPU1 neurons
    if manipulate_shift :
        # Get the original Vt
        Vt0 = devices['CPU1a'].p_dict['Vt']
        devices["CPU1a"].p_dict['Vt'] = Vt0+cpu_shift
        devices["CPU1b"].p_dict['Vt'] = Vt0+cpu_shift

    # Feed the devices into the network
    setup_nw.assign_device(devices, unity_key='TB1')
    
    # As a final step, noisify the threshold voltages
    if Vt_noise > 0.0 :
        setup_nw.noisify_Vt(Vt_noise) 
    
    return setup_nw

def generate_filename(T_outbound, T_inbound, N, **kwargs):
    filename = 'out{0}_in{1}_N{2}'.format(str(T_outbound),
                                                str(T_inbound),
                                                str(N))
    for k, v in kwargs.items():
        filename += '_' + k + str(v)
    return filename + '.pkl'

def generate_figurename(T_outbound, T_inbound, N, **kwargs):
    filename = 'out{0}_in{1}_N{2}'.format(str(T_outbound),
                                          str(T_inbound),  
                                          str(N))
    for k, v in kwargs.items():
        filename += '_' + k + str(v)
    return filename

def load_dataset(T_outbound, T_inbound, N, **kwargs):
    filename = generate_filename(T_outbound, T_inbound, N,
                                 **kwargs)
    
    with open(DATA_PATH / filename,'rb') as f :
        # Items read sequentially
        OUT=pickle.load(f)
        INB=pickle.load(f)
        # now it's empty
    
    return OUT, INB


def save_dataset(OUT, INB, T_outbound, T_inbound, N,
                 **kwargs):
    filename = generate_filename(T_outbound, T_inbound, N,
                                 **kwargs)
        
    # save to a pickle file
    with open(DATA_PATH / filename,'wb') as f :
        pickle.dump(OUT,f)
        pickle.dump(INB,f)
        
    #(OUT, INB).to_pickle(os.path.join(DATA_PATH, filename))
    
def downsample(df_x,df_t,t_sample) :
    # create a 1D interpolation
    f = interp1d(df_t, df_x)
    t0 = df_t.iloc[0]
    T = df_t.iloc[-1]
    t_vec = np.arange(t0,T,t_sample)
    return f(t_vec)     

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
    angle = -np.angle(fund_freq)
    scale = 0.2
    distance = np.absolute(fund_freq) / cpu4_mem_gain * scale
    return angle, distance

def decode_cpu4(cpu4,cpu4_mem_gain=1.0):
    """Shifts both CPU4 by +1 and -1 column to cancel 45 degree flow
    preference. When summed single sinusoid should point home."""
    cpu4_reshaped = cpu4.reshape(2, -1)
    cpu4_shifted = np.vstack([np.roll(cpu4_reshaped[0], 1),
                              np.roll(cpu4_reshaped[1], -1)])
    
    return decode_position(cpu4_shifted, cpu4_mem_gain)
        
def one_flight_results(out_res,inb_res,out_travel, inb_travel, sim_name, plot_path=PLOT_PATH, radius=20.,
                       interactive=False, show_headings=False, decode_mem=True, cpu4_mem_gain=1.0):
    if interactive :
        plt.ion()
    # Create plots
    # 1. Combined trace plot
    comb_res = pd.concat([out_res,inb_res],ignore_index=True)
    
    fig,_ = beeplotter.plot_traces(comb_res, layers=['CL1','TB1','TN2','CPU4','Pontine','CPU1'],attr='Pout',titles=True)
    # Here we need to save the figure
    plotter.save_plot(fig,'traces_'+sim_name,plot_path)
       
    # 2. Plot the combined traveled route
    Tout=out_res['Time'].iloc[-1]
    Tinb=inb_res['Time'].iloc[-1]-Tout
    min_dist, _, search_width, _ = analyze_inbound(inb_travel,Tout,Tinb)
    # Plot the combined traveled route
    fig, ax = plt.subplots()
    out_travel.plot(x='x',y='y',style='purple',ax=ax,linewidth=0.5, label='Outbound')
    inb_travel.plot(x='x',y='y',style='g',ax=ax,linewidth=0.5, label='Inbound')
    # Decode cpu4 and plot memory vector 
    if decode_mem :
        cpu4 = get_cpu4activity(out_res)
        angle, distance = decode_cpu4(cpu4,cpu4_mem_gain)
        # This angle should point from final position to nest
        ymem = distance*np.cos(angle) 
        xmem = distance*np.sin(angle)
        ax.plot(xmem,ymem,'o',color='orange',markersize=5)
        
    if show_headings:
        # Sample these datasets
        t_sample = 20
        x = downsample(inb_travel['x'],inb_travel['Time'],t_sample)
        y = downsample(inb_travel['y'],inb_travel['Time'],t_sample)
        hinb = downsample(inb_travel['heading'],inb_travel['Time'],t_sample)
        ax.quiver(x,y,np.sin(hinb),np.cos(hinb),width=0.002)
        
        x = downsample(out_travel['x'],out_travel['Time'],t_sample)
        y = downsample(out_travel['y'],out_travel['Time'],t_sample)
        hout = downsample(out_travel['heading'],out_travel['Time'],t_sample)
        ax.quiver(x,y,np.sin(hout),np.cos(hout),width=0.002)
        
    if radius is not None :
        # Draw a circle in the homing plot
        circ = patches.Circle((inb_travel['x'][0],inb_travel['y'][0]), radius=radius, color='yellow',fill=False)
        ax.add_patch(circ)
        
    ax.set_title(f'Closest dist: {min_dist:.1f}, search width: {search_width:.1f}')
    ax.set_aspect('equal')
    ax.annotate('N',(0,0),fontstyle='oblique',fontsize=14)
    # Save result again
    plotter.save_plot(fig,'route_'+sim_name,plot_path)
    
    # 3. Close figures
    if not interactive :
        plt.close('all')

def get_cpu4activity(df) :
    # Node list 
    CPU4_names = [f'CPU4_{i}-Pout' for i in range(0,16)]
    activity = np.zeros(len(CPU4_names))
    for k, name in enumerate(CPU4_names) :
        activity[k]=df[name].iloc[-1] # last instance
        
    return activity
    
def generate_dataset(T_outbound=1500, T_inbound=1500,N=10,
                     save=True, make_plots=True, plot_path=PLOT_PATH, **kwargs):
    try:
        OUT, INB = load_dataset(T_outbound, T_inbound, N,
                                           **kwargs)
    except:

        # Separate out the network arguments
        network_args = ['memscale','memupdate','cpu_shift','weight_noise','Vt_noise'] # add more here when needed
        network_kwargs = {k:v for k,v in kwargs.items() if k in network_args}

        # These are the other arguments that go into the run_trial call
        trial_kwargs = {k:v for k,v in kwargs.items() if k not in network_args}
        
        # Create somewhere to store figures
        if make_plots :
            # Make sure pyplot in in non-interactive mode
            plt.ioff()
            dirname = generate_figurename(T_outbound,T_inbound,N,**kwargs)
            plot_dir = plot_path / dirname
            if not plot_dir.is_dir() : # check first for existance
                plot_dir.mkdir() # Path object
            if 'memupdate' in kwargs :
                cpu4_mem_gain = kwargs['memupdate']
            else :
                cpu4_mem_gain = 0.001
        
        l_OUT = [] # list of DataFrames
        l_INB = [] 
        cpu4_snapshots = np.zeros((N,stone.N_CPU4))
                          
        for i in range(N):
            # Create network at every instance
            # I guess if filtered_kwargs is empty this is an empty call
            trial_nw = setup_network(**network_kwargs)
            
            out_res, inb_res, out_travel, inb_travel = run_trial( 
                    trial_nw,
                    Tout=T_outbound,
                    Tinb=T_inbound,
                    **trial_kwargs) # remaining keywords go here
            
            # Generate plots here
            if make_plots :
                plotname=generate_figurename(T_outbound,T_inbound,i,**kwargs)
                one_flight_results(out_res,inb_res,out_travel,inb_travel,
                                   plotname,plot_path=plot_dir,cpu4_mem_gain=cpu4_mem_gain)
            # Save the routes to the lists
            l_OUT.append(out_travel)
            l_INB.append(inb_travel)
            # Separate out a cpu4 snapshot
            cpu4_at_return = get_cpu4activity(out_res)
        
        # Combine the DataFrame to a big one
        key_sequence = np.arange(0,N)
        OUT = pd.concat(l_OUT,keys=key_sequence)
        INB = pd.concat(l_INB,keys=key_sequence)
        
        # Add the cpu4 activity
        cpu4_snapshots[i] = cpu4_at_return
        
        if save:
            save_dataset(OUT, INB, T_outbound, T_inbound, N,
                         **kwargs)
    return OUT, INB