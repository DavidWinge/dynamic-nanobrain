#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# %%
# %matplotlib inline
# %load_ext autoreload
# %autoreload 2

import numpy as np
from matplotlib import pyplot as plt
from scipy.interpolate import interp1d 
from pathlib import Path
import time
import pickle
# modules specific to this project
#import context
from dynamicnanobrain.core import physics
from dynamicnanobrain.core import plotter
from dynamicnanobrain.esn import esn

PLOT_PATH= Path('../plots/esn/')
DATA_PATH= Path('../data/esn/')

nature_single = 89.0 / 25.4
nature_double = 183.0 / 25.4
nature_full = 247.0 / 25.4

#%% Signal code
"""
Mackey, M. C. and Glass, L. (1977). Oscillation and chaos in physiological control systems.
Science, 197(4300):287-289.

dy/dt = -by(t)+ cy(t - tau) / 1+y(t-tau)^10
"""

def get_data(b=0.1, c=0.2, tau=17, initial_values = np.linspace(0.5,1.5, 18), iterations=1000):
    '''
    Return a list with the Mackey-Glass chaotic time series.

    :param b: Equation coefficient
    :param c: Equation coefficient
    :param tau: Lag parameter, default: 17
    :param initial_values: numpy array with the initial values of y. Default: np.linspace(0.5,1.5,18)
    :param iterations: number of iterations. Default: 1000
    :return:
    '''
    y = initial_values.tolist()

    for n in np.arange(len(y)-1, iterations+100):
        y.append(y[n] - b * y[n] + c * y[n - tau] / (1 + y[n - tau] ** 10))

    return y[100:]

def signal_generator(tend,scale) :
    # determine the size of the sequence
    N = int(tend*scale)
    signal = get_data(iterations=N)
    tseries = np.linspace(0,tend,len(signal),endpoint=True)
    
    return signal,tseries

#%% Plotting functions
def plot_memory_dist(network,plotname) :
    """ Sample the RC constant for the memory storage only."""
    A = network.sample_A()
    tau = (-np.sum(A,axis=3)[:,:,2].flatten())**-1
    
    fig, ax = plt.subplots()
    ax.hist(tau)
    ax.set_xlabel('Memory timescale (ns)')
    ax.set_ylabel('Occurance')
    plotter.save_plot(fig,'memorydist_'+plotname,PLOT_PATH)
    
    return fig, ax
    
def plot_timetrace(tseries_train,pred_train,tseries_test,pred_test,teacher_handle,
                   plotname='trace') :
    
    fig, ax1 = plt.subplots(figsize=(nature_full,nature_single))
    
    ax1.plot(tseries_train[:],pred_train[:,0],linewidth=0.5)
    ax1.plot(tseries_train,teacher_handle(tseries_train),'--',linewidth=0.5)
    if tseries_test is not None :
        teacher_series=teacher_handle(tseries_test)
        ax1.plot(tseries_test[:],pred_test[:,0],linewidth=0.5)
        ax1.plot(tseries_test,teacher_series,'--',linewidth=0.5)
    
    ax1.set_xlabel('Time (ns)')

    ax1.set_ylabel('Output signal (nA)')
    ax1.set_ylim(0,teacher_series.max()+25)
    ax1.set_xlim(tseries_test[0]-500, tseries_test[0]+1000)
    plotter.save_plot(fig,'trace_'+plotname,PLOT_PATH)
    #plt.close(fig)
    
    return fig, ax1

#%% Code to run one trial
def train_network(network,teacher_signal,Tfit=300,beta=10,teacher_forcing=False) :
    
    # Harvest states
    tseries_train, states_train, _, tend = network.harvest_states(Tfit,teacher_forcing=teacher_forcing)

    teacher_train = np.zeros((len(tseries_train),len(teacher_signal)))
    for k in range(0,len(teacher_signal)) :
        teacher_train[:,k] = np.squeeze(teacher_signal[k](tseries_train))
    
    pred_train, train_error = network.fit(states_train,teacher_train,beta)
    
    return pred_train, train_error, tseries_train, states_train, tend

def characterize_network(network, teacher_signal, T0=300, Tpred=300) :
    
    tseries_test, pred_test, movie_series, plot_series = network.predict(T0,T0+Tpred,output_all=True)
    # Get the target signal
    teacher_test = np.zeros((len(tseries_test),len(teacher_signal)))
    for k in range(0,len(teacher_signal)) :
        print(k,tseries_test.max())
        teacher_test[:,k] = np.squeeze(teacher_signal[k](tseries_test))
 
    pred_error = np.sqrt(np.mean((pred_test - teacher_test)**2))/network.Imax
    
    return pred_test, pred_error, tseries_test

def single_test(Tfit=500,Tpred=500,scale=1.0,Nreservoir=100,sparsity=10,
             transient=0,plotname='mg',generate_plots=True,seed=None,noise=0.,
             spectral_radius=0.9,input_scaling=0.5,bias_scaling=0.0,diagnostic=False,
             teacher_scaling=1.5, memory_variation=0.0, memscale=1.0, dist='uniform',
             beta=0.1) :

   # Specify device, start with the 1 ns device that we will scale
    propagator = physics.Device('../parameters/device_parameters_1ns.txt')
    # Scale the memory unit explicitly
    R_ref = propagator.p_dict['Rstore']
    C_ref = propagator.p_dict['Cstore']
    propagator.set_parameter('Rstore',R_ref*(np.sqrt(memscale)))
    propagator.set_parameter('Cstore',C_ref*(np.sqrt(memscale)))
    
    # Generate a random network 
    new_esn = esn.EchoStateNetwork(Nreservoir,sparsity=sparsity,device=propagator,seed=seed)
    new_esn.specify_network(spectral_radius,
                            input_scaling,
                            bias_scaling,
                            teacher_scaling, 
                            Noutput=1)
    
    # For teacher forcing we can set the delay here
    new_esn.set_delay(scale,1200)
    
    if memory_variation > 0.0 :
        new_esn.randomize_memory(memory_variation,dist)
        #plot_memory_dist(new_esn)
        
    signal, tseries = signal_generator(Tfit+Tpred,scale) 

    # clip values of the signal_input (for safety reasons, noise etc.)
    signal = np.clip(signal,0.0,None)
       
    # Signals as scalable input handles
    # Use interpolation to get function handles from these data
    def output_func(signal_scale) :
        handle = interp1d(tseries,signal*signal_scale,axis=0,fill_value=signal_scale,bounds_error=False)  
        return handle

    def bias_func(signal_scale) :
        return lambda t : signal_scale
    
    # Specify explicit signals by handle
    # Give it only a bias value!
    new_esn.specify_inputs(bias_func,bias_func,output_func)
    
    # Generate the target signal
    signal_handle = output_func(new_esn.Imax*new_esn.teacher_scaling)
    
    start = time.time()
    # Train
    pred_train, train_error, tseries_train, states_train, tend = train_network(new_esn,[signal_handle],
                                                                               Tfit=Tfit,beta=beta,teacher_forcing=True)
    
    # Characterize
    pred_test, pred_error, tseries_test = characterize_network(new_esn, [signal_handle], 
                                                               T0=tend,Tpred=Tpred)
    print('Prediction error:',pred_error)
    
    end = time.time()
    
    if generate_plots :
        plot_timetrace(tseries_train,pred_train,tseries_test,pred_test, signal_handle, plotname)
        if memory_variation > 0.0 :
            plot_memory_dist(new_esn,plotname)

    # Here we might need something to evaluate the network performance
    #freq_error = evaluate_nw(tseries_test,pred_test,signal_handle,fmin,control_handle,plotname)
    
    if diagnostic :
        return train_error, pred_error, end-start, new_esn, tseries_train, states_train, pred_train, tseries_test, pred_test, signal_handle
    else :
        return train_error, pred_error, end-start #, new_esn, tseries_train, pred_train, tseries_test, pred_test, (t1_handle, t2_handle)

#%% Test run

plt.ion()
train_error, pred_error, time_used, new_esn, tseries_train, states_train, pred_train, tseries_test, pred_test, signal_handle = single_test(Tfit=1000,Tpred=400,scale=0.5,Nreservoir=500,diagnostic=True,spectral_radius=1.5,sparsity=400)


#%% Test code for signal generation
T = 1000 # ns
scale = 1.

signal,  tseries = signal_generator(T,scale) 

# A fourier transform to check the frequencies


print(f'Generated a time series from 0 to {T} ns with {len(tseries)} elements')

if True :
           
    Nmax = 2999
    fig, ax1 = plt.subplots()
    
    ax1.plot(tseries[:Nmax],signal[:Nmax])
    #ax2.plot(tseries[:Nmax],frequency_output[:Nmax])

    plt.show()



#%% Driver code

#%% Look at some states

fig, ax = plotter.sample_npseries(tseries_train,states_train, transient=100, Nplots=10, onecolumn=True)
fig.show()


#%% Code to automate the test sequence

def generate_figurename(kwargs,suffix='') :
    filename = ''
    for k, v in kwargs.items():
        filename += '_' + k + str(v)
    filename+=suffix
    return filename

def generate_filename(N,kwargs,suffix='.pkl') :
    filename = f'freqgen_N{N}'
    for k, v in kwargs.items():
        filename += '_' + k + str(v)
    
    return filename+suffix

def repeat_runs(N, param_dict,save=True) :
    """Iterates through param dictionary, running batches of trials according to the param dictionary"""
      
    timing=np.zeros((N,param_dict['n']))
    freq_errors=np.zeros((N,param_dict['n']))
    train_errors=np.zeros((N,param_dict['n']))
    pred_errors=np.zeros((N,param_dict['n']))
    
    # Set 
    plt.ioff()
    
    print(param_dict)
    kwargs = {}
    for k, v in param_dict.items():
        if k != 'n' and v[0] != None:
            kwargs[k] = v[0]
            
    try :
        # See if this has already been run
        train_errors, pred_errors, freq_errors, timing = load_dataset(N,kwargs)
        
    except :
        # Otherwise run it now
        for m in range(N) :
            plotname = generate_figurename(kwargs,suffix=f'_{m}')
            train_errors[m], pred_errors[m], freq_errors[m], timing[m], states = single_test(plotname=plotname,seed=m**kwargs)
            print('Status update, m =',m)
            plt.close('all')
    
        if save:
            save_dataset((train_errors, pred_errors, freq_errors, timing), N, kwargs)
            save_training((states,m), N, kwargs, m)

    return train_errors, pred_errors, freq_errors, timing

def save_training(arrays,N,kwargs,m):
    filename = 'train_' + generate_filename(N,kwargs) + f'_{m}'
        
    # save to a pickle file
    with open(DATA_PATH / filename,'wb') as f :
        for a in arrays :
            pickle.dump(a,f)

def save_dataset(arrays,N,kwargs):
    filename = generate_filename(N,kwargs)
        
    # save to a pickle file
    with open(DATA_PATH / filename,'wb') as f :
        for a in arrays :
            pickle.dump(a,f)
            

def load_dataset(N, kwargs):
    filename = generate_filename(N,kwargs)
    
    with open(DATA_PATH / filename,'rb') as f :
        # Items read sequentially
        train_errors = pickle.load(f)
        pred_errors = pickle.load(f)
        freq_errors = pickle.load(f)
        timing = pickle.load(f)
        # now it's empty
    
    return train_errors, pred_errors, freq_errors, timing