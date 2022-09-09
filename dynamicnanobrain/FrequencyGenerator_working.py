# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.13.0
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %%
# %matplotlib inline
# %load_ext autoreload
# %autoreload 2

import matplotlib.pyplot as plt
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
TRAIN_PATH = Path('../data/esn/train/')

nature_single = 89.0 / 25.4
nature_double = 183.0 / 25.4
nature_full = 247.0 / 25.4

# %% [markdown] Setup the input/output for training
# ### Input and outputs to train the network
# Both input and output need to be supplied to the network in order to train it. 
# These are generated as a random sequence of frequecies.
import numpy as np

def frequency_step_generator(tend,fmin,fmax,dT,res=20,seed=None) :
    # determine the size of the sequence\
    rng = np.random.default_rng(seed)
    dt = fmax**-1/res
    N = int(tend/dt) # steps of total time interval
    dN = int(dT/dt) # steps of average period
    # From the info above we can setup our intervals
    n_changepoints = int(N/dN)
    changepoints = np.insert(np.sort(rng.integers(0,N,n_changepoints)),[0,n_changepoints],[0,N])
    # From here on we use the pyESN example code, with some modifications
    const_intervals = list(zip(changepoints,np.roll(changepoints,-1)))[:-1]
    frequency_control = np.zeros((N,1))
    for k, (t0,t1) in enumerate(const_intervals): # enumerate here
        frequency_control[t0:t1] = fmin + (fmax-fmin)* rng.random() # random instead of discrete (k % 2)
    # run time update through a sine, while changing the freqeuncy
    frequency_output = np.zeros((N,1))
    z = 0
    for i in range(N):
        z = z + 2*np.pi*frequency_control[i]*dt
        frequency_output[i] = (np.sin(z) + 1)/2
        
    tseries = np.linspace(0,tend,num=N,endpoint=False)
    
    return frequency_control,frequency_output,tseries
 
#%% Code for running a single test
def plot_timetrace(tseries_train,pred_train,tseries_test,pred_test,teacher_handle,
                   control_handle,
                   plotname='trace') :
    
    fig, ax1 = plt.subplots(figsize=(nature_full,nature_single))
    
    ax1.plot(tseries_train[:],pred_train[:,0],linewidth=0.5)
    ax1.plot(tseries_train,teacher_handle(tseries_train),'--',linewidth=0.5)
    if tseries_test is not None :
        teacher_series=teacher_handle(tseries_test)
        ax1.plot(tseries_test[:],pred_test[:,0],linewidth=0.5)
        ax1.plot(tseries_test,teacher_series,'--',linewidth=0.5)
        
    ax1.plot(tseries_train,control_handle(tseries_train),'k',linewidth=0.5)
    ax1.plot(tseries_test,control_handle(tseries_test),'k',linewidth=0.5)
    
    ax1.set_xlabel('Time (ns)')

    ax1.set_ylabel('Output signal (nA)')
    ax1.set_ylim(0,teacher_series[:,0].max()+25)
    ax1.set_xlim(tseries_test[0]-500, tseries_test[0]+1000)
    plotter.save_plot(fig,'trace_'+plotname,PLOT_PATH)
    #plt.close(fig)
    
    return fig, ax1
    
def plot_control_freq(ts,control_handle,tf,fp,ft,plotname):
    fig, ax1 = plt.subplots()
    #ax1.plot(ts,control_handle(ts),label='control')
    ax1.plot(tf+ts[0],fp,'ro',label='prediction') # tf is measured from 0
    ax1.plot(tf+ts[0],ft,'kx',label='teacher')
    ax1.set_xlabel('Time (ns)')

    ax1.set_ylabel('Frequency control')
    
    plotter.save_plot(fig,'control_'+plotname,PLOT_PATH)
    #plt.close(fig)
    
    return fig, ax1
    
def plot_memory_dist(network,plotname) :
    """ Sample the RC constant for the memory storage only."""
    A = network.sample_A()
    tau = (-np.sum(A,axis=3)[:,:,2].flatten())**-1
    
    fig, ax = plt.subplots()
    ax.hist(tau)
    ax.set_xlabel('Memory timescale (ns)')
    ax.set_ylabel('Occurance')
    plotter.save_plot(fig,'memorydist_'+plotname,PLOT_PATH)
    plt.close(fig)
    
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

def fspec(data,dt,fcutoff,plotname):
    fig, ax = plt.subplots()
    spectrum, freqs, t, im = ax.specgram(np.squeeze(data),Fs=int(1/dt),NFFT=128,noverlap=32)
    fmin_idx = np.flatnonzero(freqs>fcutoff)[0] # first index BELOW fmin
    max_args = np.argmax(spectrum[fmin_idx:],axis=0) + fmin_idx
    max_freq = freqs[max_args]
    ax.plot(t,max_freq,'ro')
    plotter.save_plot(fig,'spec'+plotname,PLOT_PATH)
    return t, max_freq
    
def evaluate_nw(ts,pred,signal_handle,fcutoff,control_handle,plotname):
    teacher = signal_handle(ts)
    # Now we should generate the frequency spectrum for each and compare using
    # least squares to find the error   
    dt = ts[1]-ts[0] # constant sampling time is used
    t_freqs, f_pred = fspec(pred,dt,fcutoff,'pred_'+plotname) 
    t_teach, f_teach= fspec(teacher,dt,fcutoff,'teach_'+plotname)
    
    print(t_freqs)
    plot_control_freq(ts,control_handle,t_freqs,f_pred,f_teach,plotname)
    
    # Now compare these signals
    freq_error = np.sqrt(np.mean((f_pred-f_teach)**2))
    # Tweak!
    freq_error = f_pred[0]
    return freq_error

def normalize(x,tiny=1e-15,interval=(0.,1.)):
    unscaled_delta = x.max()-x.min()
    unscaled_mean = (x.max()+x.min())/2
    # Shift to midpoint zero
    x -= unscaled_mean
    amp=interval[1]-interval[0]
    mean=sum(interval)/2
    if unscaled_delta > 0. : # catch edge case
        x *= amp/unscaled_delta
    x += mean
    return x+tiny
    
def single_test(fmin=0.025,fmax=0.1,Nf=2,dT=50,Tfit=1000,Tpred=1000,Nreservoir=100,sparsity=10,
             transient=0,plotname='freqgen',generate_plots=True,seed=None,noise=0.,
             spectral_radius=0.25,input_scaling=1.25,bias_scaling=0.1,diagnostic=False,
             teacher_scaling=1.0, memory_variation=0.0, memscale=1.0, dist='uniform',
             beta=0.1,fcutoff=0.015) :

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
    new_esn.set_delay(1.,1600)
    
    if memory_variation > 0.0 :
        new_esn.randomize_memory(memory_variation,dist)
        #plot_memory_dist(new_esn)
        
    control, signal, tseries = frequency_step_generator(Tfit+Tpred,fmin,fmax,dT,seed=seed) 

    # clip values of the signal_input (for safety reasons, noise etc.)
    signal = np.clip(signal,0.0,None)
    
    # normalize control in order to have it between 0.1 and 0.9
    control = normalize(control,interval=(0.1,0.9))
    
    # Signals as scalable input handles
    # Use interpolation to get function handles from these data
    def control_func(signal_scale) :
        handle = interp1d(tseries,control*signal_scale,axis=0,fill_value=signal_scale,bounds_error=False)
        return handle 
        
    def signal_func(signal_scale) :
        handle = interp1d(tseries,signal*signal_scale,axis=0,fill_value=signal_scale,bounds_error=False)  
        return handle

    def bias_func(signal_scale) :
        return lambda t : signal_scale
    
    # Specify explicit signals by handle
    new_esn.specify_inputs(control_func,bias_func,signal_func)
    
    # Generate the target signal
    signal_handle = signal_func(new_esn.Imax*new_esn.teacher_scaling)
    control_handle = control_func(new_esn.Imax*new_esn.input_scaling)
    
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
        plot_timetrace(tseries_train,pred_train,tseries_test,pred_test, signal_handle, control_handle,plotname)
        if memory_variation > 0.0 :
            plot_memory_dist(new_esn,plotname)

    # Here we might need something to evaluate the network performance
    freq_error = evaluate_nw(tseries_test,pred_test,signal_handle,fcutoff,control_handle,plotname)
    
    if diagnostic :
        return train_error, pred_error, freq_error, end-start, states_train, new_esn, tseries_train, pred_train, tseries_test, pred_test, signal_handle, control_handle
    else :
        return train_error, pred_error, freq_error, end-start, states_train #, new_esn, tseries_train, pred_train, tseries_test, pred_test, (t1_handle, t2_handle)

    
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

def repeat_runs(N, param_dict,save=True,seed_zero=0) :
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
            train_errors[m], pred_errors[m], freq_errors[m], timing[m], states = single_test(plotname=plotname,seed=m+seed_zero,**kwargs)
            if save :
                save_training((states), N, kwargs, seed_zero)
            print('Status update, m =',m)
            plt.close('all')
    
        if save:
            save_dataset((train_errors, pred_errors, freq_errors, timing), N, kwargs)


    return train_errors, pred_errors, freq_errors, timing

def save_training(arrays,N,kwargs,seed_zero):
    filename = f'train_seed{seed_zero}_' + generate_filename(N,kwargs) 
        
    # save to a pickle file
    with open(TRAIN_PATH / filename,'ab') as f :
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
   



#%% Test code for signal generation
T = 500 # ns
dT = 100 # ns, average period length of constant frequency
fmin = 1/40 # GHz
fmax = 1/40 # GHz

frequency_input, frequency_output, tseries = frequency_step_generator(T,fmin,fmax,dT,res=20) 

# A fourier transform to check the frequencies


print(f'Generated a time series from 0 to {T} ns with {len(tseries)} elements')

if True :
           
    Nmax = 2999
    fig, (ax1, ax2) = plt.subplots(1,2)
    
    ax1.plot(tseries[:Nmax],frequency_input[:Nmax])
    ax2.plot(tseries[:Nmax],frequency_output[:Nmax])

    plt.show()

#%%


frequency_input, frequency_output, tseries = frequency_step_generator(T,fmin,fmax,dT,res=10) 

fig, ax = plt.subplots()

dt = tseries[1]-tseries[0] # constant sampling time is used
spectrum, freqs, t, im =  ax.specgram(np.squeeze(frequency_output),Fs=int(1/dt),NFFT=128,noverlap=32)

fmin_idx = np.flatnonzero(freqs>fmin)[0]-1 # first index BELOW fmin
max_args = np.argmax(spectrum[fmin_idx:],axis=0) + fmin_idx
max_freq = freqs[max_args]

ax.plot(t,max_freq,'ro')

plt.show



#%% Driver code

plt.ion()
train_error, pred_error, freq_error, time_used, states, new_esn, tseries_train, pred_train, tseries_test, pred_test, signal_handle, control_handle = single_test(fmin=0.005,fmax=0.1,Tfit=2000,Tpred=2000,dT=100,teacher_scaling=1.75,diagnostic=True,beta=0.01,memscale=5,memory_variation=0.8,seed=4,sparsity=5)


#%% Perform a grid test to study spectral radius and scaling (could also do sparsity)
rho_vals = [0.25, 0.5, 0.75, 1.0, 1.25]
scale_vals = [0.5, 0.75, 1.0, 1.25, 1.5]
Ndf=1
param_dicts =[{'n':Ndf,'teacher_scaling':[scale],'spectral_radius':[rho],'beta':[0.01],'memscale':[5.],'memory_variation':[0.8],'seed_zero':[42]} for scale in scale_vals for rho in rho_vals]

N=10 # Each simulation is about 8 mins with a memory distribution, 1 min with plain memory

freq_errors = np.zeros((len(param_dicts),N,Ndf))
timing = np.zeros((len(param_dicts),N,Ndf))
train_errors = np.zeros((len(param_dicts),N,Ndf))
pred_errors = np.zeros((len(param_dicts),N,Ndf))

for k, param_dict in enumerate(param_dicts):    
    train_errors[k], pred_errors[k], freq_errors[k], timing[k] = repeat_runs(N,param_dict,seed_zero=5)
#%% Force f1min=f2max to check capability of network
f_vals = np.arange(0.02,0.201,step=0.01)
Ndf=1
f_vals = [0.06]
param_dicts =[{'n':Ndf,'Tfit':[200],'Tpred':[200],'fmin':[f],'fmax':[f],'teacher_scaling':[1.25],'beta':[0.01],'memscale':[5.],'memory_variation':[0.8]} for f in f_vals]

N=3 # Each simulation is about 8 mins with a memory distribution, 1 min with plain memory

freq_errors = np.zeros((len(param_dicts),N,Ndf))
timing = np.zeros((len(param_dicts),N,Ndf))
train_errors = np.zeros((len(param_dicts),N,Ndf))
pred_errors = np.zeros((len(param_dicts),N,Ndf))

for k, param_dict in enumerate(param_dicts):    
    train_errors[k], pred_errors[k], freq_errors[k], timing[k] = repeat_runs(N,param_dict,seed_zero=9)

#%%
fig, ax1 = plt.subplots(figsize=(nature_full,nature_single))

ax1.plot(tseries_train[:],pred_train[:,0],linewidth=0.5)
ax1.plot(tseries_train,signal_handle(tseries_train),'--',linewidth=0.5)

teacher_series=signal_handle(tseries_test)
ax1.plot(tseries_test[:],pred_test[:,0],linewidth=0.5)
ax1.plot(tseries_test,teacher_series,'--',linewidth=0.5)
    
ax1.plot(tseries_train,control_handle(tseries_train),'k',linewidth=0.5)
ax1.plot(tseries_test,control_handle(tseries_test),'k',linewidth=0.5)

ax1.set_xlabel('Time (ns)')

ax1.set_ylabel('Output signal (nA)')
ax1.set_ylim(0,teacher_series[:,0].max()+25)
ax1.set_xlim(tseries_test[0]-500, tseries_test[0]+1000)
#plotter.save_plot(fig,'trace_'+plotname,PLOT_PATH)
#plt.close(fig)

plt.show()
#%%


fig, ax = plt.subplots()

ax.plot(f_vals,np.squeeze(freq_errors),'-x')
ax.plot(f_vals,f_vals,'--')

plt.show()
#%%

fig, ax = plotter.sample_npseries(tseries_train,states_train, Nplots=10, onecolumn=True)
fig.show()


#%% Compare performance between dist and single tau's
Ndf = 1

param_dicts  =[{'n':Ndf,'teacher_scaling':[1.25],'beta':[0.01],'memscale':[5.]}]
param_dicts +=[{'n':Ndf,'teacher_scaling':[1.25],'beta':[0.01],'memscale':[5.],'memory_variation':[0.8]}]


N=10 # Each simulation is about 8 mins with a memory distribution, 1 min with plain memory

freq_errors = np.zeros((len(param_dicts),N,Ndf))
timing = np.zeros((len(param_dicts),N,Ndf))
train_errors = np.zeros((len(param_dicts),N,Ndf))
pred_errors = np.zeros((len(param_dicts),N,Ndf))

for k, param_dict in enumerate(param_dicts):    
    train_errors[k], pred_errors[k], freq_errors[k], timing[k] = repeat_runs(N,param_dict)
    
    
#%% Construct plot over frequency errors to identigy succesful runs

fig, ax = plt.subplots(figsize=(nature_single,nature_single))

ax.plot(freq_errors[0],label='fixed memory')
ax.plot(freq_errors[1],label='dist. memory')
#ax.set_xticklabels(input_scale_vals)
#plt.colorbar(im,ax=ax)
ax.set_xlabel('run #')
ax.set_ylabel('Frequency prediction RMS error')
ax.legend()
plt.tight_layout()
plt.show()
   
