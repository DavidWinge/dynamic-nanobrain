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
from matplotlib import cm
from scipy.interpolate import interp1d 
from scipy.signal import square
from pathlib import Path
import pickle
# modules specific to this project
#import context
from dynamicnanobrain.core import physics
from dynamicnanobrain.core import plotter
from dynamicnanobrain.esn import esn

# %% [markdown] Setup the input/output for training
# ### Input and outputs to train the network
# Both input and output need to be supplied to the network in order to train it. 
# These are generated as a random sequence of frequecies.
import numpy as np
SEED=10
rng = np.random.RandomState()

PLOT_PATH= Path('../plots/esn/')
DATA_PATH= Path('../data/esn/')

def signal_shape_generator(tend,tnoise,f0,dT,res=25,noise=0.0) :
    # determine the size of the sequence
    dt = f0**-1/res
    N = int(tend/dt) # steps of total time interval
    N1 = int((tend-tnoise)/dt) 
    dN = int(dT/dt) # steps of average period
    # From the info above we can setup our intervals
    n_changepoints = int(N1/dN)
    changepoints = np.insert(np.sort(rng.randint(0,N1,n_changepoints)),[0,n_changepoints],[0,N1])
    # From here on we use the pyESN example code, with some modifications
    const_intervals = list(zip(changepoints,np.roll(changepoints,-1)))[:-1]
    # Generate the two functions that we switch between
    tau_end = tend*f0*2*np.pi
    tau = np.linspace(0,tau_end,num=N).reshape((N,1))
    square_shape = square(tau)
    sine_shape = np.sin(tau)
    
    shape_control = np.zeros((N,1))
    signal = np.zeros((N,1))
    for k, (t0,t1) in enumerate(const_intervals): # enumerate here
        shape_control[t0:t1] = k % 2
        signal[t0:t1] = (k % 2) * square_shape[t0:t1] + (1. - k % 2) * sine_shape[t0:t1]
  
    # Add the part where we diagnose the signal
    N2 = int((N+N1)/2)
    shape_control[N1:N2] = 0.0
    shape_control[N2:] = 1.0
    signal[N1:N2] = sine_shape[N1:N2]
    signal[N2:] = square_shape[N2:]
    
    # Corresponding time vector
    tseries = np.arange(0,tend,step=dt)
    if tseries.shape[0]>N :
        tseries = tseries[:N] 
    # Add some noise here
    input_noise = np.zeros_like(signal)
    if noise > 0. :
        input_noise = np.random.normal(scale=noise,size=(N,1))
        
    return shape_control,signal+input_noise,tseries

def train_network(network,teacher_signal,Tfit=300,beta=10) :
    
    # Harvest states
    tseries_train, states_train, _ = network.harvest_states(Tfit,teacher_forcing=False)

    teacher_train = np.zeros((len(tseries_train),len(teacher_signal)))
    for k in range(0,len(teacher_signal)) :
        teacher_train[:,k] = np.squeeze(teacher_signal[k](tseries_train))
    
    pred_train, train_error = network.fit(states_train,teacher_train,beta)
    
    return pred_train, train_error, tseries_train, states_train

def characterize_network(network, tseries_train, pred_train, 
                         teacher_signal, T0=300, Tpred=300) :
    
    tseries_test, pred_test, movie_series, plot_series = network.predict(T0,T0+Tpred,output_all=True)
    # Get the target signal
    teacher_test = np.zeros((len(tseries_test),len(teacher_signal)))
    for k in range(0,len(teacher_signal)) :
        teacher_test[:,k] = np.squeeze(teacher_signal[k](tseries_test))
        
    pred_error = np.sqrt(np.mean((pred_test - teacher_test)**2))/network.Imax
    
    return pred_test, pred_error, tseries_test

            
def evaluate_noise(tseries_test, pred_test, Tnoise,transient=10,
                   plotname='noise',makeplot=False) :
       
    transN = np.flatnonzero(tseries_test>tseries_test[0]+transient)[0]
    N=len(tseries_test)
    M=int(N/2)
    # For the two parts of the signal, quantify the mean and variance
    mean_low = np.mean(pred_test[transN:M])
    var_low = np.var(pred_test[transN:M])
    mean_high = np.mean(pred_test[M+transN:])
    var_high = np.var(pred_test[M+transN:])
    
    if makeplot  :
        fig, ax1 = plt.subplots()

        ax1.plot(tseries_test,pred_test)
        ax1.plot(tseries_test[transN:M],mean_low*np.ones_like(tseries_test[transN:M]),'r--')
        ax1.plot(tseries_test[M+transN:],mean_high*np.ones_like(tseries_test[M+transN:]),'r--')
 
        ax1.set_xlabel('Time (ns)')
        ax1.set_ylabel('Output signal (nA)')
        ax1.set_ylim(0,pred_test[:,0].max()+25)
        plotter.save_plot(fig,'noise'+plotname,PLOT_PATH)
        
    return mean_low, mean_high, var_low,var_high

def plot_timetrace(tseries_train,pred_train,tseries_test=None,pred_test=None,teacher_handle=None,
                   plotname='trace') :
    
    fig, ax1 = plt.subplots()
    
    ax1.plot(tseries_train[:],pred_train[:,0])
    ax1.plot(tseries_train,teacher_handle[0](tseries_train),'--')
    if tseries_test is not None :
        ax1.plot(tseries_test[:],pred_test[:,0])
        ax1.plot(tseries_test,teacher_handle[0](tseries_test),'--')
        

    ax1.set_xlabel('Time (ns)')

    ax1.set_ylabel('Output signal (nA)')
    ax1.set_ylim(0,pred_test[:,0].max()+25)
    
    plotter.save_plot(fig,'trace_'+plotname,PLOT_PATH)
    #plt.close()
    
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
    
def run_shape_test(f0=0.05,dT=100,Tfit=2000,Tpred=1000,Tnoise=400,Nreservoir=100,sparsity=0.9,
             transient=0,plotname='shapetest',generate_plots=True,seed=None,noise=0.,
             spectral_radius=0.6,input_scaling=0.5,bias_scaling=0.1,diagnostic=False,
             teacher_scaling=1.0, memory_variation=0.0, memscale=1.0, dist='exp') :

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
                            feedback=False,
                            Noutput=1)
    
    if memory_variation > 0.0 :
        new_esn.randomize_memory(memory_variation,dist)
        #plot_memory_dist(new_esn)
        
    # input signal is a varying sine/square pulse
    shape_control, signal_input, tseries = signal_shape_generator(Tfit+Tpred+Tnoise,Tnoise,f0,dT,noise=noise) 

    # Signals as scalable input handles
    # Use interpolation to get function handles from these data
    def control_signal(signal_scale) :
        handle = interp1d(tseries,shape_control*signal_scale,axis=0,fill_value=signal_scale,bounds_error=False)
        return handle 
        
    def input_signal(signal_scale) :
        handle = interp1d(tseries,signal_input*signal_scale,axis=0)  
        return handle

    def bias_signal(signal_scale) :
        return lambda t : signal_scale
    
    # Specify explicit signals by handle
    new_esn.specify_inputs(input_signal,bias_signal,control_signal)
    
    # Generate the target signal
    control_handle = control_signal(new_esn.Imax*new_esn.teacher_scaling)

    # Train
    pred_train, train_error, tseries_train, states_train = train_network(new_esn,[control_handle],Tfit=Tfit)
    #states_train, teacher_train = train_network(new_esn,[t1_handle,t2_handle],Tfit=Tfit)
    #return states_train, teacher_train, new_esn
    
    # Characterize
    pred_test, pred_error, tseries_test = characterize_network(new_esn, tseries_train, pred_train, [control_handle], 
                                                               T0=Tfit,Tpred=Tpred+Tnoise)
    print('Prediction error:',pred_error)
    
    if generate_plots :
        plot_timetrace(tseries_train,pred_train,tseries_test,pred_test, [control_handle],plotname)
        if memory_variation > 0.0 :
            plot_memory_dist(new_esn,plotname)
            
    # Evaluate noise
    noise_mask = np.where(tseries_test>Tfit+Tpred)
    mean_low, mean_high, var_low, var_high = evaluate_noise(tseries_test[noise_mask],
                                                      pred_test[noise_mask],
                                                      Tnoise,transient,plotname=plotname,
                                                      makeplot=generate_plots)
    
    # Save only detected signal diff and average variance. 
    avg_var = (var_high+var_low)/2
    signal_diff = mean_high-mean_low            
    # Deallocate the class instance
    #del new_esn
    if diagnostic :
        return train_error, pred_error, signal_diff, avg_var, new_esn, tseries_train, pred_train, tseries_test, pred_test, control_handle    
    else :
        return train_error, pred_error, signal_diff, avg_var #, new_esn, tseries_train, pred_train, tseries_test, pred_test, (t1_handle, t2_handle)

def generate_figurename(kwargs,suffix='') :
    filename = ''
    for k, v in kwargs.items():
        filename += '_' + k + str(v)
    filename+=suffix
    return filename

def generate_filename(N,kwargs,suffix='.pkl') :
    filename = f'shapedata_N{N}'
    for k, v in kwargs.items():
        filename += '_' + k + str(v)
    
    return filename+suffix

def repeat_runs(N, param_dict,save=True) :
    """Iterates through param dictionary, running batches of trials according to the param dictionary"""
      
    signal_diffs=np.zeros((N,param_dict['n']))
    noise_vars=np.zeros((N,param_dict['n']))
    train_errors=np.zeros((N,param_dict['n']))
    pred_errors=np.zeros((N,param_dict['n']))
    
    # Set 
    plt.ioff()
    
    for i in range(param_dict['n']):
        print(param_dict)
        kwargs = {}
        for k, v in param_dict.items():
            if k != 'n' and v[i] != None:
                kwargs[k] = v[i]
                
        try :
            # See if this has already been run
            signal_diffs[:,i], noise_vars[:,i], train_errors[:,i], pred_errors[:,i] = load_dataset(N,kwargs)
            
        except :
            # Otherwise run it now
            for m in range(N) :
                plotname = generate_figurename(kwargs,suffix=f'_{m}')
                train_errors[m,i], pred_errors[m,i], signal_diffs[m,i], noise_vars[m,i] = run_shape_test(plotname=plotname,**kwargs)
                print('Status update, m =',m)
        
            if save:
                save_dataset((signal_diffs[:,i], noise_vars[:,i], train_errors[:,i], pred_errors[:,i]), N, kwargs)
    
    return signal_diffs, noise_vars, train_errors, pred_errors

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
        signal_diffs = pickle.load(f)
        noise_vars = pickle.load(f)
        train_errors = pickle.load(f)
        pred_errors = pickle.load(f)
        # now it's empty
    
    return signal_diffs, noise_vars, train_errors, pred_errors
   

#%% START NEW DEVELOPMENT HERE

# %% 
# Plot the frequency control and periods together
f0 = 0.1 # GHz
Tpred = 1000
Tfit = 2000
Tnoise = 400
T=Tpred+Tfit+Tnoise

shape_control, signal_input, tseries = signal_shape_generator(T,Tnoise,f0,100,res=25,noise=0.1) 

if True :
    Nmax = -1
    xmax =3400 #ns
    fig, axs = plt.subplots(2,1,sharex=True)

    
    axs[0].plot(tseries[:Nmax],shape_control[:Nmax])
    axs[1].plot(tseries[:Nmax],signal_input[:Nmax])

    #ax2.plot(periods[:1000])
    
    for k, ax in enumerate(axs.flatten()):
        ax.set_xlim(0,xmax)
        ax.set_xlabel('Time (ns)')
        if k%2 == 0 :
            ax.set_ylabel('Shape control (arb. units)')
        else :
            ax.set_ylabel('Signal input (nA)')
    
    plt.tight_layout()
    plt.show()


#%% Setup a network that draw memory timescales from an uniform distribution
#%%
train_error, pred_error, signal_diff, noise_var, trial_nw, tseries_train, pred_train, tseries_test, pred_test, teacher_handles = run_shape_test(f0=0.1,dT=100,generate_plots=True,memory_variation=0.9,memscale=10.0, diagnostic=True,dist='uniform')
 
#%% Look at the memory constants
A = trial_nw.sample_A()

tau = (-np.sum(A,axis=3)[:,:,2].flatten())**-1

fig, ax = plt.subplots()
ax.hist(tau)
ax.set_xlabel('Memory timescale (ns)')
ax.set_ylabel('Occurance')

fig.show()
#%% Make a nice plot showing what's going on
def plot_neat_timetrace(tseries_train,pred_train,tseries_test,pred_test,teacher_handle,
                        plotname='neat_timetrace',Tnoise=1000) :
    
    # Get the mean and noise values
    T = tseries_test[-1]
    noise_mask = np.where(tseries_test>(T-Tnoise))
    mean_low, mean_high, var_low, var_high = evaluate_double_noise(tseries_test[noise_mask], pred_test[noise_mask], Tnoise)
        
    # Get the colors used
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']
    teacher_color=colors[3]
    noise_color='black'
    train_color=colors[2]
    pred_color=colors[0]
    
    # Figure sizes
    nature_single = 89.0 / 25.4
    nature_double = 183.0 / 25.4
    nature_full = 247.0 / 25.4
    my_dpi = 300
    
    # Plot options
    font = {'family' : 'sans',
            'weight' : 'normal',
            'size'   : 10}
    plt.rc('font', **font)
    
    tseries_pred = tseries_test[np.where(tseries_test<(T-Tnoise))]
    tseries_snr = tseries_test[np.where(tseries_test>(T-Tnoise))]

    # Separate the tseries_test part
    transient=10
    transN = np.flatnonzero(tseries_snr>tseries_snr[0]+transient)[0] 
    N=len(tseries_snr)
    M=int(N/2)
    Q=int(N/4)
    sq_alpha=0.2
    
    fig, (ax1, ax2) = plt.subplots(2,1,sharex=True,figsize=(nature_full*0.6,nature_single))
    
    
    ax1.plot(tseries_train[:],pred_train[:,0],label='training',color=train_color)
    ax1.plot(tseries_train,teacher_handle[0](tseries_train),'--',color=teacher_color,label='modulation')
    ax1.plot(tseries_test,pred_test[:,0],label='prediction',color=pred_color)
    ax1.plot(tseries_pred,teacher_handle[0](tseries_pred),'--',color=teacher_color)
    
    ax1.plot(tseries_snr[transN:M],mean_low[0]*np.ones_like(tseries_snr[transN:M]),'--',color=noise_color,label=r'mean$\pm$std')
    ax1.plot(tseries_snr[M+transN:],mean_high[0]*np.ones_like(tseries_snr[M+transN:]),'--',color=noise_color)

    ax1.fill_between([tseries_snr[transN],tseries_snr[M]],
                    [mean_low[0]+np.sqrt(var_low[0])]*2,
                    [mean_low[0]-np.sqrt(var_low[0])]*2,
                    facecolor=noise_color, alpha=sq_alpha, zorder=2)
    ax1.fill_between([tseries_snr[M+transN],tseries_snr[-1]],
                    [mean_high[0]+np.sqrt(var_high[0])]*2,
                    [mean_high[0]-np.sqrt(var_high[0])]*2,
                    facecolor=noise_color, alpha=sq_alpha, zorder=2)
    
    ax1.legend(loc='lower center',ncol=4,fontsize=8)    
    
    ax2.plot(tseries_train[:],pred_train[:,1],color=train_color)
    ax2.plot(tseries_train,teacher_handle[1](tseries_train),'--',color=teacher_color)
    ax2.plot(tseries_test[:],pred_test[:,1],color=pred_color)
    ax2.plot(tseries_pred,teacher_handle[1](tseries_pred),'--',color=teacher_color)
    
    # low part (start and end)
    ax2.plot(tseries_snr[transN:M-Q],mean_low[1]*np.ones_like(tseries_snr[transN:M-Q]),'--',color=noise_color)
    ax2.plot(tseries_snr[M+Q+transN:],mean_low[1]*np.ones_like(tseries_snr[M+Q+transN:]),'--',color=noise_color)
    # high part (middle)
    ax2.plot(tseries_snr[M-Q+transN:M+Q],mean_high[1]*np.ones_like(tseries_snr[M-Q+transN:M+Q]),'--',color=noise_color)
    
    ax2.fill_between([tseries_snr[transN],tseries_snr[M-Q]],
                    [mean_low[1]+np.sqrt(var_low[1])]*2,
                    [mean_low[1]-np.sqrt(var_low[1])]*2,
                    facecolor=noise_color, alpha=sq_alpha, zorder=2)
    
    ax2.fill_between([tseries_snr[transN+M-Q],tseries_snr[M+Q]],
                    [mean_high[1]+np.sqrt(var_high[1])]*2,
                    [mean_high[1]-np.sqrt(var_high[1])]*2,
                    facecolor=noise_color, alpha=sq_alpha, zorder=2)
    
    ax2.fill_between([tseries_snr[transN+M+Q],tseries_snr[-1]],
                    [mean_low[1]+np.sqrt(var_low[1])]*2,
                    [mean_low[1]-np.sqrt(var_low[1])]*2,
                    facecolor=noise_color, alpha=sq_alpha, zorder=2)
     
    ax2.set_xlabel('Time (ns)')
    ax2.set_ylabel('Slow signal (nA)')
    ax1.set_ylabel('Fast signal (nA)')
    ax1.set_xlim(0,4000)
    ax1.set_ylim(0,pred_test[:,0].max()+25)
    ax2.set_ylim(0,pred_test[:,1].max()+25)
    ax1.tick_params('x',top=True,direction='in')
    ax2.tick_params('x',top=True,direction='in')
    
    plt.subplots_adjust(hspace=0.0)
    
    plotter.save_plot(fig,'neat_timetrace_'+plotname,PLOT_PATH)
    #plt.close()
    
    return fig, (ax1, ax2)
    
#%%

fig, ax = plot_neat_timetrace(tseries_train, pred_train, tseries_test, pred_test, teacher_handles)

fig.show()
#%%
Ndf=1
# Focus on a single df here
Nreservoir=100

noise_vals = [0., 0.1]
#noise_vals = [0.]
param_dicts  =[{'n':Ndf,'Nreservoir':[Nreservoir]*Ndf,'memscale':[10.0]*Ndf,'f0':[0.1],'dT':[100],'noise':[noise]} for noise in noise_vals]
param_dicts +=[{'n':Ndf,'Nreservoir':[Nreservoir]*Ndf,'memscale':[10.0]*Ndf,'memory_variation':[0.9],'dist':['uniform'],'f0':[0.1],'dT':[100],'noise':[noise]} for noise in noise_vals]


# Study the memscale effect on the predictabilty
memscale_vals = [2,4,6,8,10]
param_dicts =[{'n':Ndf,'Nreservoir':[Nreservoir]*Ndf,'memscale':[mem]*Ndf,'f0':[0.1],'dT':[100],'noise':[noise_vals[0]]} for mem in memscale_vals]
param_dicts +=[{'n':Ndf,'Nreservoir':[Nreservoir]*Ndf,'memscale':[mem]*Ndf,'f0':[0.1],'dT':[100],'noise':[noise_vals[1]]} for mem in memscale_vals]
param_dicts +=[{'n':Ndf,'Nreservoir':[Nreservoir]*Ndf,'memscale':[10.0]*Ndf,'memory_variation':[0.9],'dist':['uniform'],'f0':[0.1],'dT':[100],'noise':[noise]} for noise in noise_vals]
           
N=20 # Each simulation is about 8 mins with a memory distribution, 1 min with plain memory

signal_diffs = np.zeros((len(param_dicts),N,Ndf))
noise_vars = np.zeros((len(param_dicts),N,Ndf))
train_errors = np.zeros((len(param_dicts),N,Ndf))
pred_errors = np.zeros((len(param_dicts),N,Ndf))

for k, param_dict in enumerate(param_dicts):    
    signal_diffs[k], noise_vars[k], train_errors[k], pred_errors[k] = repeat_runs(N,param_dict)
    
#%% Plot the SNR as a function of memory variation
snr = signal_diffs/np.sqrt(noise_vars)

std_vs_mem = np.std(signal_diffs,axis=1)
mean_vs_mem = np.mean(signal_diffs,axis=1)

snr_mean = np.mean(snr, axis=1)
snr_std = np.std(snr, axis=1)

xaxis_data = noise_vals*3
M = len(noise_vals)

fig, ax = plt.subplots()

ax.errorbar(xaxis_data[:M],np.squeeze(mean_vs_mem)[:M],np.squeeze(std_vs_mem)[:M],fmt='s',capsize=3.0,label=r'single-$\tau$')
ax.errorbar(xaxis_data[M:M+1],np.squeeze(mean_vs_mem)[M:M+1],np.squeeze(std_vs_mem)[M:M+1],fmt='^',capsize=3.0,label=r'dist-$\tau$')
ax.errorbar(xaxis_data[M+1:],np.squeeze(mean_vs_mem)[M+1:],np.squeeze(std_vs_mem)[M+1:],fmt='s',capsize=3.0,label=r'single-$\tau$:5 ns')


ax.set_xlabel('Noise fraction')
ax.set_ylabel('Signal (high-low)')
#ax.set_ylim(0,12)
#ax.set_xlim(-0.05,0.55)

fig.show()

fig, ax = plt.subplots()

ax.errorbar(xaxis_data[:M],np.squeeze(snr_mean)[:M],np.squeeze(snr_std)[:M],fmt='s',capsize=3.0,label=r'single-$\tau$')
ax.errorbar(xaxis_data[M:M+1],np.squeeze(snr_mean)[M:M+1],np.squeeze(snr_std)[M:M+1],fmt='^',capsize=3.0,label=r'dist-$\tau$')
ax.errorbar(xaxis_data[M+1:],np.squeeze(snr_mean)[M+1:],np.squeeze(snr_std)[M+1:],fmt='s',capsize=3.0,label=r'single-$\tau$:5 ns')

ax.set_xlabel('Noise fraction')
ax.set_ylabel('SNR (mean/std. dev.)')
ax.legend()
ax.grid(True)
#ax.set_ylim(0,12)
#ax.set_xlim(-0.05,0.55)

fig.show()
#%% Plot the SNR as a function of memscale
snr = signal_diffs/np.sqrt(noise_vars)

std_vs_mem = np.std(signal_diffs,axis=1)
mean_vs_mem = np.mean(signal_diffs,axis=1)

snr_mean = np.mean(snr, axis=1)
snr_std = np.std(snr, axis=1)

xaxis_data = memscale_vals
M = len(xaxis_data)

fig, ax = plt.subplots()

ax.errorbar(xaxis_data,np.squeeze(mean_vs_mem[:M]),np.squeeze(std_vs_mem[:M]),fmt='s',capsize=3.0,label=r'single-$\tau$')
ax.errorbar(10,np.squeeze(mean_vs_mem)[-1],np.squeeze(std_vs_mem)[-1],fmt='^',capsize=3.0,label=r'dist-$\tau$')
#ax.errorbar(xaxis_data[M:M+1],np.squeeze(mean_vs_mem)[M:M+1],np.squeeze(std_vs_mem)[M:M+1],fmt='^',capsize=3.0,label=r'dist-$\tau$')
#ax.errorbar(xaxis_data[M+1:],np.squeeze(mean_vs_mem)[M+1:],np.squeeze(std_vs_mem)[M+1:],fmt='s',capsize=3.0,label=r'single-$\tau$:5 ns')


ax.set_xlabel('Memory timescale (ns)')
ax.set_ylabel('Signal (high-low)')
#ax.set_ylim(0,12)
#ax.set_xlim(-0.05,0.55)

fig.show()

fig, ax = plt.subplots(len(noise_vals),1,sharex=True)

ax.errorbar(xaxis_data,np.squeeze(snr_mean[:M]),np.squeeze(snr_std[:M]),fmt='s',capsize=3.0,label=r'single-$\tau$')
ax.errorbar(10,np.squeeze(snr_mean)[-1],np.squeeze(snr_std)[-1],fmt='^',capsize=3.0,label=r'dist-$\tau$')
#ax.errorbar(xaxis_data[M:M+1],np.squeeze(snr_mean)[M:M+1],np.squeeze(snr_std)[M:M+1],fmt='^',capsize=3.0,label=r'dist-$\tau$')
#ax.errorbar(xaxis_data[M+1:],np.squeeze(snr_mean)[M+1:],np.squeeze(snr_std)[M+1:],fmt='s',capsize=3.0,label=r'single-$\tau$:5 ns')

ax.set_xlabel('Memory timescale (ns)')
ax.set_ylabel('SNR (mean/std. dev.)')
ax.legend()
ax.grid(True)
#ax.set_ylim(0,12)
#ax.set_xlim(-0.05,0.55)

fig.show()
#%% Make a nicer figure for the 2 points
nature_single = 89.0 / 25.4
# Plot options
font = {'family' : 'sans',
        'weight' : 'normal',
        'size'   : 12}
plt.rc('font', **font)

fig, ax = plt.subplots(figsize=(nature_single,nature_single))

memscale_vals[1]=1.0
ax.errorbar(memscale_vals[:],np.squeeze(snr_mean)[:,0],np.squeeze(snr_std)[:,0],fmt='s',capsize=3.0,label='fast')
ax.errorbar(memscale_vals[:],np.squeeze(snr_mean)[:,1],np.squeeze(snr_std)[:,1],fmt='^',capsize=3.0,label='slow')

ax.set_xticks([0,1])
ax.set_xticklabels([r'Single $\tau$',r'Distributed $\tau$'])
ax.set_xlim(-0.5,1.5)
ax.set_ylim(0,snr_mean.max()+2)

ax.set_ylabel('SNR')
ax.legend()

fig.show()



#%%            
# Generate the target signal
#teacher_handle = teacher_signal(my_esn.Imax*teacher_scaling)
# Need the teacher signal as arrays in order to calculate the errors
teacher_train = np.zeros_like(pred_train)
teacher_test = np.zeros_like(pred_test)
for k in [0,1] :
    teacher_train[:,k] = teacher_handles[k](tseries_train)[:,0]
    teacher_test[:,k] = teacher_handles[k](tseries_test)[:,0]

#plt.plot(tseries_test,pred_test)
#plt.plot(tseries_train,pred_train)
#plt.plot(tseries_train,teacher_train,'k--')
#plt.plot(tseries_test,teacher_test,'k--')

#plt.show()

fig, (ax1, ax2) = plt.subplots(2,1,sharex=True)
k=0
ax1.plot(tseries_train,pred_train[:,k])
ax1.plot(tseries_train,teacher_train[:,k],'--')
ax1.plot(tseries_test,pred_test[:,k])
ax1.plot(tseries_test,teacher_test[:,k],'--' )
k=1
ax2.plot(tseries_train,pred_train[:,k])
ax2.plot(tseries_train,teacher_train[:,k],'--')
ax2.plot(tseries_test[:],pred_test[:,k])
ax2.plot(tseries_test,teacher_test[:,k],'--' )
# low part (start and end)

    
ax2.set_xlabel('Time (ns)')
ax2.set_ylabel('Output signal (nA)')
ax1.set_ylabel('Output signal (nA)')
ax1.set_ylim(0,pred_test[:,0].max()+25)
ax2.set_ylim(0,pred_test[:,1].max()+25)

plt.show()

train_error = np.sqrt(np.mean((pred_train - teacher_train)**2))/my_esn.Imax
pred_error = np.sqrt(np.mean((pred_test - teacher_test)**2))/my_esn.Imax

#%%plt.show()

fig, (ax1, ax2) = plt.subplots(2,1,sharex=True)
k=0
ax1.plot(tseries_train,(pred_train[:,k]-teacher_train[:,k])**2,'--')
ax1.plot(tseries_test,(pred_test[:,k] - teacher_test[:,k])**2,'--')

k=1
ax2.plot(tseries_train,(pred_train[:,k]-teacher_train[:,k])**2,'--')
ax2.plot(tseries_test,(pred_test[:,k] - teacher_test[:,k])**2,'--')
# low part (start and end)

    
ax2.set_xlabel('Time (ns)')
ax2.set_ylabel('Output signal (nA)')
ax1.set_ylabel('Output signal (nA)')
ax1.set_ylim(0,((pred_test[:,0] - teacher_test[:,0])**2).max()+200)
ax2.set_ylim(0,((pred_test[:,1] - teacher_test[:,1])**2).max()+200)

plt.show()
#%%
def plotstatevectors(network,states,tau_min,tau_max) :
    
    # Get the relevant tau matrix
    A = my_esn.sample_A()
    tau_0 = (-np.sum(A,axis=3)[0,:,2])**-1
    mem_mask = (tau_0>tau_min) * (tau_0<tau_max)

    selected_states = states[:,np.concatenate((mem_mask,np.array([False,False])))]
    selected_taus = tau_0[mem_mask]
    
    fig, ax = plt.subplots()
    
    # Plot the selected states
    for k in range(0,selected_states.shape[1]) :
        ax.plot(states[:,k],label=f'tau={selected_taus[k]}')
    
    ax.set_xlabel('Time (ns)')
    ax.set_ylabel('Output current (nA)')
    ax.legend()
    
    return mem_mask
    

# Need a way to look at the slow/fast nodes and their state vectors
#plotstatevectors(my_esn,states,0,5)



#%% Need to check the Bode plot for this device
propagator = physics.Device('../parameters/device_parameters_1ns.txt')
RC_scale = 1
Rref = propagator.p_dict['Rstore']
Cref = propagator.p_dict['Cstore']
propagator.set_parameter('Rstore',Rref*np.sqrt(RC_scale))
propagator.set_parameter('Cstore',Cref*np.sqrt(RC_scale))

fig, _, _ = plotter.bode_plot(propagator,indicate_freq=0.1)
fig.show()

#%%

fig, ax = plt.subplots()

for k in range(states.shape[1]-2) :
    ax.plot(states[:,k])

fig.show()

#%% ALL BELOW ARE REMNANTS OF OLD CODE
    
#%%
T1=200
train_error, pred_error, snr, noise, my_esn = run_test(0.002,Tfit=T1,Tpred=T1,Tnoise=100,fmin=0.3,generate_plots=True,memory_variation=0.25)

#%%
A33 = my_esn.sample_memory()

#%%
Ndf=7
df_array = 10**np.linspace(-2,-4,Ndf)
mem_var_vals = [0.0, 0.25, 0.5]
fmin=0.3
param_dicts = [{'n':Ndf,'df':df_array,'memory_variation':[mem_var]*Ndf, 'fmin':[fmin]*Ndf} for mem_var in mem_var_vals]

N=10
signal_diffs = np.zeros((len(mem_var_vals),N,Ndf))
noise_vars = np.zeros((len(mem_var_vals),N,Ndf))
train_errors = np.zeros((len(mem_var_vals),N,Ndf))
pred_errors = np.zeros((len(mem_var_vals),N,Ndf))

for k, param_dict in enumerate(param_dicts):    
    signal_diffs[k], noise_vars[k], train_errors[k], pred_errors[k] = repeat_runs(N,param_dict)
  
#%%
Ndf=1
# Focus on a single df here
df_array = [2e-3] # GHz
mem_var_vals = [0.0, 0.25, 0.5]
mem_var_vals = [0.0, 0.25, 0.5] # One at a time
fmin=0.3
param_dicts = [{'n':Ndf,'df':df_array,'memory_variation':[mem_var]*Ndf, 'fmin':[fmin]*Ndf} for mem_var in mem_var_vals]

N=100 # A hundred simulations with T1=300 takes about 1 hour
signal_diffs = np.zeros((len(mem_var_vals),N,Ndf))
noise_vars = np.zeros((len(mem_var_vals),N,Ndf))
train_errors = np.zeros((len(mem_var_vals),N,Ndf))
pred_errors = np.zeros((len(mem_var_vals),N,Ndf))

for k, param_dict in enumerate(param_dicts):    
    signal_diffs[k], noise_vars[k], train_errors[k], pred_errors[k] = repeat_runs(N,param_dict)
    
        
#%% Analyze the results from the parameter study
study_name='XXXSecondstudy_f0_0.3GHz_'

nature_single = 89.0 / 25.4
nature_double = 183.0 / 25.4
nature_full = 247.0 / 25.4

# 1. Signal to noise ratio plot 
def plot_distance_v_param(means, stds, dfs, param_vals,
                          param_name,ylabel='SNR',
                          ax=None, label_font_size=11, unit_font_size=10,
                          title=None, xmin=1e-4,xmax=0.01, ymax=200,
                          ylogscale=False):
    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=(nature_single, nature_single))

    colors = [cm.viridis(x) for x in np.linspace(0, 1, len(param_vals))]

    for i in range(len(param_vals)):
        noise = param_vals[i]
        mu = means[i]
        sigma = stds[i]
        if noise != 'Random':
            ax.semilogx(dfs, mu, color=colors[i], label=noise, lw=1)
        else:
            ax.semilogx(dfs, mu, color=colors[i], label='Random walk',
                        lw=1)
        ax.fill_between(dfs,
                        [m+s for m, s in zip(mu, sigma)],
                        [m-s for m, s in zip(mu, sigma)],
                        facecolor=colors[i], alpha=0.2)

    if ylogscale :
        ax.set_yscale('log')
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(0, ymax)
    ax.set_title(title, fontsize=label_font_size)
    ax.tick_params(labelsize=unit_font_size)
    ax.set_xlabel('Delta f (GHz)', fontsize=label_font_size)
    ax.set_ylabel(ylabel, fontsize=label_font_size)

    handles, labels = ax.get_legend_handles_labels()

    l = ax.legend(handles,
                  labels,
                  loc='best',
                  fontsize=label_font_size,
                  handlelength=0,
                  handletextpad=0,
                  title=f'{param_name}:')
    l.get_title().set_fontsize(label_font_size)
    for i, text in enumerate(l.get_texts()):
        text.set_color(colors[i])
    for handle in l.legendHandles:
        handle.set_visible(False)
    l.draw_frame(False)
    plt.tight_layout()
    return fig, ax


snr = signal_diffs/noise_vars
snr = np.sqrt(signal_diffs)/np.sqrt(noise_vars)

snr_mean = np.mean(snr, axis=1)
snr_std = np.std(snr, axis=1)

fig, ax = plot_distance_v_param(snr_mean, snr_std, df_array, mem_var_vals[:3], 'Mem. noise',ymax=20)
#plotter.save_plot(fig,study_name+'snr',PLOT_PATH)

#%%

pred_mean = np.mean(pred_errors, axis=1)
pred_std = np.std(pred_errors, axis=1)

fig, ax = plot_distance_v_param(pred_mean, pred_std, df_array, mem_var_vals, 'Mem. noise',ymax=0.04,
                                ylogscale=True,ylabel='Pred. error')
plotter.save_plot(fig,study_name+'pred_error',PLOT_PATH)

train_mean = np.mean(train_errors, axis=1)
train_std = np.std(train_errors, axis=1)

fig, ax = plot_distance_v_param(train_mean, train_std, df_array, mem_var_vals, 'Mem. noise',ymax=0.04,
                                ylogscale=True,ylabel='Train. error')
plotter.save_plot(fig,study_name+'train_error',PLOT_PATH)

noise_mean = np.mean(noise_vars, axis=1)
noise_std = np.std(noise_vars, axis=1)

fig, ax = plot_distance_v_param(noise_mean, noise_std, df_array, mem_var_vals, 'Mem. noise',ymax=0.1,
                                ylogscale=False,ylabel='Noise variance')
plotter.save_plot(fig,study_name+'noise_var',PLOT_PATH)

#%% Plot the SNR as a function of memory variation
snr = np.sqrt(signal_diffs/noise_vars)

snr_mean = np.mean(snr, axis=1)
snr_std = np.std(snr, axis=1)


fig, ax = plt.subplots()

ax.errorbar(mem_var_vals[:3],np.squeeze(snr_mean),np.squeeze(snr_std),fmt='o',ecolor='k',capsize=3.0)

ax.set_xlabel('Memory variation')
ax.set_ylabel('SNR (mean/std. dev.)')
ax.set_ylim(0,12)
ax.set_xlim(-0.05,0.55)

fig.show()

#%% Need to check the Bode plot for this device
propagator = physics.Device('../parameters/device_parameters_1ns.txt')
fig, _, _ = plotter.bode_plot(propagator,indicate_freq=0.25)
fig.show()

#%% Check how variance in memory change Bode plot
devices = {}
noise = 0.5
device = physics.Device('../parameters/device_parameters.txt')
Rstore_ref = device.p_dict['Rstore']
for k in [-1, 0, 1] :
    devices[k]=physics.Device('../parameters/device_parameters.txt')
    devices[k].set_parameter('Rstore', Rstore_ref + k*noise*Rstore_ref)
    print(devices[k].p_dict['Rstore'])
    
fig, _, _ = plotter.bode_multi_plot(devices,indicate_freq=0.3)
fig.show()

#%% Specify the network
Nreservoir = 100
#SEED=10
# Get me a network
my_esn = esn.EchoStateNetwork(Nreservoir,seed=SEED,sparsity=0.9)
# Specify a standard device for the hidden layers
propagator = physics.Device('../parameters/device_parameters.txt')
# %% Show netork
my_esn.show_network(savefig=True, arrow_size=5,font_scaling=2)

# %% Look at specific solutions

# Reiterate these constants
teacher_scaling=1.0
beta = 10 # regularization, 10 is default
T=2000
fmin=0.1
fmax=0.2
dT=50
# Training paraemters
Tfit = 200. # spend two thirds on training
scl = 2.0

# train and test a network
my_esn.specify_network(0.6,
                       1.5,
                       0.0,
                       teacher_scaling, feedback=False)

# Specify device
my_esn.assign_device(propagator)
my_esn.randomize_memory(noise=0.1)

frequency_control, frequency_output, tseries = freqeuncy_signal_generator(T,200,fmin,fmax,dT) 
# Specify explicit signals by handle
def teacher_signal(signal_scale) :
    handle = interp1d(tseries,frequency_control*signal_scale,axis=0,fill_value=frequency_control[-1],bounds_error=False)
    return handle 

def input_signal(signal_scale) :
    handle = interp1d(tseries,frequency_output*signal_scale,axis=0)  
    return handle

def bias_signal(signal_scale) :
    return lambda t : signal_scale

my_esn.specify_inputs(input_signal,bias_signal,teacher_signal)
# Set the system delay time
#my_esn.set_delay(0.5) # units of ns

#%% Harvest states
tseries_train, states_train, _ = my_esn.harvest_states(Tfit,teacher_forcing=False)
# Generate the target signal
teacher_handle = teacher_signal(my_esn.Imax*teacher_scaling)
teacher_train = teacher_handle(tseries_train)



#%%
# Fit output weights
pred_train, train_error = my_esn.fit(states_train, teacher_train,beta=beta)

# Test trained network by running scl times Tfit
scl = 2.0
#my_esn.set_delay(0.5) # units of ns
# %%
tseries_test, pred_test, movie_series, plot_series = my_esn.predict(Tfit,scl*Tfit,output_all=True)
# Generate the target signal
teacher_handle = teacher_signal(my_esn.Imax*teacher_scaling)
teacher_test = teacher_handle(tseries_test)
pred_error = np.sqrt(np.mean((pred_test - teacher_test)**2))/my_esn.Imax
   
#%%            
plt.plot(tseries_test,pred_test)
plt.plot(tseries_train,pred_train)
plt.plot(tseries_train,teacher_train,'k--')
plt.plot(tseries_test,teacher_test,'k--')

plt.show()


# %% Loop over hyperparameters 

# Hyperparameters
spectral_radii = np.arange(0.6,0.9,step=0.2)
input_scaling = np.arange(1.0,2.6,step=0.5)
#bias_scaling = np.arange(0.1,0.2,step=0.2) 
teacher_scaling=np.arange(0.6, 1.5, step=0.2)
beta = 100 # regularization
bias_scaling=0.0

# Training paraemters
Tfit = 600. # spend two thirds on training
scl = 1.5
teacher_handle = teacher_signal(my_esn.Imax*teacher_scaling)

# Save results on file
with open('training_result.txt','w') as f :
    f.write('Pred. error, train error, spectral radius, input scaling, bias_scaling\n')
    f.close()
    
for k in range(len(teacher_scaling)) :
    for l in range(len(spectral_radii)) :
        for m in range(len(input_scaling)) :
            # train and test a network
            my_esn.specify_network(spectral_radii[l],
                                   input_scaling[m],
                                   bias_scaling,
                                   teacher_scaling[k])
            
            # Specify device
            my_esn.assign_device(propagator)
            # Specify explicit signals by handle
            my_esn.specify_inputs(input_signal,bias_signal,teacher_signal)
            # Set the system delay time
            my_esn.set_delay(0.5) # units of ns
            
            # Harvest states
            tseries_train, states_train, teacher_train = my_esn.harvest_states(Tfit)
            # Fit output weights
            pred_train, train_error = my_esn.fit(states_train, teacher_train,beta=beta)
            # Test trained network by running scl times Tfit
            tseries_test, pred_test = my_esn.predict(Tfit,scl*Tfit)
            # Generate the target signal
            teacher_test = teacher_handle(tseries_test)
            pred_error = np.sqrt(np.mean((pred_test - teacher_test)**2))/my_esn.Imax
            #print('Prediction error:',pred_error)
            # Write parameters and errors to file
            with open('training_result.txt','a') as f :
                f.write(f'{pred_error:.3f},{train_error:.3f},{spectral_radii[l]:.1f},{input_scaling[m]:.1f},{teacher_scaling[k]:.1f}\n')
                f.close()
                
                

# %%

# At this point, we send all info to the movie_maker to construct our movie of
# Copy DataFrame
movie_copy = movie_series.copy()
plot_copy = plot_series.copy()

time_interval=(750,870)

#select_result = plot_copy[(plot_copy["Time"]>=time_interval[0]) & (plot_copy["Time"]<=time_interval[1])]

plotter.plot_nodes(plot_copy,['H2','H3','H5'],onecolumn=True,time_interval=time_interval)
plotter.plot_nodes(plot_copy,['K0','K1','K3','K4'],onecolumn=True,time_interval=time_interval)

plotter.visualize_scaled_result(plot_copy,['H3-Iinh','H3-Iexc'],scaling=[-2,1],time_interval=time_interval)

# %%

plotter.plot_sum_nodes(plot_copy,['I','H','K','O'],'Pout',time_interval=time_interval)


# %%

# time frame to use
tstart = 770
tend = 870
idx_start = np.nonzero(tseries_test>tstart)[0][0]-1 # include also the start
idx_end = np.nonzero(tseries_test>tend)[0][0]
movie_selection = movie_copy.iloc[idx_start:idx_end]
                                  
my_esn.produce_movie(movie_selection)

# %%

my_esn.show_network(layout='spring')

# %% Need a spectrogram to visualize the frequency of the signal
def draw_spectogram(data):
    plt.specgram(data,Fs=2,NFFT=64,noverlap=32,cmap=plt.cm.bone,detrend=lambda x:(x-250))
    plt.gca().autoscale('x')
    plt.ylim([0,0.5])
    plt.ylabel("freq")
    plt.yticks([])
    plt.xlabel("time")
    plt.xticks([])

plt.figure(figsize=(7,1.5))
draw_spectogram(teacher_train.flatten())
plt.title("training: target")
plt.figure(figsize=(7,1.5))
draw_spectogram(pred_train.flatten())
plt.title("training: model")

# %%

plt.figure(figsize=(7,1.5))
draw_spectogram(teacher_test.flatten())
plt.title("test: target")
plt.figure(figsize=(7,1.5))
draw_spectogram(pred_test.flatten())
plt.title("test: model")