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
    ax1.set_xlim(tseries_test[0]-500, tseries_test[0]+2000)
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
    ax.hist(tau,rwidth=0.75)
    ax.hist(np.array([5.]*100),rwidth=5)
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

def plot_spectrogram(data,dt,fcutoff,ax=None,xstart=0,NFFT=128,noverlap=32) :
    if ax is None :
        fig, ax = plt.subplots() # create new if no one is passed
    else :
        fig = None
    
    # Caclulate the size of the bins to get the xextent
    N = len(data)
    bin_pos=[]
    k = NFFT // 2
    while k + NFFT//2 < N :        
        bin_pos.append(k)
        k += NFFT-noverlap
    # bin width is NFFT-noverlap
    xextent = (xstart+noverlap//2, xstart + bin_pos[-1]+(NFFT-noverlap)//2)    
    spectrum, freqs, t, im = ax.specgram(np.squeeze(data),Fs=int(1/dt),NFFT=NFFT,noverlap=noverlap,xextent=xextent)
    print(freqs)
    fmin_idx = np.flatnonzero(freqs>fcutoff)[0] # first index BELOW fmin
    max_args = np.argmax(spectrum[fmin_idx:],axis=0) + fmin_idx
    max_freq = freqs[max_args]
    scale_t = xstart+t
    ax.plot(scale_t,max_freq,'ro')

    return fig, ax, t, max_freq, freqs[1]-freqs[0]
    
def fspec(data,dt,fcutoff,plotname):
    fig, ax, t, max_freq, df = plot_spectrogram(data,dt,fcutoff)
    plotter.save_plot(fig,'spec'+plotname,PLOT_PATH)
    return t, max_freq, df
    
def evaluate_nw(ts,pred,signal_handle,fcutoff,control_handle,plotname,Nmax=6):
    teacher = signal_handle(ts)
    # Now we should generate the frequency spectrum for each and compare using
    # least squares to find the error   
    dt = ts[1]-ts[0] # constant sampling time is used
    t_freqs, f_pred, df = fspec(pred,dt,fcutoff,'pred_'+plotname) 
    t_teach, f_teach, _ = fspec(teacher,dt,fcutoff,'teach_'+plotname)
    
    print(t_freqs)
    print(np.where(f_pred==f_teach))
    print(np.where(abs(f_pred-f_teach)==abs(df)))
    print(np.where(abs(f_pred-f_teach)==abs(2*df)))
    
    hits = []
    for k in range(0,Nmax) :
        hits.append(len(np.where(abs(f_pred-f_teach)==abs(k*df))[0]))
    # when we have calculated the number we want, we add the remaining
    hits.append(len(t_freqs)-sum(hits))
    
    plot_control_freq(ts,control_handle,t_freqs,f_pred,f_teach,plotname)
    
    # Now compare these signals
    freq_error = np.sqrt(np.mean((f_pred-f_teach)**2))
    # Tweak!
    #freq_error = f_pred[0]
    return freq_error, hits

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
    
def single_test(fmin=0.005,fmax=0.1,Nf=2,dT=50,Tfit=2000,Tpred=2000,Nreservoir=100,sparsity=10,
             transient=0,plotname='freqgen',generate_plots=True,seed=None,noise=0.,
             spectral_radius=0.25,input_scaling=0.5,bias_scaling=0.1,diagnostic=False,
             teacher_scaling=0.5, memory_variation=0.0, memscale=1.0, dist='uniform',
             beta=0.1,fcutoff=0.015,states_train=None,delay=1.,nsave=1000) :

   # Specify device, start with the 1 ns device that we will scale
    propagator = physics.Device('../parameters/device_parameters_1ns_revisedtest.txt')
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
    new_esn.set_delay(delay,nsave)
    
    if memory_variation > 0.0 :
        new_esn.randomize_memory(memory_variation,dist,seed)
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
    freq_error, hits = evaluate_nw(tseries_test,pred_test,signal_handle,fcutoff,control_handle,plotname)
    
    if diagnostic :
        return train_error, pred_error, freq_error, end-start, states_train, hits, new_esn, tseries_train, pred_train, tseries_test, pred_test, signal_handle, control_handle
    else :
        return train_error, pred_error, freq_error, end-start, states_train, hits #, new_esn, tseries_train, pred_train, tseries_test, pred_test, (t1_handle, t2_handle)

    
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
    Nhits = 7
    timing=np.zeros((N,param_dict['n']))
    freq_errors=np.zeros((N,param_dict['n']))
    train_errors=np.zeros((N,param_dict['n']))
    pred_errors=np.zeros((N,param_dict['n']))
    hits=np.zeros((N,param_dict['n'],Nhits))
    
    # Set 
    plt.ioff()
    
    print(param_dict)
    kwargs = {}
    for k, v in param_dict.items():
        if k != 'n' and v[0] != None:
            kwargs[k] = v[0]
            
    try :
        # See if this has already been run
        train_errors, pred_errors, freq_errors, timing, hits = load_dataset(N,kwargs)
        
    except :
        # Otherwise run it now
        for m in range(0,N) :
            plotname = generate_figurename(kwargs,suffix=f'_{m}')
            try :
                train_errors[m], pred_errors[m], freq_errors[m], timing[m], states, hits[m] = single_test(plotname=plotname,seed=m+seed_zero,**kwargs)
            except RuntimeError :
                print(f'singletest() failed with m={m}, seed={m+seed_zero}')
                train_errors[m,:]=np.nan; pred_errors[m,:]=np.nan 
                freq_errors[m,:]=np.nan; timing[m,:]=np.nan; hits[m,:,:]=np.nan
                
            print('Status update, m =',m)
            plt.close('all')
    
        if save:
            save_dataset((train_errors, pred_errors, freq_errors, timing, hits), N, kwargs)


    return train_errors, pred_errors, freq_errors, timing, hits

def validation_runs(N, param_dict, save=True, seed_zero=0) :
    return
    
def save_training(arrays,N,kwargs,seed_zero):
    filename = f'train_seed{seed_zero}_' + generate_filename(N,kwargs) 
        
    # save to a pickle file
    with open(TRAIN_PATH / filename,'ab') as f :
        for a in arrays :
            pickle.dump(a,f)

def load_training(N,kwargs,seed_zero=0) :
    filename = f'train_seed{seed_zero}_' + generate_filename(N,kwargs) 
    states = []
    with open(TRAIN_PATH / filename,'rb') as f :
        # Items read sequentially
        for k in range(N) :
            states[k] = pickle.load(f)
    return states
    
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
        hits = pickle.load(f)
        # now it's empty
    
    return train_errors, pred_errors, freq_errors, timing, hits
   



#%% Test code for signal generation
T = 1000 # ns
dT = 50 # ns, average period length of constant frequency
fmin = 1/200 # GHz
fmax = 1/10# GHz

frequency_input, frequency_output, tseries = frequency_step_generator(T,fmin,fmax,dT,res=20) 

# A fourier transform to check the frequencies


print(f'Generated a time series from 0 to {T} ns with {len(tseries)} elements')

if True :
           
    Nmax = 2999
    fig, (ax1, ax2) = plt.subplots(1,2)
    
    ax1.plot(tseries[:Nmax],frequency_input[:Nmax])
    ax2.plot(tseries[:Nmax],frequency_output[:Nmax])
    
    ax1.set_xlabel('Time (ns)')
    ax2.set_xlabel('Time (ns)')

    ax1.set_ylabel('Frequency control')
    ax2.set_ylabel('Target signal')
        
    plt.tight_layout()
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

# At the moment the delay is the same as the system timescale in esn.py. This means that the 
# the delay cannot be changed independently of, say, the savestep of the system, which creates problem
# for the frequency analysis.
new_delay = 1.
set_seed=18
plt.ion()
train_error, pred_error, freq_error_0, time_used, states, hits, new_esn, tseries_train_0, pred_train_0, tseries_test_0, pred_test_0, signal_handle, control_handle = single_test(diagnostic=True,fmin=0.005,fmax=0.15,beta=0.01,memscale=5,memory_variation=0.8,seed=set_seed,delay=new_delay)
print(hits)
train_error, pred_error, freq_error_0, time_used, states, hits, new_esn, tseries_train_1, pred_train_1, tseries_test_1, pred_test_1, signal_handle, control_handle = single_test(diagnostic=True,fmin=0.005,fmax=0.15,beta=0.01,memscale=5,memory_variation=0.0,seed=set_seed,delay=new_delay)
print(hits)
#train_error, pred_error, freq_error_0, time_used, states, new_esn, tseries_train_1, pred_train_1, tseries_test_1, pred_test_1, signal_handle, control_handle = single_test(diagnostic=True,beta=0.01,memscale=5,memory_variation=0.0,seed=4)

#%%

plot_memory_dist(new_esn, 'hist')

#%% Compare spectrogram, teacher and prediction
teacher = signal_handle(tseries_test_0)
dt = tseries_test_0[1]-tseries_test_0[0]
fmax = 0.1
xmin = 2000
xmax = 4000

fig, axs = plt.subplots(2,1,figsize=(nature_single,nature_single),sharex=True)

_, _, t, max_freq_test0 = plot_spectrogram(pred_test_0,dt,fcutoff=0.015,ax=axs[0],xstart=xmin)
_, _, t, max_freq_teacher = plot_spectrogram(teacher,dt,fcutoff=0.015,ax=axs[1],xstart=xmin)

#axs[0].set_xlim(xmin,xmax)
axs[0].set_ylim(0,fmax)
axs[1].set_ylim(0,fmax)

# Tweak the tick labels
y1_labels = [f'{val:.2f}' for val in np.arange(0.02,0.12,0.02)]
y1_labels.insert(0,'')
axs[0].set_yticklabels(y1_labels)
#y1_labels[0] = ''
#axs[0].set_yticklabels(y1_labels) # fick labels

axs[0].tick_params(direction='inout',top=True)
axs[1].tick_params(direction='inout',top=True)

axs[1].set_xlabel('Time (ns)')
axs[1].set_ylabel('Teacher freq. (GHz)')
axs[0].set_ylabel('Pred. freq. (GHz)')
plt.subplots_adjust(hspace=0,left=0.2,right=0.95,bottom=0.15,top=0.95)
#plt.tight_layout()
plt.show()

#%% Get both results into a control frequency plot
def plot_control_multifreq(ts,control_handle,tf,fp,ft,plotname='',labels=None):
    fig, ax1 = plt.subplots(figsize=(nature_single,nature_single))
    #ax1.plot(ts,control_handle(ts),label='control')
    if labels is None :
        labels = ['']*len(fp)
        
    markers = ['b^','ro']    
    for k,f in enumerate(fp) :
        ax1.plot(tf+ts[0],f,markers[k],label=labels[k]) # tf is measured from 0
        
    ax1.plot(tf+ts[0],ft,'kx',label='teacher')
    ax1.set_xlabel('Time (ns)')
    ax1.set_ylabel('Frequency control')
    ax1.legend()
    
    plotter.save_plot(fig,'control_'+plotname,PLOT_PATH)
    #plt.close(fig)
    
    return fig, ax1

_, _, t, max_freq_test1 = plot_spectrogram(pred_test_1,dt,fcutoff=0.015,ax=axs[0],xstart=xmin)

fmax_cut=0.12
max_freq_test1 = np.clip(max_freq_test1,None,fmax_cut)

fig, ax = plot_control_multifreq(tseries_test_0,control_handle,t,[max_freq_test1,max_freq_test0],max_freq_teacher,labels=[r'fixed-$\tau$',r'dist-$\tau$'])

ax.set_ylim(0,fmax_cut)
ax.set_xlim(xmin,xmax)
ax.grid(True)
plt.tight_layout()

#%% Compare the traces, single vs. dist

# Get the colors 
my_dpi = 300
prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']
color_single_train=colors[0]
color_dist_train=colors[1]
color_teacher=colors[2]
lw_test = 1.0
lw_signal = 0.75

fig, (ax1, ax2) = plt.subplots(2,1,figsize=(nature_full,nature_single*1),sharex=True)

# Time series from network
# ax1 should be fixed memory
ax1.plot(tseries_train_1[:],pred_train_1[:,0],linewidth=lw_test,color=color_dist_train,label=r'fixed-$\tau$')
ax1.plot(tseries_test_1[:],pred_test_1[:,0],linewidth=lw_test,color=color_dist_train)
# ax2 is distributed memory
ax2.plot(tseries_train_0[:],pred_train_0[:,0],linewidth=lw_test,color=color_single_train,label=r'dist-$\tau$')
ax2.plot(tseries_test_0[:],pred_test_0[:,0],linewidth=lw_test,color=color_single_train)

# Teacher signal, training + prediction
teacher_series=signal_handle(tseries_test_0)
ax1.plot(tseries_train_0,signal_handle(tseries_train_0),'--',linewidth=lw_signal,color=color_teacher)
ax2.plot(tseries_train_0,signal_handle(tseries_train_0),'--',linewidth=lw_signal,color=color_teacher)
ax2.plot(tseries_test_0,teacher_series,'--',linewidth=lw_signal,color=color_teacher)
ax1.plot(tseries_test_0,teacher_series,'--',linewidth=lw_signal,color=color_teacher)

# Control signal in both plots
ax1.plot(tseries_train_0,control_handle(tseries_train_0),'k',linewidth=0.5,label='control input')
ax1.plot(tseries_test_0,control_handle(tseries_test_0),'k',linewidth=0.5)
ax2.plot(tseries_train_0,control_handle(tseries_train_0),'k',linewidth=0.5,label='control input')
ax2.plot(tseries_test_0,control_handle(tseries_test_0),'k',linewidth=0.5)
 
vistrain=300
vispred=1000
ax2.set_xlabel('Time (ns)')
ax1.set_ylabel('Output signal (nA)')
ax2.set_ylabel('Output signal (nA)')
ax1.set_ylim(0,teacher_series[:,0].max()+25)
ax1.set_xlim(tseries_test_0[0]-vistrain, tseries_test_0[0]+vispred)
ax2.set_ylim(0,teacher_series[:,0].max()+25)
ax2.set_xlim(tseries_test_0[0]-vistrain, tseries_test_0[0]+vispred)

from matplotlib import patches
# Create a Rectangle patch
rect1 = patches.Rectangle((tseries_test_0[0]-500, 0), 500, teacher_series[:,0].max()+25, linewidth=1, edgecolor='none', facecolor='gray',alpha=0.4)
rect2 = patches.Rectangle((tseries_test_0[0]-500, 0), 500, teacher_series[:,0].max()+25, linewidth=1, edgecolor='none', facecolor='gray',alpha=0.4)

# Add the patch to the Axes
ax1.add_patch(rect1)
ax2.add_patch(rect2)

ax1.legend(loc='upper right')
ax2.legend(loc='upper right')
#plotter.save_plot(fig,'trace_'+plotname,PLOT_PATH)
#plt.close(fig)

plt.tight_layout()

plt.show()
#%% Compare the traces, single vs. dist with both attempts in the same panel
# USED IN PAPER!

# Get the colors 
my_dpi = 300
prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']
color_single_train=colors[0]
color_dist_train=colors[1]
color_teacher=colors[2]
lw_test = 1.0
lw_signal = 0.75

fig, (ax1, ax2) = plt.subplots(2,1,figsize=(nature_double*1,nature_single*1),sharex=True)

# Time series from network
# ax1 should be fixed memory
ysep = 250
ax1.plot(tseries_train_1[:],pred_train_1[:,0]+ysep,linewidth=lw_test,color=color_dist_train,label=r'single-$\tau$')
ax1.plot(tseries_test_1[:],pred_test_1[:,0]+ysep,linewidth=lw_test,color=color_dist_train)
# ax2 is distributed memory
ax1.plot(tseries_train_0[:],pred_train_0[:,0],linewidth=lw_test,color=color_single_train,label=r'many-$\tau$')
ax1.plot(tseries_test_0[:],pred_test_0[:,0],linewidth=lw_test,color=color_single_train)

# Teacher signal, training + prediction
teacher_series=signal_handle(tseries_test_0)
#ax1.plot(tseries_train_0,signal_handle(tseries_train_0),'--',linewidth=lw_signal,color=color_teacher)
ax2.plot(tseries_train_0,signal_handle(tseries_train_0),'-',linewidth=lw_signal,color=color_teacher,label='target signal')
ax2.plot(tseries_test_0,teacher_series,'-',linewidth=lw_signal,color=color_teacher)
#ax1.plot(tseries_test_0,teacher_series,'--',linewidth=lw_signal,color=color_teacher)

# Control signal in both plots
#ax1.plot(tseries_train_0,control_handle(tseries_train_0),'k',linewidth=0.5,label='control input')
#ax1.plot(tseries_test_0,control_handle(tseries_test_0),'k',linewidth=0.5)
ax2.plot(tseries_train_0,control_handle(tseries_train_0),'k',linewidth=0.5,label='control input')
ax2.plot(tseries_test_0,control_handle(tseries_test_0),'k',linewidth=0.5)
 
vistrain=300
vispred=1000
ax2.set_xlabel('Time (ns)')
ax1.set_ylabel('Output signal (nA)')
ax2.set_ylabel('Output signal (nA)')
ax1_extray = 30
ax1_negy = -50
ax2_extray = ax1_extray
a=1.2
ax1_extray += 1.35*ysep
ax1.set_ylim(ax1_negy,teacher_series[:,0].max()+ax1_extray)
ax1.set_xlim(tseries_test_0[0]-vistrain, tseries_test_0[0]+vispred)
ax2.set_ylim(0,teacher_series[:,0].max()+ax2_extray)
ax2.set_xlim(tseries_test_0[0]-vistrain, tseries_test_0[0]+vispred)

ax1.text(2020,teacher_series[:,0].max()+a*ysep,'prediction',fontsize=10,fontstyle='italic')
ax1.text(1840,teacher_series[:,0].max()+a*ysep,'training',fontsize=10,fontstyle='italic')
from matplotlib import patches
# Create a Rectangle patch
rect1 = patches.Rectangle((tseries_test_0[0]-500, 0), 500, teacher_series[:,0].max()+ax1_extray, linewidth=1, edgecolor='none', facecolor='gray',alpha=0.4)
rect2 = patches.Rectangle((tseries_test_0[0]-500, 0), 500, teacher_series[:,0].max()+ax2_extray, linewidth=1, edgecolor='none', facecolor='gray',alpha=0.4)

# Add the patch to the Axes
ax1.add_patch(rect1)
ax2.add_patch(rect2)

# Add grids
#ax1.grid(True)
#ax2.grid(True)

# Panel labels
ax1.text(1640,225+a*ysep,'(a)')
ax2.text(1640,225,'(b)')

ax1.legend(loc='lower left')
ax2.legend(loc='lower left')

ax1.tick_params(axis='x',direction='inout',top=True,length=5.0)
ax2.tick_params(axis='x',direction='inout',top=True,length=5.0)
#ax1.set_yticklabels(['','100','200'])

#plt.close(fig)

plt.tight_layout()
plt.subplots_adjust(hspace=0.00)
plotter.save_plot(fig,'trace_seed18',PLOT_PATH)
plt.show()
#%% Look at the traces

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

#%% Perform a grid test to study spectral radius and scaling (could also do sparsity)
rho_vals = [0.25, 0.5, 0.75, 1.0, 1.25]
scale_vals = [0.5, 0.75, 1.0, 1.25, 1.5]
Ndf=1
param_dicts =[{'n':Ndf,'teacher_scaling':[scale],'input_scaling':[scale],'spectral_radius':[rho],'beta':[0.01],'memscale':[5.],'memory_variation':[0.8]} for scale in scale_vals for rho in rho_vals]

N=10 # Each simulation is about 8 mins with a memory distribution, 1 min with plain memory

freq_errors = np.zeros((len(param_dicts),N,Ndf))
timing = np.zeros((len(param_dicts),N,Ndf))
train_errors = np.zeros((len(param_dicts),N,Ndf))
pred_errors = np.zeros((len(param_dicts),N,Ndf))

for k, param_dict in enumerate(param_dicts):    
    train_errors[k], pred_errors[k], freq_errors[k], timing[k] = repeat_runs(N,param_dict,seed_zero=5)
    
#%% Analyze the results of the first grid search

# The first result that we want is the mean error vs scale and rho
Nscale = len(scale_vals)
Nrho = len(rho_vals)
mean_error = np.zeros((Nscale,Nrho))
for k in range(0,Nscale) :
    for l in range(0,Nrho) :
        mean_error[k,l] = freq_errors[Nscale*k+l].mean()
        print(Nscale*k+l)
    
#%% Show a pcolor plot of this

fig, ax = plt.subplots()

scale, rho = np.meshgrid(scale_vals,rho_vals)
im = ax.pcolor(scale,rho,mean_error.T)

ax.set_yticks(rho_vals)
ax.set_xticks(scale_vals)
ax.set_ylabel('Spectral radius')
ax.set_xlabel('Input/teacher scaling')
plt.colorbar(im,ax=ax,label='Avg. freq. error')

plt.show()

#%% Show also all results with std as well

# How to name the entries on the xaxis
x_labels = [f'scale:{scale}, rho:{rho}' for scale in scale_vals for rho in rho_vals]

fig, ax = plt.subplots()

mean_error_lin = np.squeeze(freq_errors.mean(axis=1))
std_error = np.squeeze(freq_errors.std(axis=1))
xvals = np.arange(0,Nrho*Nscale)
ax.errorbar(xvals,mean_error_lin,std_error,ecolor='black',elinewidth=0.5,capsize=1.)

ax.set_xticks(xvals,labels=x_labels,rotation='vertical')
ax.set_ylabel('Freq. error')
plt.tight_layout()
plt.show()

#%% Perform a new grid test after revising time scales to study spectral radius and delay (could also do sparsity)
rho_vals = [0.15, 0.25, 0.5]
delay_vals = [0.5, 1., 2., 3.]
Ndf=1
scale=0.5 # From earlier grid search
param_dicts =[{'n':Ndf,'teacher_scaling':[scale],'input_scaling':[scale],'spectral_radius':[rho],'beta':[0.01],'memscale':[5.],'memory_variation':[0.8],'delay':[delay]} for delay in delay_vals for rho in rho_vals]

N=10 # Each simulation is about 8 mins with a memory distribution, 1 min with plain memory

freq_errors = np.zeros((len(param_dicts),N,Ndf))
timing = np.zeros((len(param_dicts),N,Ndf))
train_errors = np.zeros((len(param_dicts),N,Ndf))
pred_errors = np.zeros((len(param_dicts),N,Ndf))

for k, param_dict in enumerate(param_dicts):    
    train_errors[k], pred_errors[k], freq_errors[k], timing[k] = repeat_runs(N,param_dict,seed_zero=5)

#%% Analyze the results of the second grid search

# The first result that we want is the mean error vs scale and rho
Ndelay = len(delay_vals)
Nrho = len(rho_vals)
mean_error = np.zeros((Ndelay,Nrho))

for k in range(0,Ndelay) :
    for l in range(0,Nrho) :
        mean_error[k,l] = freq_errors[Nrho*k+l].mean()
        print(Nrho*k+l)
    
#%% Show a pcolor plot of this

fig, ax = plt.subplots()

delay, rho = np.meshgrid(delay_vals,rho_vals)
im = ax.pcolor(delay,rho,mean_error.T)

ax.set_yticks(rho_vals)
ax.set_xticks(delay_vals)
ax.set_ylabel('Spectral radius')
ax.set_xlabel('Delay (ns)')
plt.colorbar(im,ax=ax,label='Avg. freq. error')

plt.show()

#%% Show also all results with std as well

# How to name the entries on the xaxis
x_labels = [f'delay:{delay}, rho:{rho}' for delay in delay_vals for rho in rho_vals]

fig, ax = plt.subplots()

mean_error_lin = np.squeeze(freq_errors.mean(axis=1))
std_error = np.squeeze(freq_errors.std(axis=1))
xvals = np.arange(0,Nrho*Ndelay)
ax.errorbar(xvals,mean_error_lin,std_error,ecolor='black',elinewidth=0.5,capsize=1.)

ax.set_xticks(xvals,labels=x_labels,rotation='vertical')
ax.set_ylabel('Freq. error')
plt.tight_layout()
plt.show()


#%% Compare performance between dist and single tau's
Ndf = 1

param_dicts  =[{'n':Ndf,'teacher_scaling':[0.5],'input_scaling':[0.5],'spectral_radius':[0.5],'beta':[0.01],'memscale':[5.],'delay':[1.]}]
param_dicts  +=[{'n':Ndf,'teacher_scaling':[0.5],'input_scaling':[0.5],'spectral_radius':[0.5],'beta':[0.01],'memscale':[5.],'memory_variation':[0.8],'delay':[1.]}]

N=10 # Each simulation is about 8 mins with a memory distribution, 1 min with plain memory

freq_errors = np.zeros((len(param_dicts),N,Ndf))
timing = np.zeros((len(param_dicts),N,Ndf))
train_errors = np.zeros((len(param_dicts),N,Ndf))
pred_errors = np.zeros((len(param_dicts),N,Ndf))

for k, param_dict in enumerate(param_dicts):    
    train_errors[k], pred_errors[k], freq_errors[k], timing[k] = repeat_runs(N,param_dict,seed_zero=42)
    
#%% Compare performance between dist and single tau's
#################****************************
###################**************************
Ndf = 1

param_dicts  =[{'n':Ndf,'teacher_scaling':[0.5],'input_scaling':[0.5],'spectral_radius':[0.25],'beta':[0.01],'memscale':[5.],'memory_variation':[0.8],'delay':[1.],'fmax':[0.15]}]
param_dicts  +=[{'n':Ndf,'teacher_scaling':[0.5],'input_scaling':[0.5],'spectral_radius':[0.25],'beta':[0.01],'memscale':[5.],'delay':[1.],'fmax':[0.15]}]

N=25 # Each simulation is about 8 mins with a memory distribution, 1 min with plain memory

freq_errors = np.zeros((len(param_dicts),N,Ndf))
timing = np.zeros((len(param_dicts),N,Ndf))
train_errors = np.zeros((len(param_dicts),N,Ndf))
pred_errors = np.zeros((len(param_dicts),N,Ndf))
hits = np.zeros((len(param_dicts),N,Ndf,7))

for k, param_dict in enumerate(param_dicts):    
    train_errors[k], pred_errors[k], freq_errors[k], timing[k], hits[k] = repeat_runs(N,param_dict,seed_zero=0)
    
#%% POST-GRIDSEARCH: Compare performance between dist and single tau's
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

# Get the colors 
my_dpi = 300
prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']
color_single_train=colors[0]
color_dist_train=colors[1]
color_teacher=colors[2]
lw_test = 1.0
lw_signal = 0.75

fig, ax = plt.subplots(figsize=(nature_single,nature_single))

# Statistical analysis
# First we do a masking with respect to the fixed-tau results
f_mask = np.ma.masked_invalid(freq_errors[0]).mask + np.ma.masked_invalid(freq_errors[1]).mask
ma_f_err = np.ma.empty_like(freq_errors)
for k in range(0,2):
    ma_f_err[k]=np.ma.array(freq_errors[k],mask=f_mask)
means = np.mean(freq_errors,axis=1)
stds = np.std(freq_errors,axis=1)
means = np.mean(ma_f_err,axis=1)
stds = np.std(ma_f_err,axis=1)

# Plot the data
ax.plot(ma_f_err[1].compressed(),'o',label=r'single-$\tau$',color=colors[1])
ax.plot(ma_f_err[0].compressed(),'^',label=r'many-$\tau$',color=colors[0])


# Plot means as lines
base = np.ones_like(ma_f_err[0].compressed()) # compressed excludes the invalid values
for k in range(0,2) :
    ax.plot(means[k]*base,'--',color=colors[k])

# Draw rectangles showing the std
from matplotlib import patches

RMS_success = 0.03125 # 4 times df=0.0078125
RMS_success = 0.0390625 # 5 times df=0.0078125
# Create a Rectangle patches
l_rects = []
for k in range(0,2) :
    #l_rects.append(patches.Rectangle((-0.5, means[k]-stds[k]),len(base)+0.5, 2*stds[k], linewidth=1, edgecolor='none', facecolor=colors[k],alpha=0.2))
    l_rects.append(patches.Rectangle((-0.5, 0),len(base)+1.5, RMS_success, linewidth=1, edgecolor='none', facecolor='gray',alpha=0.2))
    ax.add_patch(l_rects[-1])

ax.text(-3.2,0.084,'(a)')
ax.set_xlim(-0.5,23.5)
ax.set_ylim(0,0.087)
ax.set_xlabel('run #')
ax.set_ylabel('Frequency generation RMS error')
ax.legend()
plt.tight_layout()
plt.savefig('mean_std.png',bbox_inches='tight')
plt.show()
   
#%% Plot a histogram with the hits
# Mask the hits as well with the f_mask calculated above
summed_hits = np.sum(hits[:,~f_mask],axis=1)

fig, ax = plt.subplots(figsize=(nature_single,nature_single))
ymax = summed_hits.max()+10

l_rects = []
for k in range(0,2) :
    #l_rects.append(patches.Rectangle((-0.5, means[k]-stds[k]),len(base)+0.5, 2*stds[k], linewidth=1, edgecolor='none', facecolor=colors[k],alpha=0.2))
    l_rects.append(patches.Rectangle((-1., 0),6.5, ymax, linewidth=1, edgecolor='none', facecolor='gray',alpha=0.2))
    ax.add_patch(l_rects[-1])

tick_list=['0',r'$\Delta f$',r'$2\Delta f$',r'$3\Delta f$',r'$4\Delta f$',r'$5\Delta f$',r'$+6\Delta f$']
ax.bar(np.arange(0,hits.shape[-1]),summed_hits[1],align='edge',width=-0.3,label=r'single-$\tau$',tick_label=tick_list,color=colors[1])
ax.bar(np.arange(0,hits.shape[-1]),summed_hits[0],align='edge',width=0.3,label=r'many-$\tau$',tick_label=tick_list,capstyle='round',color=colors[0])


ax.set_xlabel('Discrete frequency error')
ax.set_ylabel('Occurrence')
ax.text(-1.2,ymax-5,'(b)')
ax.set_xlim(-0.5,6.5)
ax.set_ylim(0,ymax)
ax.legend()
plt.tight_layout()
plt.savefig('histtest.svg')
plt.savefig('hist_errors.png',bbox_inches='tight')
plt.show()
#%% Plot mean of the frequency error (with standard deviation)
# Import the handler
from matplotlib.legend_handler import HandlerErrorbar

fig, ax = plt.subplots(figsize=(nature_single,nature_single))

l1 = ax.errorbar([0],freq_errors[0].mean(),freq_errors[0].std(),fmt='^',label='fixed memory',elinewidth=1.0,capsize=2.)
l2 = ax.errorbar([0],freq_errors[1].mean(),freq_errors[1].std(),fmt='o',label='dist. memory',elinewidth=1.0,capsize=2.)
#ax.set_xticklabels(input_scale_vals)
#plt.colorbar(im,ax=ax)
ax.set_xlabel('')
ax.set_xticks([0])
ax.set_xticklabels(['Network tests'])
ax.set_xlim(-0.5,0.5)
ax.set_ylabel('Frequency prediction RMS error')
ax.legend(loc='upper left',
          fontsize=8,
          frameon=False,
          labelspacing=1.75,
          handler_map={type(l1): HandlerErrorbar(xerr_size=1.0)})
plt.tight_layout()
plt.show()
   