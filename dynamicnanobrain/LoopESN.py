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
# modules specific to this project
#import context
from dynamicnanobrain.core import physics
from dynamicnanobrain.core import plotter
from dynamicnanobrain.esn import esn

Nreservoir = 40
SEED=47
# Get me a network
my_esn = esn.EchoStateNetwork(Nreservoir,seed=SEED,sparsity=10)

# %%

my_esn.show_network(savefig=True, arrow_size=5,font_scaling=2)
# %%
# Specify a standard device for the hidden layers
propagator = physics.Device('../parameters/device_parameters.txt')
my_esn.assign_device(propagator)


# %% [markdown] Setup the input/output for training
# ### Input and outputs to train the network
# Both input and output need to be supplied to the network in order to train it. 
# These are generated as a random sequence of frequecies.
import numpy as np
rng = np.random.RandomState(SEED)

def freqeuncy_step_generator(tend,fmin,fmax,dT,res=10) :
    # determine the size of the sequence
    dt = fmax**-1/res
    N = int(tend/dt) # steps of total time interval
    dN = int(dT/dt) # steps of average period
    # From the info above we can setup our intervals
    n_changepoints = int(N/dN)
    changepoints = np.insert(np.sort(rng.randint(0,N,n_changepoints)),[0,n_changepoints],[0,N])
    # From here on we use the pyESN example code, with some modifications
    const_intervals = list(zip(changepoints,np.roll(changepoints,-1)))[:-1]
    frequency_control = np.zeros((N,1))
    for k, (t0,t1) in enumerate(const_intervals): # enumerate here
        frequency_control[t0:t1] = fmin + (fmax-fmin)* (k % 2)
    # run time update through a sine, while changing the freqeuncy
    frequency_output = np.zeros((N,1))
    z = 0
    for i in range(N):
        z = z + 2*np.pi*frequency_control[i]*dt
        frequency_output[i] = (np.sin(z) + 1)/2
        
    tseries = np.arange(0,tend,step=dt)
    
    return np.hstack([np.ones((N,1)),frequency_control]),frequency_output,tseries
    
T = 2000 # ns
dT = 50 # ns, average period length of constant frequency
fmin = 1/10 # GHz
fmax = 1/5 # GHz

frequency_input, frequency_output, tseries = freqeuncy_step_generator(T,fmin,fmax,dT) 

print(f'Generated a time series from 0 to {T} ns with {len(tseries)} elements')

# Now we use interpolation to get function handles from these data
from scipy.interpolate import interp1d 
# Everything is scaled by Imax
def teacher_signal(signal_scale) :
    handle = interp1d(tseries,frequency_output*signal_scale,axis=0)
    return handle 

def input_signal(signal_scale) :
    handle = interp1d(tseries,frequency_input[:,1]*signal_scale,axis=0)  
    return handle

def bias_signal(signal_scale) :
    return lambda t : signal_scale

# %% 
# Plot the frequency control and periods together

if True :
    Nmax = 2999
    fig, (ax1, ax2) = plt.subplots(1,2)
    
    ax1.plot(tseries[:Nmax],frequency_input[:Nmax,1])
    ax2.plot(tseries[:Nmax],frequency_output[:Nmax])
    #ax3.plot(frequency_output[:1000])
    #ax2.plot(periods[:1000])
    
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
                
                
# %% Look at specific solutions

# Reiterate these constants
teacher_scaling=1.0
beta = 1e2 # regularization

# Training paraemters
Tfit = 600. # spend two thirds on training
scl = 2.0

# train and test a network
my_esn.specify_network(0.6,
                       2.5,
                       0.0,
                       teacher_scaling)

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
scl = 2.0
#my_esn.set_delay(0.5) # units of ns
# %%
tseries_test, pred_test, movie_series, plot_series = my_esn.predict(Tfit,scl*Tfit,output_all=True)
# Generate the target signal
teacher_handle = teacher_signal(my_esn.Imax*teacher_scaling)
teacher_test = teacher_handle(tseries_test)
pred_error = np.sqrt(np.mean((pred_test - teacher_test)**2))/my_esn.Imax
                
# %%
teacher_handle = teacher_signal(my_esn.Imax*teacher_scaling)
teacher_test = teacher_handle(tseries_test)

fig, ax = plt.subplots()

ax.plot(tseries_train[:],pred_train[:])
ax.plot(tseries_train,teacher_train,'--')
ax.plot(tseries_test[:],pred_test[:])
ax.plot(tseries_test,teacher_test,'--')


plt.show()

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