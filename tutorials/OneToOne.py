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

# %% [markdown]
# # Leaky integrate and fire neuron by 3 neural nodes
# In this example we show how a single node performs temperol summation, a key
# feature in real neuron. Within a certian period in time, input signals are 
# summed and if the resulting potential reaches above a threshold value, 
# an output signal is generated.

# %%
# %matplotlib inline
# %load_ext autoreload
# %autoreload 2

import matplotlib.pyplot as plt
import time
import numpy as np

# load the modules specific to this project
from context import dynamicnanobrain
import dynamicnanobrain.core.networker as nw
import dynamicnanobrain.core.timemarching as tm
import dynamicnanobrain.core.plotter as plotter
import dynamicnanobrain.core.physics as physics
import dynamicnanobrain.core.logger as logger

plt.rcParams['figure.dpi'] = 100 # 200 e.g. is really fine, but slower

# %% [markdown]
# ### 1. Define the layers
# Define the layers of nodes in terms of how they are connected to the channels. Layers and weights are organized in dictionaries. The input and output layers do not need to be changed, but for the hidden layer we need to specify the number of nodes N and assign the correct channels to the input/output of the node.

# %%
# Create layers ordered from 0 to P organized in a dictionary
layers = {} 
# An input layer automatically creates on node for each channel that we define
layers[0] = nw.InputLayer(N=1)
# Forward signal layer
layers[1] = nw.HiddenLayer(N=1, output_channel='blue',excitation_channel='blue',inhibition_channel='red')
# Inhibiting memory layer
#layers[2] = nw.HiddenLayer(N=1, output_channel='red' ,excitation_channel='blue',inhibition_channel='red')
# Self-enforcing hidden layer
#layers[3] = nw.HiddenLayer(N=1, output_channel='blue',excitation_channel='blue',inhibition_channel='red')
layers[2] = nw.OutputLayer(N=1) # similar to input layer

# %% [markdown]
# ### 3. Define existing connections between layers
# The weights are set in two steps. 
# First the connetions between layers are defined. This should be done using the keys defined for each layer above, i.e. 0, 1, 2 ... for input, hidden and output layers, respectively. The `connect_layers` function returns a weight matrix object that we store under a chosen key, for example `'inp->hid'`.
# Second, the specific connections on the node-to-node level are specified using the node index in each layer

# %%
# Define the overall connectivity
weights = {}
# The syntax is connect_layers(from_layer, to_layer, layers, channels)
weights['inp->hd0'] = nw.connect_layers(0, 1, layers, channel='blue')
#weights['hd0->hd1'] = nw.connect_layers(1, 2, layers, channel='blue')
weights['hd0->out'] = nw.connect_layers(1, 2, layers, channel='blue')
# Backwards connection from the memory
#weights['hd1->hd0'] = nw.connect_layers(2, 1, layers, channel='red')
# Loop connection with the third hidden layer
#weights['hd0->hd2'] = nw.connect_layers(1, 3, layers, channel='blue')
# Backwards connection from third layer
#weights['hd2->hd0'] = nw.connect_layers(3, 1, layers, channel='blue')

# Define the specific node-to-node connections in the weight matrices
self_inhib = 1.0
self_excite = 1.0
# The syntax is connect_nodes(from_node, to_node, channel=label, weight=value in weight matrix)
# Input to first ring layer node
weights['inp->hd0'].connect_nodes(0, 0, weight=1.0) # channels['blue']=1
#weights['inp->hd0'].connect_nodes(channels['red'] ,0, channel='red', weight=1.0) # channels['blue']=1
# Hidden layer connections
#weights['hd0->hd1'].connect_nodes(0 ,0 , weight=self_inhib) 
# Loop connections
#weights['hd0->hd2'].connect_nodes(0 ,0 , weight=self_excite)
#weights['hd2->hd0'].connect_nodes(0 ,0 , weight=self_excite)
# Add damping connection
#weights['hd1->hd0'].connect_nodes(0 ,0 , weight=self_inhib)    
# Connect to output
weights['hd0->out'].connect_nodes(0, 0, weight=0.9)

# %% [markdown]
# ### 4. Visualize the network 
# The `plotter` module supplies functions to visualize the network structure. The nodes are named by the layer type (Input, Hidden or Output) and the index. To supress the printing of weight values on each connection, please supply `show_edge_labels=False`.
#
# #### Available layouts:
# **multipartite**: Standard neural network appearance. Hard to see recurrent couplings within layers.  
# **circular**: Nodes drawn as a circle  
# **shell**: Layers drawn as concetric circles  
# **kamada_kawai**: Optimization to minimize weighted internode distance in graph  
# **spring**: Spring layout which is standard in `networkx` 
#
# #### Shell layout
# This is my current favorite. It is configured to plot the input and output nodes on the outside of the hidden layer circle, in a combined outer concentric circle.
# %%
plotter.visualize_network(layers, weights, layout='shell', show_edge_labels=False,savefig=True)

# %% [markdown]
# ### 5. Specify the physics of the nodes
# Before running any simulations, we need to specify the input currents and the physics of the hidden layer nodes. Parameters can either be specified directly or coupled from the `physics` module. 

# %%
"""
# Specify different types of devices for the hidden layers
PtGate = physics.Device('../parameters/device_parameters_PtGate.txt')
AuGate = physics.Device('../parameters/device_parameters.txt')
# Tune the Rstore of the main node
PtGate.set_parameter('Rstore',5e6)
print('Rstore for PtGate device:')
PtGate.print_parameter('Rstore')
print('Rstore for AuGate device:')
AuGate.print_parameter('Rstore')
# 2. Memory (modify the parameters)
memory = physics.Device('../parameters/device_parameters_PtGate.txt')
memory.set_parameter('Rstore',2e7)
print('Rstore for memory device:')
memory.print_parameter('Rstore')

# Plot the two different transistors
plotter.visualize_transistor([AuGate.transistorIV_example(),PtGate.transistorIV_example()],labels=['AuGate-','PtGate-'])
"""

# %%
ns_device = physics.Device('../parameters/device_parameters_1ns_revised.txt','../parameters/LED_iv_s1e4.txt')
# Scale the memory constant 100-fold
def scalememory(dev,memscale) :
    R_ref = dev.p_dict['Rstore']
    C_ref = dev.p_dict['Cstore']
    dev.set_parameter('Rstore',R_ref*(np.sqrt(memscale)))
    dev.set_parameter('Cstore',C_ref*(np.sqrt(memscale)))
    return dev

ns_device = scalememory(ns_device,1)
# Specify the internal dynamics of each layer by assigning a device
layers[1].assign_device(ns_device)

# Memory layer Vthres
print('Main node Vthres=',layers[1].Vthres)

# Calculate the unity_coeff to scale the weights accordingly
unity_coeff, Imax = ns_device.inverse_gain_coefficient(ns_device.eta_ABC, layers[1].Vthres)
print(f'Unity coupling coefficient calculated as unity_coeff={unity_coeff:.4f}')
print(f'Imax is found to be {Imax} nA')

# %%
# Specify an exciting arbitrary pulse train mixing 0.5 and 1 ns pulses
t0 = 2.0
tau = 5.0
t_blue = [(t0,t0+tau)] # 
#t_blue = [(5.0,6.0), (8.0,8.5), (9.0,9,5), (10.0,11.0)] # 

# Use the square pulse function and specify which node in the input layer gets which pulse
layers[0].set_input_vector_func(func_handle=physics.square_pulse, func_args=(t_blue, Imax))
# Use the costant function to specify the inhibition from I0 to H0
#layers[0].set_input_func(channel='red', func_handle=physics.constant, func_args=(I_red,))

# %% [markdown]
# ### 6. Evolve in time

# %%
# Start time t, end time T
t = 0.0
T = 20.0 # ns
# To sample result over a fixed time-step, use savetime
savestep = 0.1
savetime = savestep
# These parameters are used to determine an appropriate time step each update
dtmax = 0.1 # ns 
dVmax = 0.005 # V

nw.reset(layers)
# Create a log over the dynamic data
time_log = logger.Logger(layers) # might need some flags

start = time.time()

while t < T:
    # evolve by calculating derivatives, provides dt
    dt = tm.evolve(t, layers, dVmax, dtmax )

    # update with explicit Euler using dt
    # supplying the unity_coeff here to scale the weights
    tm.update(dt, t, layers, weights, unity_coeff)
    
    t += dt
    # Log the progress
    if t > savetime :
        # Put log update here to have (more or less) fixed sample rate
        # Now this is only to check progress
        print(f'Time at t={t} ns') 
        savetime += savestep
    
    time_log.add_tstep(t, layers, unity_coeff)

end = time.time()
print('Time used:',end-start)

# This is a large pandas data frame of all system variables
result = time_log.get_timelog()

#%% Integrate and compare input and output power

import numpy as np

Win = np.trapz(result['H0-Iexc'],result['Time'])
Wout = np.trapz(result['H0-Pout'],result['Time'])

print(f'Win={Win}')
print(f'Wout={Wout}')
print(f'fraction={Wout/Win}')

# %% [markdown]
# ### 7. Visualize results
# Plot results specific to certain nodes

# %%
nodes = ['H0']
plotter.plot_nodes(result, nodes, onecolumn=True)

# %% [markdown]
# For this system it's quite elegant to use the `plot_chainlist` function, taking as arguments a graph object, the source node (I1 for blue) and a target node (O1 for blue)

# %%
# Variable G contains a graph object descibing the network
G = plotter.retrieve_G(layers, weights)
#plotter.plot_chainlist(result,G,'I1','L0')
plotter.plot_chainlist(result,G,'I0','O0')
#plotter.plot_chainlist(result,G,'I0','L0')

# %% [markdown]
# Plot specific attributes

# %%
attr_list = ['Vled']
plotter.plot_attributes(result, attr_list)

# %% [markdown]
# We can be totally specific if we want. First we list the available columns to choose from

# %%
print(result.columns)




