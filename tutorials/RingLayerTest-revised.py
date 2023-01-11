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
import time

# load the modules specific to this project
from context import dynamicnanobrain
import dynamicnanobrain.core.networker as nw
import dynamicnanobrain.core.timemarching as tm
import dynamicnanobrain.core.plotter as plotter
import dynamicnanobrain.core.physics as physics
import dynamicnanobrain.core.logger as logger

plt.rcParams['figure.dpi'] = 100 # 200 e.g. is really fine, but slower

# %% [markdown]
# ### 1. Define the broadcasting channels of the network
# This is done by creating a list of the channel names. The names are arbitrary and can be set by the user, such as 'postive', 'negative' or explicit wavelenghts like '870 nm', '700 nm'. Here I chose the colors 'red' and 'blue'.

# %%
channel_list = ['red', 'blue']
# Automatically generate the object that handles them
channels = {channel_list[v] : v for v in range(len(channel_list))}

# %% [markdown]
# ### 2. Define the layers
# Define the layers of nodes in terms of how they are connected to the channels. Layers and weights are organized in dictionaries. The input and output layers do not need to be changed, but for the hidden layer we need to specify the number of nodes N and assign the correct channels to the input/output of the node.

# %%
# Create layers ordered from 0 to P organized in a dictionary
layers = {} 
Nring=5
# An input layer for the incoming signal
layers[0] = nw.InputLayer(N=1)
layers[1] = nw.HiddenLayer(N=Nring, output_channel='blue',excitation_channel='blue',inhibition_channel='red')
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
weights['hd0->out'] = nw.connect_layers(1, 2, layers, channel='blue')
# Recurrent layer for the ring
weights['hd0->hd0'] = nw.connect_layers(1, 1, layers, channel='blue')

# Define the specific node-to-node connections in the weight matrices
ring_weight = 1.1 
# The syntax is connect_nodes(from_node, to_node, channel=label, weight=value in weight matrix)

# Draw a ring network with Nring nodes (Nring defined above)

# Input to first ring layer node
weights['inp->hd0'].connect_nodes(0, 0, weight=1.0) # channels['blue']=1

for k in range(0,Nring) :
    # Intra-ring connections 
    if k<Nring-1 :
        weights['hd0->hd0'].connect_nodes(k ,k+1, weight=ring_weight) 
    # Add damping connection
    #weights['inp->hd0'].connect_nodes(channels['red'] ,k, channel='red', weight=1.0)
    
# Connect to output
weights['hd0->out'].connect_nodes(Nring-1, 0, weight=0.9)


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
plotter.visualize_network(layers, weights, layout='shell', show_edge_labels=False)

# %% [markdown]
# ### 5. Specify the physics of the nodes
# Before running any simulations, we need to specify the input currents and the physics of the hidden layer nodes. Parameters can either be specified directly or coupled from the `physics` module. 

# %%
# Specify devices for the hidden layer
propagator = physics.Device('../parameters/device_parameters_1ns_revisedtest.txt')
layers[1].assign_device(propagator)

# Calculate the unity_coeff to scale the weights accordingly
unity_coeff, Imax = propagator.inverse_gain_coefficient(propagator.eta_ABC, layers[1].Vthres)
print(f'Unity coupling coefficient calculated as unity_coeff={unity_coeff:.4f}')
print(f'Imax is found to be {Imax} nA')

# %%
# Specify an exciting current square pulse and an inhibition square pulse
t_blue = [(3.0,6.0)] # 

# Use the square pulse function and specify which node in the input layer gets which pulse
layers[0].set_input_vector_func(func_handle=physics.square_pulse, func_args=(t_blue, 2*Imax))
#layers[0].set_input_func(channel='red', func_handle=physics.constant, func_args=I_red)

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

# %% [markdown]
# ### 7. Visualize results
# Plot results specific to certain nodes

# %%
nodes = ['H0','H1','H2','H3','H4']

#nodes = ['H0','H1','H2']
plotter.plot_nodes(result, nodes)

# %% [markdown]
# For this system it's quite elegant to use the `plot_chainlist` function, taking as arguments a graph object, the source node (I1 for blue) and a target node (O1 for blue)

# %%
# Variable G contains a graph object descibing the network
G = plotter.retrieve_G(layers, weights)
plotter.plot_chainlist(result,G,'I0','O0')

# %% [markdown]
# Plot specific attributes

# %%
attr_list = ['Vgate','Vexc']
plotter.plot_attributes(result, attr_list)

# %% [markdown]
# We can be totally specific if we want. First we list the available columns to choose from

# %%
print(result.columns)

# %%
plotter.visualize_dynamic_result(result, ['H0-ISD','H1-ISD','H2-ISD'])

# %%
