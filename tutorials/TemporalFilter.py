# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.13.8
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
# modules specific to this project
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
# Create layers ordered from 0 organized in a dictionary. Arbitrary labels can be used, i.e. 'TB1','CPU4' etc.
layers = {} 
# An input layer automatically creates on node for each channel that we define
layers[0] = nw.InputLayer(N=1)
# Forward signal layer
layers[1] = nw.HiddenLayer(N=1, output_channel='blue',excitation_channel='blue',inhibition_channel='red')
# Inhibiting memory layer
layers[2] = nw.HiddenLayer(N=1, output_channel='red' ,excitation_channel='blue',inhibition_channel='red')
layers[3] = nw.OutputLayer(N=1) # similar to input layer

# %% [markdown]
# ### 2. Define abstract connections between layers
# First, we define which layers are connected to which. The explicit weights
# are set in the next step.
# The weights are set in two steps. 
# First the connetions between layers are defined. This should be done using the keys defined for each layer above, i.e. 0, 1, 2 ... for input, hidden and output layers, respectively. The `connect_layers` function returns a weight matrix object that we store under a chosen key, for example `'inp->hid'`.
# Second, the specific connections on the node-to-node level are specified using the node index in each layer

# %%
# Define the overall connectivity
weights = {}
# The syntax is connect_layers(from_layer, to_layer, layers, channel)
weights['inp->hd0'] = nw.connect_layers(0, 1, layers, channel='blue')
weights['hd0->hd1'] = nw.connect_layers(1, 2, layers, channel='blue')
weights['hd0->out'] = nw.connect_layers(1, 3, layers, channel='blue')
# Backwards connection from the memory
weights['hd1->hd0'] = nw.connect_layers(2, 1, layers, channel='red')

# %% [markdown]
# ### 3. Define the specific connections
# Here, the specific connections on the node-to-node level are specified using the node index in each layer.
# Weights can also be set using a matrix (a bit boring here but possible)

# %%
# Define the specific node-to-node connections in the weight matrices
# -----------------------------------------
# Input to first ring layer node
weights['inp->hd0'].connect_nodes(0 ,0,weight=1.0) # channels['blue']=1
#weights['inp->hd0'].connect_nodes(channels['red'] ,0, channel='red', weight=1.0) # channels['blue']=1
# Hidden layer connections
weights['hd0->hd1'].connect_nodes(0 ,0 , weight=1.0) 
# Add damping connection
weights['hd1->hd0'].connect_nodes(0 ,0 , weight=1.0)    
# Connect to output, we can also use matrices
import numpy as np
weight_mat = np.array([[0.9]])
weights['hd0->out'].set_W(weight_mat)


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
plotter.visualize_network(layers, weights, layout='shell',
                          show_edge_labels=False,#shell_order=[1,[2,3],[0,4]],
                          savefig=True)

# %% [markdown]
# ### 5. Specify the physics of the nodes
# Before running any simulations, we need to specify the input currents and the physics of the hidden layer nodes. Parameters can either be specified directly or coupled from the `physics` module. 

# %%
# Specify two types of devices for the hidden layer
# 1. Propagator (standard parameters)
propagator = physics.Device('../parameters/device_parameters_1ns_revisedtest.txt')
propagator.set_parameter('Rstore',3e6) # Ohms, original value is 2e6 Ohms
propagator.print_parameter('Rstore')
# 2. Memory (modify the parameters)
memory = physics.Device('../parameters/device_parameters_1ns_revisedtest.txt')
memory.set_parameter('Rstore',1e7) # Ohms, original value is 2e6 Ohms
memory.print_parameter('Rstore')

# Specify the internal dynamics by supplying the RC constants to the hidden layer (six parameters)
layers[1].assign_device(propagator)
layers[2].assign_device(memory)

# %%

# Tweak the threshold voltage
Vthres=1.0
#layers[1].Vthres=Vthres
#layers[2].Vthres=Vthres

# Calculate the unity_coeff to scale the weights accordingly
unity_coeff, Imax = propagator.inverse_gain_coefficient(propagator.eta_ABC, Vthres)
print(f'Unity coupling coefficient calculated as unity_coeff={unity_coeff:.4f}')
print(f'Imax is found to be {Imax} nA')

# %%
# Specify an exciting current pulse train of squares
# Pulse train of 2 ns pulses
t_blue = [(6.0,6.5),(10.0,11.5),(12.0,13.5),(14.0,15.5)]#, (255.0,455.0)] # 

# Use the square pulse function and specify which node in the input layer gets which pulse
layers[0].set_input_vector_func(func_handle=physics.square_pulse, func_args=(t_blue, 0.4*Imax))

# %% [markdown]
# ### 6. Evolve in time

# %%
# Start time t, end time T
t = 0.0
T = 20.0 # ns
# To sample result over a fixed time-step, use savetime
savestep = 1.0
savetime = savestep
# These parameters are used to determine an appropriate time step each update
dtmax = 0.1 # ns 
dVmax = 0.001 # V

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
#nodes = ['H0','H1','H2','H3','H4']

nodes = ['H0','K0']
plotter.plot_nodes(result, nodes, onecolumn=True)

# %% [markdown]
# For this system it's quite elegant to use the `plot_chainlist` function, taking as arguments a graph object, the source node (I1 for blue) and a target node (O1 for blue)

# %%
# Variable G contains a graph object descibing the network
G = plotter.retrieve_G(layers, weights)
plotter.plot_chainlist(result,G,'I0','K0')


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
plotter.visualize_dynamic_result(result, ['H0-Iout','H0-Pout'])

# %% [markdown]
# We can also look in depth at the physical aspects of the node, like transistor IV and the LED efficiency.

# %%
plotter.visualize_transistor(propagator.transistorIV_example())

# %%
plotter.visualize_LED_efficiency(propagator.eta_example(propagator.eta_ABC))

# %%
tau_memory = memory.calc_tau_gate() # 0.056 ns
tau_propagator = propagator.calc_tau_gate()
gammas_memory = memory.calc_gammas()
gammas_propagator = propagator.calc_gammas()

# %% [markdown]
# Produce Bode plots to determine what our frequency cutoff will be

# %%
# Setup the gain function
eigvals = propagator.setup_gain(propagator.gammas)

# The eigenvalues
print('System eigenvalues:')
k=0
for l in eigvals :
    print(f'eig[{k}]={l:.2f} ns^-1 ')
    k+=1
# Regarding the eigenvalues, eig[0]=-25.81 ns^-1, eig[1]=-5.00 ns^-1, eig[2]=-0.19 ns^-1
# eig[1] regards the charge collecting subsystem, this is their RC constant
# eig[0] regards the exchange between the collectors and the storage, I believe. For A33=20 it's zero
# eig[2] regards the time scale of the storage unit, the longest timescale in the system.
    
# PUT THIS IN THE PLOTTER MODULE
import numpy as np
# Visualize the response function 
Ns = 100
# Units are already in GHz so this creates a space from 1 MHz to 10 GHz
s_exp = np.linspace(-5,1,Ns)
s = 1j*10**s_exp

# Sample the gain function
G11, _ = propagator.gain(s,eigvals,propagator.gammas)

mag_G11 = np.absolute(G11) / np.absolute(G11[0])
arg_G11 = np.angle(G11)

mag_G11_dB = 20*np.log10(mag_G11)

# %%
# Produce Bode plots of the results
# Define parameters
my_dpi = 300
prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']

# Figure sizes
inchmm = 25.4
nature_single = 89.0 / 25.4
nature_double = 183.0 / 25.4
nature_full = 247.0 / 25.4

# Plot options
font = {'family' : 'sans',
        'weight' : 'normal',
        'size'   : 12}
plt.rc('font', **font)

fig, (ax1, ax2) = plt.subplots(2,1,figsize=(nature_double,nature_single),sharex=True)

f_min = abs(s[0])
plot_f_max = 3.0 # GHz

ax1.plot(abs(s),mag_G11_dB)
ax1.plot([f_min,plot_f_max],[-3,-3],'k--')
ax1.grid('True')
ax1.set_xscale('log')
ax1.set_title('Bode plots for dynamic node')
#ax1.set_xlabel('Frequency (GHz)')
ax1.set_ylabel('|G11|/|G11[0]| (dB)')
ax1.set_ylabel('|H| (dB)')
ax1.set_xlim(f_min,plot_f_max)
ax1.set_ylim(-20,2)

ax2.plot(abs(s),arg_G11*180/np.pi)
ax2.grid('True')
#ax2.set_xscale('log')
ax2.set_xlabel('Frequency (GHz)')
ax2.set_ylabel('Phase (radians)')
ax2.set_ylim(-180,10)

plt.show()

# %%
