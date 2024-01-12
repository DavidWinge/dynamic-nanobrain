#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  8 15:53:48 2022

@author: dwinge
"""
import time

import numpy as np
import pandas as pd
from ..core import networker as nw
from ..core import plotter
from ..core import logger
from ..core import timemarching as tm
from scipy.interpolate import interp1d

N_TB1=8
N_CL1=16
N_CPU4=16
N_Pontine=16
N_TN2=2
N_CPU1a=14
N_CPU1b=2


default_update_m = 2e-4
default_cpu1_input_m = 0.5
default_a = 0.1

# By Tom Stone (Stone et al Curr Biology 2017)
def gen_tb_tb_weights(weight=1.):
    """Weight matrix to map inhibitory connections from TB1 to other neurons"""
    W = np.zeros([N_TB1, N_TB1])
    sinusoid = -(np.cos(np.linspace(0, 2*np.pi, N_TB1, endpoint=False)) - 1)/2
    for i in range(N_TB1):
        values = np.roll(sinusoid, i)
        W[i, :] = values
    return weight * W

class TravelLog :
    
    # This should store the velocity and heading at each instance. Might as well
    # store also the position
    def __init__(self) :
        self.list_data = []
        self.column_labels = self.columnnames()
        
    def columnnames(self) :
        names=['Time','x','y','vx','vy','heading']
        return names
        
    def add_tstep(self,t,x,v,heading,motor) :
        row = [t]
        # position
        pos = x.flatten(order='F').tolist()
        row += pos
        # velocity
        vel = v.flatten(order='F').tolist()
        row += vel
        row += [heading] # skipped motor here, plus needs to be a list to be appended
        self.list_data.append(row)
        
    def get_timelog(self) :
        # Convert to pandas data frame
        return pd.DataFrame(self.list_data, columns=self.column_labels)
        
class StoneNetwork :
    
    def __init__(self,name='StoneNetwork', tb1_c=0.33, 
                 mem_update_h=0.0025,cpu4_cpu1_m=default_cpu1_input_m,
                 pon_cpu1_m=default_cpu1_input_m,
                 tb1_cpu1_m=1.0, weight_noise=0.0, Vt_noise=0.0) :
        self.name = name
        self.tb1_c=tb1_c
        self.mem_update_h = mem_update_h
        self.cpu4_cpu1_m = cpu4_cpu1_m
        self.pon_cpu1_m = pon_cpu1_m
        self.tb1_cpu1_m = tb1_cpu1_m

        # Specify noise in weights before setting weights
        self.weight_noise = weight_noise
        if self.weight_noise > 0.0 :
            self.noisy_weights=True
        else :
            self.noisy_weights=False
        # Set connections and weights
        self.layers, self.weights = self.initialize_nw()
            
        # Initialize a dictionary for the devices
        self.devices = {}
        
    def noisify_Vt(self, noise) :
        # Loop through layers
        for key in self.layers :
            # Original Vt
            if self.layers[key].layer_type == 'hidden' :
                Vt0 = self.layers[key].device.p_dict['Vt']
                self.layers[key].Vt_vec = Vt0 + np.random.normal(scale=noise,size=self.layers[key].N)

    def noisify_weights(self, W, noise=0.01):
        """Takes a weight matrix and adds some noise on to non-zero values."""
        N = np.random.normal(scale=noise, size=W.shape)
        # Only noisify the connections (positive values in W). Not the zeros.
        N_nonzero = N * W
        return W + N_nonzero
    
    def randomize_memory(self,noise=0.1,dist='uniform',seed=None) :
        """Introduce a distribution of memory constants in the memory layer"""
        self.layers['CPU4'].multiA = True
        if dist=='uniform' :
            self.layers['CPU4'].generate_uniform_Adist(noise,seed)
        else :
            print('Unexpected dist in randomize_memory')
        
    def initialize_nw(self) :
        # Defining layers with custom labels        
        layers = {}
        # input layers
        layers['CL1'] = nw.InputLayer(N=N_CL1)
        layers['TN2'] = nw.InputLayer(N=N_TN2)
        layers['Bias'] = nw.InputLayer(N=1)
        # ring layers
        layers['TB1'] = nw.HiddenLayer(N=N_TB1, output_channel='green', inhibition_channel='green', excitation_channel='blue')
        # Add a rectifying layer that is not in the Stone model
        layers['Rectifier'] = nw.HiddenLayer(N=N_CPU4, output_channel='brown', inhibition_channel='green', excitation_channel='brown')
        layers['CPU4'] = nw.HiddenLayer(N=N_CPU4, output_channel='orange', inhibition_channel='green', excitation_channel='brown')
        layers['Pontine'] = nw.HiddenLayer(N=N_Pontine,output_channel='green',inhibition_channel='white',excitation_channel='orange')
        layers['CPU1a'] = nw.HiddenLayer(N=N_CPU1a, output_channel='black', inhibition_channel='green', excitation_channel='orange')
        layers['CPU1b'] = nw.HiddenLayer(N=N_CPU1b, output_channel='purple', inhibition_channel='green', excitation_channel='orange')
        
        weights = {}
        # Set up the connections
        weights['CL1->TB1']=nw.connect_layers('CL1', 'TB1', layers, channel='blue')
        W = np.tile(np.diag([1.0]*N_TB1),2)
        W *= (1.-self.tb1_c) # scaling factor to weigh TB1 and CL1 input to TB1
        if self.noisy_weights : W = self.noisify_weights(W,self.weight_noise)
        weights['CL1->TB1'].set_W(W)
        
        weights['TB1->TB1']=nw.connect_layers('TB1','TB1',layers,channel='green')
        W = gen_tb_tb_weights(weight=self.tb1_c)
        if self.noisy_weights : W = self.noisify_weights(W,self.weight_noise)
        weights['TB1->TB1'].set_W(W)
        
        weights['TB1->Rectifier']=nw.connect_layers('TB1', 'Rectifier', layers, channel='green')
        W = np.tile(np.diag([1.0]*N_TB1),(2,1)) 
        if self.noisy_weights : W = self.noisify_weights(W,self.weight_noise)
        weights['TB1->Rectifier'].set_W(W)
        # The memory is updated with a lower weight
       # weights['TB1->Rectifier'].scale_W(self.mem_update_h)
        
        weights['TB1->CPU1a']=nw.connect_layers('TB1', 'CPU1a', layers, channel='green')
        W = np.tile(np.diag([1.0]*N_TB1),(2,1))
        if self.noisy_weights : W = self.noisify_weights(W,self.weight_noise)
        weights['TB1->CPU1a'].set_W(W[1:-1]*self.tb1_cpu1_m)
        
        weights['TB1->CPU1b']=nw.connect_layers('TB1', 'CPU1b', layers, channel='green')
        W = np.zeros((2,N_TB1))
        W[0,-1] = 1.0
        W[1,0]  = 1.0
        if self.noisy_weights : W = self.noisify_weights(W,self.weight_noise)
        weights['TB1->CPU1b'].set_W(W*self.tb1_cpu1_m)
        
        weights['TN2->Rectifier']=nw.connect_layers('TN2', 'Rectifier', layers, channel='brown')
        W = np.concatenate((np.tile(np.array([1.,0.]),(N_CPU4//2,1)),np.tile(np.array([0.,1.]),(N_CPU4//2,1)))) 
        if self.noisy_weights : W = self.noisify_weights(W,self.weight_noise)
        weights['TN2->Rectifier'].set_W(W)
        # The memory is updated with a lower weight
        #weights['TN2->Rectifier'].scale_W(self.mem_update_h)

        # Diagonal weights connecting the rectifier to the CPU4 layer 
        weights['Rectifier->CPU4']=nw.connect_layers('Rectifier', 'CPU4', layers, channel='brown')
        W = np.diag([1.0]*N_CPU4)
        if self.noisy_weights : W = self.noisify_weights(W,self.weight_noise)
        weights['Rectifier->CPU4'].set_W(W)
        weights['Rectifier->CPU4'].scale_W(self.mem_update_h)
        
        # Diagonal weights connecting the rectifier to the CPU4 layer 
        weights['Bias->CPU4']=nw.connect_layers('Bias', 'CPU4', layers, channel='green')
        W = np.ones((N_CPU4,1))
        if self.noisy_weights : W = self.noisify_weights(W,self.weight_noise)
        weights['Bias->CPU4'].set_W(W)     
        weights['Bias->CPU4'].scale_W(self.mem_update_h)
        
        weights['CPU4->CPU1a']=nw.connect_layers('CPU4', 'CPU1a', layers, channel='orange')
        W =   np.array([[0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0.], #2
                        [0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0.],
                        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0.],
                        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0.],
                        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0.],
                        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0.],
                        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0.],
                        [0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                        [0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                        [0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                        [0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                        [0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                        [0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                        [0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0.], #15
                        ])
        if self.noisy_weights : W = self.noisify_weights(W,self.weight_noise)
        weights['CPU4->CPU1a'].set_W(W*self.cpu4_cpu1_m) # scale with 0.5 as in cx_rate.py
        
        weights['CPU4->CPU1b']=nw.connect_layers('CPU4', 'CPU1b', layers, channel='orange')
        W = np.zeros((2,N_CPU4))
        W[0,0]=1.0
        W[-1,-1]=1.0
        if self.noisy_weights : W = self.noisify_weights(W,self.weight_noise)
        weights['CPU4->CPU1b'].set_W(W*self.cpu4_cpu1_m) # scale with 0.5 as in cx_rate.py
        
        weights['CPU4->Pontine']=nw.connect_layers('CPU4', 'Pontine', layers, channel='orange')
        W = np.diag([1.0]*N_CPU4)
        if self.noisy_weights : W = self.noisify_weights(W,self.weight_noise)
        weights['CPU4->Pontine'].set_W(W)
        
        weights['Pontine->CPU1a']=nw.connect_layers('Pontine', 'CPU1a', layers, channel='green')
        W =   np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0], #2
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                        [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], #15])
                        ],dtype=float)
        if self.noisy_weights : W = self.noisify_weights(W,self.weight_noise)
        weights['Pontine->CPU1a'].set_W(W*self.pon_cpu1_m)
        
        weights['Pontine->CPU1b']=nw.connect_layers('Pontine', 'CPU1b', layers, channel='green')
        W = np.zeros((2,N_Pontine))
        W[0,4] = 1.0
        W[1,11]= 1.0
        if self.noisy_weights : W = self.noisify_weights(W,self.weight_noise)
        weights['Pontine->CPU1b'].set_W(W*self.pon_cpu1_m)
        
        return layers, weights
        
    def show_network(self, savefig=False,layout='spring',**kwargs) :
        pos = plotter.visualize_network(self.layers, self.weights, 
                                        #exclude_nodes={0:['I1','I2'],3:['O1','O2']},
                                        node_size=100,
                                        layout=layout, 
                                        show_edge_labels=False,
                                        savefig=savefig,
                                        **kwargs)
        
        return pos
        
    def show_weights(self, **kwargs) :
        del_bias = self.weights.copy()
        print(del_bias.keys())
        del del_bias['Bias->CPU4']
        fig, ax = plotter.plot_weights(del_bias, **kwargs)
        return fig, ax
        
    def assign_device(self, device_dict, unity_key) :
        for key in device_dict :
            self.layers[key].assign_device(device_dict[key])
            self.devices[key] = device_dict[key]
        unity_device = device_dict[unity_key]
        self.unity_coeff, self.Imax = unity_device.inverse_gain_coefficient(unity_device.eta_ABC, self.layers[unity_key].Vthres)

    def show_devices(self, Vleak_dict, device_dict=None, **kwargs) :
        if device_dict is None :
            # Send the class own device dict
            plotter.plot_devices(self.devices, Vleak_dict, **kwargs)
        else :
            plotter.plot_devices(device_dict, Vleak_dict, **kwargs)
    
    def assign_memory(self, key, device) :
        self.layers[key].assign_device(device)
        self.mem_coeff, self.mem_Imax = device.inverse_gain_coefficient(device.eta_ABC, self.layers[key].Vthres)
        return self.mem_coeff, self.mem_Imax
    
    def specify_inputs(self,key,input_handle,scaling=1.0,fixImax=None) :
        if fixImax is None :
            self.layers[key].set_input_vector_func(func_handle=input_handle(scaling*self.Imax))
        else :  
            self.layers[key].set_input_vector_func(func_handle=input_handle(scaling*fixImax))
        
    def updateheading(self,heading,layers,update_m=default_update_m,noise=0.0) :
        # sum left and right hand turning neurons
        r_right = sum(layers['CPU1a'].P[:7]) + layers['CPU1b'].P[1]
        r_left  = sum(layers['CPU1a'].P[7:]) + layers['CPU1b'].P[0]
        r_sum = (r_right-r_left) + np.random.normal(scale=noise*self.Imax)
        return update_m*r_sum*self.unity_coeff
        
    def evolve(self,T,reset=True,t0=0.,inbound=False,savestep=1.0,
               initial_pos=(0,0),initial_vel=(0,0),initial_heading=0,
               a=default_a, drag=0.15, hupdate=default_update_m,
               tn2scaling=0.9,noise=0.05,tb1scaling=0.9,
               mem_init_c=0.3, turn_noise=0.0, tau_noise=1.,
               printtime=False) : 
               
        # Start time is t
        t=t0
        # These parameters are used to determine an appropriate time step each update
        dtmax = 0.1 # ns 
        dVmax = 0.01 # V # I loosen this a little bit now
        # To sample result over a fixed time-step, use savetime
        savetime = max(savestep,dtmax) + t0
        
        if reset :
            # option to keep reservoir in its current state
            nw.reset(self.layers)
            
        if not inbound :
            # Initialize activity in the memory cells
            self.layers['CPU4'].V[:,:]=self.layers['CPU4'].Vthres*mem_init_c
            
        # Create a log over the dynamic data
        time_log = logger.Logger(self.layers,feedback=True) # might need some flags
        # write zero point
        time_log.add_tstep(t, self.layers, self.unity_coeff)
        
        if inbound :
            # Create a travel log for the return journey
            travel_log = TravelLog()
            v = np.array(initial_vel)
            x = np.array(initial_pos)
            heading = initial_heading
            travel_log.add_tstep(t, x, v, heading, 0.0) # motor=0 for duplcate first step
            
            # Allow the agent to determine its speed through TN2 layer
            def get_flow(heading, v, pref_angle=np.pi/4):
                head_arr = np.array([[np.sin(heading + pref_angle),
                                      np.cos(heading + pref_angle)],
                                     [np.sin(heading - pref_angle),
                                      np.cos(heading - pref_angle)]])
                return head_arr @ v

            def tn2_activity(t) :
                tn2 = get_flow(heading, v)
                # Add noise
                tn2 += np.random.normal(scale=noise, size=len(tn2))
                # scale by the standard current factor
                return np.clip(tn2,0,1)*self.Imax*tn2scaling
            
            # Specify the output function
            self.layers['TN2'].set_input_vector_func(tn2_activity)
            
            # Allow the agent to determine its heading through CL1 layer
            TL_angles = np.tile(np.arange(0,8),2)*np.pi/4
            
            from scipy.special import expit
                
            ### NOT SURE THAT THIS IS THE BEST PLACE TO PUT IT
            # From Stone Curr Biol 2017
            # TUNED PARAMETERS:
            tl2_slope_tuned = 6.8
            tl2_bias_tuned = 3.0   
            cl1_slope_tuned = 3.0
            cl1_bias_tuned = -0.5
            
            # TODO: Shorter if clip is no longer needed
            def noisy_sigmoid(v, slope=1.0, bias=0.5, noise=0.01):
                """Takes a vector v as input, puts through sigmoid and
                adds Gaussian noise. Results are clipped to return rate
                between 0 and 1"""
                sig = expit(v * slope - bias)
                if noise > 0:
                    sig += np.random.normal(scale=noise, size=len(v))
                return np.clip(sig, 0, 1)
            ###
            
            def tl2_activity(heading):
                output = np.cos(heading - TL_angles)
                return noisy_sigmoid(output, tl2_slope_tuned, tl2_bias_tuned, noise=0.)
            
            def cl1_activity(t):
                output = tl2_activity(heading)
                sig = noisy_sigmoid(-output, cl1_slope_tuned, cl1_bias_tuned, noise=0.)
                # scale by the standard current factor
                return sig*self.Imax*tb1scaling
            
            def rotate(theta, r):
                """Return new heading after a rotation around Z axis."""
                return (theta + r + np.pi) % (2.0 * np.pi) - np.pi

            # Specify the output function
            self.layers['CL1'].set_input_vector_func(cl1_activity)
        
        # Generate the noise arrays for the relevant times:
        #noisy_layers = ['CL1','TN2','TB1','CPU4'] version used for the N=100 results
        noisy_layers = ['CL1','TN2','CPU1a','CPU1b']
        noise_arrays = {}
        noise_funcs = {}
        times = np.arange(t0,T,step=tau_noise)
        for key in noisy_layers :
            noise_arrays[key] = np.random.normal(scale=noise,size=(len(times),self.layers[key].N))
            if self.layers[key].layer_type=='hidden' :
                noise_arrays[key] *= self.Imax/self.unity_coeff
            else :
                noise_arrays[key] *= self.Imax
            # Now we interpolate
            noise_funcs[key] = interp1d(times, noise_arrays[key], axis=0,kind='nearest')  
        
        start = time.time()
        
        while t < T:
            # evolve by calculating derivatives, provides dt
            dt = tm.evolve(t, self.layers, dVmax, dtmax )
        
            # update with explicit Euler using dt
            # supplying the unity_coeff here to scale the weights
            # TODO: Add noise if needed, send as a dictionary of arrays
            tm.update(dt, t, self.layers, self.weights, self.unity_coeff, t0, noise=noise_funcs)
            #tm.update(dt, t, self.layers, self.weights, self.unity_coeff, t0)
            
            if inbound :
                # update agent heading
                motor = self.updateheading(heading,self.layers,hupdate,turn_noise)
                # use a dt here since we do many turns per time unit
                heading = rotate(heading,motor*dt)
                # velocity using drag force and acceleration
                v[0] += np.sin(heading)*a*dt
                v[1] += np.cos(heading)*a*dt
                v -= v*drag*dt
                x += v*dt
            
            t += dt
            #print(t,' ns')
            # Log the progress
            if t > savetime :
                # Put log update here to have (more or less) fixed sample rate
                # Now this is only to check progress
                if printtime :
                    print(f'Time at t={t} ns') 
                savetime += savestep         
                time_log.add_tstep(t, self.layers, self.unity_coeff)
                
                if inbound :
                    travel_log.add_tstep(t,x,v,heading,motor)
        
        end = time.time()
        print(f'Travel steps: T={T}, time used:{end-start}')
        
        # This is a large pandas data frame of all system variables
        result = time_log.get_timelog()
        
        if inbound :
            travel = travel_log.get_timelog()
            return result, travel
        else :
            return result