#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 11 09:10:12 2021

@author: dwinge
"""
import numpy as np
import time
from ..core import networker as nw
from ..core import plotter
from ..core import logger
from ..core import timemarching as tm

class EchoStateNetwork :
    
    def __init__(self,N,input_handle=None,bias_handle=None,teacher_handle=None,
                 input_scaling=1.0,bias_scaling=1.0,teacher_scaling=1.0,
                 spectral_radius=1.0, sparsity=10, timescale=1.0,
                 device=None,silent=False,savefig=False,seed=None) :
        
        self.N = N
        self.input_handle = input_handle
        self.bias_handle = bias_handle
        self.teacher_handle = teacher_handle
        self.sparsity = sparsity
        self.timescale = timescale
        self.silent = False
        self.device = device
        self.savefig = savefig
        self.seed = seed
                
        # Setup the network
        self.specify_network(spectral_radius,input_scaling,bias_scaling,teacher_scaling)

        # If the handles are stored from the start, this can be given as an
        # internal update if layers are redefined.
        self.specify_inputs(input_handle,bias_handle,teacher_handle)

    def _unscale_teacher(self, teacher_scaled):
        """inverse operation of the _scale_teacher method."""
        return teacher_scaled / self.teacher_scaling
    
    def specify_network(self,spectral_radius,input_scaling=None,
                        bias_scaling=None,
                        teacher_scaling=None,**kwargs) :
        
        self.spectral_radius = spectral_radius
        
        if input_scaling is not None :
            self.input_scaling = input_scaling
        if teacher_scaling is not None :
            self.teacher_scaling = teacher_scaling   
        if bias_scaling is not None :
             self.bias_scaling = bias_scaling   

        # instead of a return statement we do this to be able to call it from
        # outside the class scope easily
        if self.N==1 :
            self.layers, self.weights =  self.initialize_single_node_nw(self.N,**kwargs)
        else :
            self.layers, self.weights = self.initialize_nw(self.N,self.sparsity,**kwargs)
        
        # Reassign the device
        if self.device is not None :
            self.assign_device(self.device)
            
    def initialize_single_node_nw(self,N=1,Ninput=2,Noutput=1,initialize_output=False,feedback=True,Wfb_scale=1.0) :
    
        layers = {} 
        # An input layer automatically creates on node for each channel that we define
        layers[0] = nw.InputLayer(Ninput) # One for bias, one for exciting input
        # Hidden layer, one excitatiory only
        layers[1] = nw.HiddenLayer(N, output_channel='blue',excitation_channel='blue',inhibition_channel='red')
        # Output layer
        layers[3] = nw.OutputLayer(Noutput) # Only readout exciting output
        
        weights = {}
        # The syntax is connect_layers(from_layer, to_layer, layers, channel)
        # Connections into the reservoir from input layer
        weights['inp->hd0'] = nw.connect_layers(0, 1, layers, 'blue')
        #rng = np.random.RandomState(self.seed)
        W_in = np.ones((N, Ninput)) 
        weights['inp->hd0'].set_W(W_in) # random weights
        
        return layers, weights
        
    def initialize_nw(self,N,sparsity,initialize_output=False,feedback=True,Wfb_scale=1.0,Noutput=1) :
        # We will use a standard set of channels for now    
        #channel_list = ['blue','red']
        # Automatically generate the object that handles them
        #channels = {channel_list[v] : v for v in range(len(channel_list))}
        
        layers = {} 
        Ninput=2
        # An input layer automatically creates on node for each channel that we define
        layers[0] = nw.InputLayer(Ninput) # One for bias, one for exciting input
        # Two intertwined hidden layers, one excitatiory, one inhibitory
        layers[1] = nw.HiddenLayer(N//2, output_channel='blue',excitation_channel='blue',inhibition_channel='red')
        layers[2] = nw.HiddenLayer(N//2, output_channel='red' ,excitation_channel='blue',inhibition_channel='red')
        # Output layer
        layers[3] = nw.OutputLayer(Noutput) # Only readout exciting output
        
        # Define the overall connectivity
        weights = {}
        # The syntax is connect_layers(from_layer, to_layer, layers, channel)
        # Connections into the reservoir from input layer
        weights['inp->hd0'] = nw.connect_layers(0, 1, layers, 'blue')
        weights['inp->hd1'] = nw.connect_layers(0, 2, layers, 'blue')
        # Connections between reservoir nodes
        weights['hd0->hd1'] = nw.connect_layers(1, 2, layers, 'blue')
        weights['hd1->hd0'] = nw.connect_layers(2, 1, layers, 'red')
        # Intralayer connections
        weights['hd0->hd0'] = nw.connect_layers(1, 1, layers, 'blue')
        weights['hd1->hd1'] = nw.connect_layers(2, 2, layers, 'red')
        if feedback :
            # Connections back into reservoir from output
            weights['out->hd0'] = nw.connect_layers(3, 1, layers, 'blue')
            weights['out->hd1'] = nw.connect_layers(3, 2, layers, 'blue')
            
        # Initiate the weights randomly
        rng = np.random.RandomState(self.seed)

        # Input weights to all of the input units
        W_in = rng.rand(N, Ninput) 
        weights['inp->hd0'].set_W(W_in[:N//2]) # first half
        weights['inp->hd1'].set_W(W_in[N//2:]) # second half
        
        W_res = np.zeros((N,N))
        # Generate a set of random connections for each node
        adj_sparsity = min(N,sparsity)
        for row in range(N) :
            selection = rng.choice(N,size=adj_sparsity)
            rnd_w = rng.rand(adj_sparsity) # all positive [0,1)
            W_res[row][selection] = rnd_w
        
        # Generate a large matrix of values for the whole reservoir 
        #W_res = rng.rand(N, N)  # all positive [0,1) # old generation
        # Rightmost half side of this matrix will effectively be negative weights (-- and -+)
        W_res[:,N//2:] *= -1 
        # Delete the fraction of connections given by sparsity:
        #W_res[rng.rand(*W_res.shape) < sparsity] = 0 # old sparsity process
        # Delete any remaining diagonal elements (we can't have those)
        for k in range(0,N) :
            W_res[k,k] = 0.
            
        radius = np.max(np.abs(np.linalg.eigvals(W_res)))
        # rescale them to reach the requested spectral radius:
        W_res = W_res * (self.spectral_radius / radius)
        # Now shift the signs again for the implementation
        W_res[:,N//2:] *= -1 

        # Now initiate reservoir weights
        W_partition = {'hd0->hd0':(0,N//2,0,N//2), # ++
                       'hd0->hd1':(N//2,N,0,N//2), # +- 
                       'hd1->hd1':(N//2,N,N//2,N), # --
                       'hd1->hd0':(0,N//2,N//2,N)} # -+
 
        for connection in W_partition :
            A,B,C,D = W_partition[connection]
            weights[connection].set_W(W_res[A:B,C:D])

        # Output weights from reservoir to the output units
        if initialize_output :
            W_out = rng.rand(Noutput,N)
            # Connections from reservoir to output
            weights['hd0->out'] = nw.connect_layers(1, 3, layers, 'blue') # Not sure here whether I had an inhibition coupling, I don't think so
            weights['hd0->out'].set_W(W_out[:,:N//2])
            #weights['hd1->out'].set_W(W_out[:,N//2:])

        if feedback :
            # Feedback weights from output nodes back into reservoir 
            # These are random. The weights from the extended network state are trained.
            # AT THIS POINT, WE USE ONLY POSITIVE SIGNAL AT OUTPUT
            W_feedb = rng.rand(N,Noutput)
            weights['out->hd0'].set_W(Wfb_scale*W_feedb[:N//2,:])
            weights['out->hd1'].set_W(Wfb_scale*W_feedb[N//2:,:])
        
        return layers, weights
        
    def add_trained_weights(self, W_out, channel) :
        # First we define the weights to add
        self.weights['hd0->out'] = nw.connect_layers(1, 3, self.layers, channel)
        # We don't add the inhibition layer for now
        #self.weights['hd1->out'] = nw.connect_layers(2, 3, self.layers, self.channels)
        self.weights['inp->out'] = nw.connect_layers(0, 3, self.layers, channel)
        
        W_res_out = np.zeros(self.weights['hd0->out'].ask_W(silent=True)) # ask_W returns shape
        # Set the weights for the blue output
        if self.N > 1 :
            W_res_out=W_out[:,:self.N//2] # from hd0 only (excitatory)
        else : 
            W_res_out=W_out[:,:self.N]
        self.weights['hd0->out'].set_W(W_res_out)
        # Update the B value correspondingly, as they are newly connected
        #C = np.einsum('i,j->ij',self.weights['hd0->out'].D,self.layers[1].P)
        C = np.copy(self.layers[1].P)
        # At this point we normalize with the unity coupling coefficient
        C *= self.unity_coeff
        # Add the corresponding currents to Output layer
        self.layers[3].update_B(self.weights['hd0->out'],C)
            
        W_inout = np.zeros(self.weights['inp->out'].ask_W(silent=True))
        # Set the weights for the input-output coupling
        if self.N > 1 :
            W_inout=W_out[:,self.N//2:] # elements not from reservoir
        else :
            W_inout=W_out[:,self.N:] # elements not from reservoir
            
        self.weights['inp->out'].set_W(W_inout)
        # Also here we add the currents to B
        C = self.layers[0].C
        self.layers[3].update_B(self.weights['inp->out'],C)
        
        
    def show_network(self, savefig=False,layout='shell',**kwargs) :
        plotter.visualize_network(self.layers, self.weights, 
                                  #exclude_nodes={0:['I1','I2'],3:['O1','O2']},
                                  node_size=100,
                                  layout=layout, 
                                  show_edge_labels=False,
                                  savefig=savefig,
                                  **kwargs)
            
    def produce_movie(self,movie_series) :
        plotter.movie_maker(movie_series,self.layers, self.weights, 
                            exclude_nodes={0:['I1','I2'],3:['O1','O2']},
                            node_size=100,
                            layout='spring', 
                            show_edge_labels=False)
            
    def assign_device(self, device) :
        self.layers[1].assign_device(device)
        if self.N > 1 :
            self.layers[2].assign_device(device)
        self.unity_coeff, self.Imax = device.inverse_gain_coefficient(device.eta_ABC, self.layers[1].Vthres)
        
    def randomize_memory(self,noise=0.1,dist='normal',seed=None) :
        """Introduce a distribution of memory constants in the hidden layers"""
        for k in [1,2] :
            self.layers[k].multiA = True
            if dist=='normal' :
                self.layers[k].generate_Adist(noise)
            elif dist=='uniform' :
                self.layers[k].generate_uniform_Adist(noise,seed)
            elif dist=='exp' :
                self.layers[k].generate_exp_Adist(noise)
            elif dist=='poisson' :
                self.layers[k].generate_poisson_Adist(noise)
            else :
                print('Unexpected dist in randomize_memory')
                break
                
    def sample_memory(self):
        A33 = np.zeros((2,self.N//2))
        for k in [1,2] :
            A33[k-1] = self.layers[k].Adist[:,2,2]
        return A33
    
    def sample_A(self) :
        A33 = np.zeros((2,self.N//2,3,3))
        for k in [1,2] :
            A33[k-1] = self.layers[k].Adist
        return A33
    
    def specify_inputs(self,input_handle,bias_handle,teacher_handle,fixImax=None) :
        if fixImax is None :
            scaleImax = self.Imax
        else :
            scaleImax = fixImax
            
        if input_handle is not None :
            # Since the last revision, input layers are more general and nodes
            # are not strictly connected to a specific channel
            # Here we take 0 as bias and 1 as input, both in blue channel
            self.layers[0].set_input_func_per_node(1,
                                          func_handle=input_handle(scaleImax*self.input_scaling))    
            self.input_handle = input_handle
        
        if bias_handle is not None :
            self.layers[0].set_input_func_per_node(0,
                                          func_handle=bias_handle(scaleImax*self.bias_scaling))
            self.bias_handle = bias_handle
            
        if teacher_handle is not None :
            self.layers[3].set_output_func(func_handle=teacher_handle(scaleImax*self.teacher_scaling))
            self.teacher_handle = teacher_handle
            
    def set_delay(self, delay,nsave=None) :
        # This is the timescale of the system
        self.delay=delay
        self.layers[3].set_teacher_delay(self.delay,nsave)
        
    def evolve(self,T,reset=True,t0=0.,teacher_forcing=False,savestep=1,printstep=50) : 
        
        # Start time is t
        t=t0
        # These parameters are used to determine an appropriate time step each update
        dtmax = 0.5 # ns 
        dVmax = 0.01 # V # I loosen this a little bit now
        # To sample result over a fixed time-step, use savetime
        #savetime = max(savestep,dtmax)
        savetime = t0
        printtime = t0
        
        if reset :
            # option to keep reservoir in its current state
            nw.reset(self.layers)
            
        # Create a log over the dynamic data
        time_log = logger.Logger(self.layers,feedback=True) # might need some flags
        # write zero point
        time_log.add_tstep(t, self.layers, self.unity_coeff)
        
        start = time.time()
        
        while t < T:
            # evolve by calculating derivatives, provides dt
            dt = tm.evolve(t, self.layers, dVmax, dtmax )
        
            # update with explicit Euler using dt
            # supplying the unity_coeff here to scale the weights
            tm.update(dt, t, self.layers, self.weights, unity_coeff=self.unity_coeff, t0=t0, teacher_forcing=teacher_forcing,delay=self.timescale)
            
            t += dt
            # Log the progress
            if t > savetime :
                # Put log update here to have (more or less) fixed sample rate
                savetime += savestep                
                time_log.add_tstep(t, self.layers, self.unity_coeff)
            if t > printtime :
                # Now this is only to check progress
                print(f'Time at t={t} ns') 
                printtime += printstep   
                
            #time_log.add_tstep(t, self.layers, self.unity_coeff)
        
        end = time.time()
        print('Time used:',end-start)
        
        # This is a large pandas data frame of all system variables
        result = time_log.get_timelog()
        
        return result, t # send the exact time back as well
    
    def interp_columns(self,result,tseries,header_exp=None,columns=None,regex=None,return_df=False) :
        # TODO: Could be tidied up a bit with the regex
        from scipy.interpolate import interp1d 
        # Extract time column
        tcol = result['Time']
        # Extract all Pout for the states
        if header_exp is not None :
            headers = [c for c in result.columns if header_exp in c]
            df = result[headers]
        elif columns is not None :
            headers= columns
            df = result[headers]
        elif regex is not None :
            df = result.filter(regex=regex)
        else :
            df = result
            
        # Create interpolation function
        df_interp = interp1d(tcol,df,axis=0)
        
        if return_df :
            import pandas as pd
            df_new = pd.DataFrame(df_interp(tseries),columns=df.columns)
            try :
                # Try to add the time column as well
                df_new.insert(0,'Time',tseries)
            except ValueError :
                # If it is already there we end up here
                pass
            
        else :
            df_new = df_interp(tseries)
        
        return df_new
        
    def harvest_states(self,T,t0=0.,reset=True,teacher_forcing=True) :
        # First we evolve to T in time from optional t0
        result, tend = self.evolve(T,t0=t0,teacher_forcing=teacher_forcing,reset=reset)
        # Now we fit the output weights using ridge regression
        if not self.silent:
            print("harvesting states...")

        # Secondly, we employ a discrete sampling of the signals
        tseries = np.arange(t0,T,step=self.timescale,dtype=float)
        # States
        states_series = self.interp_columns(result,tseries,regex='H\d+-Pout')
        # Input signals
        inputs_columns = [c for c in result.columns if ('I0' in c) or ('I1' in c) or ('I2' in c)]
        inputs_series = self.interp_columns(result,tseries,columns=inputs_columns)
        # Teacher signal
        if teacher_forcing : 
            output_string='O0-Pinp'
        else :
            output_string='O0-Pout'
        teacher_series = self.interp_columns(result,tseries,header_exp=output_string)

        # Now we formulate extended states including the input signal
        extended_states = np.hstack((states_series, inputs_series))
        
        #print('Voltages at the last point (H,K):\n', self.layers[1].V)
        #print(self.layers[2].V)   
        #print('Currents out from H:', self.layers[1].P)
        
        
        return tseries, extended_states, teacher_series, tend

    def fit(self, states, target, beta=100, regularization=True) :
        # we'll disregard the first few states:
        transient = min(int(target.shape[0] / 10), 100)
        
        if regularization:
            # Use regularization parameter beta to find output weights
            Nx = states.shape[1]
            # Part to invert
            X = states[transient:].T @ states[transient:] + beta * self.Imax**2 * np.diag([1]*Nx)
            # Part including the target
            YX = target[transient:].T @ states[transient:]
            # Final expression
            W_out = YX @ np.linalg.inv(X)
        else :
            # Solve for W_out using direct pseudoinverse
            W_out = np.dot(np.linalg.pinv(states[transient:, :]),
                           target[transient:, :]).T
        
   
        # Now we need to specify some weights from the input and reservoir unit
        print('The following weights were found:\n', W_out)
        # apply learned weights to the collected states:
        self.add_trained_weights(W_out,'blue')

        # Generate the prediction for the traning data
        #pred_train = self._unscale_teacher(np.dot(states, 
        #                                          W_out.T))
        pred_train = np.dot(states,W_out.T) # scaling not necessary
        
        error = np.sqrt(np.mean((pred_train[transient:] - target[transient:])**2))/self.Imax
        
        if not self.silent:
            print('Training error:', error)
            
        return pred_train, error
        
    def predict(self,t0,T,output_all=False) :
        # Assume here that we continue on from the state of the reservoir
        if not self.silent:
            print("predicting...")
                
        # First we evolve to T in time from optional t0, without resetting 
        result, tend = self.evolve(T,t0=t0,reset=False)
        
        # Secondly, we employ a discrete sampling of the signals
        tseries = np.arange(t0,T,step=self.timescale,dtype=float)
        # States TWEAK TO WRITE OUT ALL OUTPUT
        #output_series = self.interp_columns(result,tseries,header_exp='O0-Pout')
        output_series = self.interp_columns(result,tseries,regex='O\d-Pout')
        movie_series = self.interp_columns(result,tseries,regex='Pout',return_df=True)
        unscaled_output = self._unscale_teacher(output_series) # not used at the moment
        plot_series = self.interp_columns(result,tseries,return_df=True)
        
        if output_all :
            return tseries, output_series, movie_series, plot_series
        else :
            return tseries, output_series, movie_series
        