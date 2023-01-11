#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  3 15:18:44 2021

@author: dwinge
"""

import numpy as np

def check_physics(layers) :
    """
    Small routine to check so that there is a device attached to the hidden
    layer nodes

    Parameters
    ----------
    layers : dictionary of layer objects

    Raises
    ------
    Exception
        If there is no attached Device object

    Returns
    -------
    None.

    """
    for layer in layers.values() :
        if layer.layer_type == 'hidden' :
            if layer.device is None :
                print('Error: No device assigned to hidden layer')
                raise Exception
    
# TODO: This part should be updated with an algorithm takes a time-step also
# in input currents to check if the time-step needs to be limited due to a 
# future pulse. For this the input currents need to updated 'ahead of time'
def evolve(t, layers, dVmax, dtmax) :
    # First we calculate the derivative of the internal variables to get dt
    max_dV = 0.
    for layer in layers.values() :
        if layer.layer_type == 'hidden' :
            # Calculate the derivative of the voltages
            dV = layer.get_dV(t)
            max_dV_layer = np.abs(dV).max()
            # Save the largest dV we find
            if max_dV_layer > max_dV :
                max_dV = max_dV_layer
                

    # Choose time-step accordingly
    if max_dV > 0 :
        dt = dVmax/max_dV
    else :
        dt = dtmax
        
    #dt = min(dtled,dt)
    # Check also against maximum timestep
    dt = min(dt, dtmax)
    
    return dt
            
def update(dt, t, layers, weights, unity_coeff=1.0, t0=0.,teacher_forcing=False, noise=None,delay=0) :   
    # Time updating sequence
    # Update first all voltages V and reset currents in matrices B
    for key, layer in layers.items() :
        if layer.layer_type == 'hidden' :
            layer.update_V(dt)
            layer.update_I(dt) 
        
        if layer.layer_type == 'input' :
            layer.update_C(t)
        
        # This should be done after receiving the information from the network
        if layer.layer_type == 'output' :
            if teacher_forcing :
                layer.update_C(t-delay)           
            else :
                # Send the signals back into the network, if connected
                layer.update_C_from_B(t,t0)
                    
        # Add a prespecified noise value
        if noise is not None :
            try :
                layer.add_noise(noise[key](t))
            except :
                pass
            
        layer.reset_B()
    
    # Now we rewrite the currents B according to the weight rules
    for w in weights.values() :
        from_idx = w.from_layer
        to_idx = w.to_layer
        if layers[from_idx].layer_type == 'hidden' :
            # The values are basically the output signal
            C = np.copy(layers[from_idx].P)
            # At this point we normalize with the unity coupling coefficient
            C *= unity_coeff
        else :
            C = layers[from_idx].C
                       
        layers[to_idx].update_B(w,C)
    
    # Time step complete
    return 