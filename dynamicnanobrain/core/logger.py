#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  8 13:46:38 2021

@author: dwinge
"""

import pandas as pd

class Logger :
    
    def __init__(self,layers,feedback=False) :
        self.list_data = []
        # Need to get the node names
        self.feedback = feedback
        self.column_labels = self.column_names(layers)
        
    def column_names(self, layers) :
        names=['Time']
        for idx in layers.keys():
            node_list = layers[idx].get_names(idx)          
            if layers[idx].layer_type == 'input' :
                for node in node_list :
                    names.append(node+'-Pout')
                
            elif layers[idx].layer_type == 'hidden' :
                # Voltages
                for node in node_list :
                    names.append(node+'-Vinh')
                    names.append(node+'-Vexc')
                    names.append(node+'-Vgate')
                # Input currents
                for node in node_list :
                    names.append(node+'-Iinh')
                    names.append(node+'-Iexc')
                # Output currents
                for node in node_list :
                    names.append(node+'-Iout')
                # ISD (source drain) done separately to get correct ordering
                for node in node_list :
                    names.append(node+'-ISD')  
                for node in node_list :
                    names.append(node+'-Pout')
                # Add the LED voltage
                #for node in node_list :
                #    names.append(node+'-Vled')
             
            elif layers[idx].layer_type == 'output' :
                # Currents
                for node in node_list :
                    names.append(node+'-Pout')
                if self.feedback :
                    # add some extra columns for the signal fed back in (C)
                    for node in node_list :
                        names.append(node+'-Pinp')
                    
            else :
                print('Unexpected layer_type in logger.column_names')
                raise RuntimeError
        return names
        
    def add_tstep(self,t,layers, unity_coeff=1.0) :
        # Extract the data from each node in layers
        row = [t]
        for idx in layers.keys():
            # Node names
            #name_list = layers[idx].get_names(idx)       
            if layers[idx].layer_type == 'input' :
                curr=layers[idx].C.flatten(order='F').tolist()
                row += curr 
            
            elif layers[idx].layer_type == 'hidden' :
                # Voltages
                volt=layers[idx].V.flatten(order='F').tolist()
                row +=volt
                # Input currents
                curr=layers[idx].B[:2].flatten(order='F').tolist()
                row += curr 
                # Output currents
                # Here I add the normalization also to the recorded output
                curr=layers[idx].I*unity_coeff
                # Now convert to list
                curr=curr.flatten(order='F').tolist() 
                #curr=layers[idx].I.flatten(order='F').tolist() 
                row += curr
                curr=layers[idx].ISD.flatten(order='F').tolist()
                row += curr 
                pout=layers[idx].P*unity_coeff
                pout=pout.flatten(order='F').tolist()
                row +=pout
                #volt=layers[idx].Vled.flatten(order='F').tolist()
                #row +=volt
                
            elif layers[idx].layer_type == 'output' :
                # Voltages
                curr=layers[idx].B.flatten(order='F').tolist()
                row += curr 
                if self.feedback :
                    curr=layers[idx].C.flatten(order='F').tolist()
                    row += curr
            else :
                print('Unexpected layer_type in logger.add_tstep')
                raise RuntimeError
                
        self.list_data.append(row)
        
    def get_timelog(self) :
        # Convert to pandas data frame
        return pd.DataFrame(self.list_data, columns=self.column_labels)
