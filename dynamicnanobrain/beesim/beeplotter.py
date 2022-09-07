#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 12 11:02:27 2022

@author: dwinge
"""
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

nature_single = 89.0 / 25.4
nature_double = 183.0 / 25.4
nature_full = 247.0 / 25.4

def filter_trace(trace,sigma) :
    from scipy.ndimage import gaussian_filter1d
    return gaussian_filter1d(trace, sigma,axis=0)

def subplot_trace(target, res, layer, attr, titles, filtering) :
    # Get the relevant nodes
    columns = [name for name in res.columns if (attr in name) and (layer in name)]
    
    # If we are plotting CPU1, we permute the one step forward
    if layer == 'CPU1' :
        columns.insert(0,columns.pop(-1))
    if layer == 'CPUin' :
        columns.insert(0,columns.pop(-1))
    
    if layer == 'TN2' and filtering :
        print('Filtering')
        res[columns] = filter_trace(res[columns],sigma=50)
    #print(columns)
   
    if layer=='motor' :
        target.plot(res['Time'],res['motor-R'],label='motor-R')
        target.plot(res['Time'],res['motor-L'],label='motor-L')
        target.legend()
        target.set_ylabel('Summed activity')
    else :
        if columns[0][3] == 'V' :
            ylabel = 'Voltage (V)' 
        else:
            ylabel = 'Current (nA)'
        
        #TIME, INDEX = np.meshgrid(res['Time'])
        node_idx = [x+1 for x in range(0,len(columns))]
        # Need to copy as assigned by reference
        node_labels = node_idx.copy()
        if layer == 'CPU1' :
            node_labels[0] = 'CPU1b_9'
            node_labels[-1] = 'CPU1b_8'
            
        import numpy as np
        TIME, INDEX = np.meshgrid(res['Time'].values,node_idx)
        # Produce a 2D plot of values over time
        im = target.pcolormesh(TIME,INDEX,res[columns].values.transpose(),
                               cmap='viridis', rasterized=True,
                               shading='auto')
    
        plt.colorbar(im, ax=target, label=ylabel)  
        target.set_yticks(np.array(node_idx))
        target.set_yticklabels(node_labels)
        target.set_ylabel('Node idx')
    
    target.set_xlabel('Time (ns)')
    if titles :
        target.set_title(layer)
    
def plot_motor(res, onecolumn=False,doublewidth=True) :
    Nrows = 1 ; Ncols = 1
    if doublewidth : 
        nature_width = nature_double 
    else :  
        nature_width = nature_single
        
    fig, ax = plt.subplots(Nrows, Ncols, 
                            figsize=(nature_width*Ncols, 0.5*nature_single*Nrows),
                            sharex=True) 
    

    return fig, ax

def plot_homing_dist(cpu4, summed=None) :
    
    mem_color_L = '#F37D4D'
    mem_color_R = '#FABD5E'

    fig, ax = plt.subplots(figsize=(nature_double,nature_single),
                            sharex=True) 
    
    indices = range(1,9)
    ax.plot(indices,cpu4[:8],color=mem_color_L,label='cpu4[:8]')
    ax.plot(indices,cpu4[8:],color=mem_color_R,label='cpu4[8:]')
    if summed is not None :
        ax.plot(indices,summed,'k--')
    ax.legend()
    ax.set_xticks(np.arange(1,len(indices)+1,dtype=int))
    ax.set_xticklabels(['1/16','2/9','3/10','4/11','5/12','6/13','7/14','8/15'])
    ax.set_xlabel('CPU4 neuron index')
    ax.set_ylabel('Output (nA)')
    plt.tight_layout()
    
    return fig, ax
    

def plot_traces(res, layers, attr, onecolumn=False, doublewidth=True,
                time_interval=None, titles=False, filtering=False)    :
           
    import warnings
    warnings.filterwarnings('ignore',category=UserWarning) # get rid of some red text...
    
    Nrows = len(layers)
    Ncols = 1 # Put traces with a shared x-axis
    if doublewidth : 
        nature_width = nature_double 
    else :  
        nature_width = nature_single
        
    fig, axs = plt.subplots(Nrows, Ncols, 
                            figsize=(nature_width*Ncols, 0.5*nature_single*Nrows),
                            sharex=True) 
    
    # Select the approperiate time interval
    if time_interval is not None :
        select_res = res[(res["Time"]>=time_interval[0]) & (res["Time"]<=time_interval[1])]
    else : 
        select_res = res
        
    if Nrows > 1 :
        for k, ax in enumerate(axs.flatten()) :
            subplot_trace(ax, select_res, layers[k], attr, titles, filtering)
    else:
        subplot_trace(axs, select_res, layers[0], attr, titles, filtering)
        
    plt.subplots_adjust(left=0.124, right=0.9, bottom=0.1, top=0.9, wspace=0.1)
    plt.tight_layout()
    
    return fig, axs

def plot_tortuosity(cum_min_dist, ax=None,
                    label_font_size=10, unit_font_size=10,title=None) :
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(nature_single, nature_single))

    mu = np.nanmean(cum_min_dist, axis=1)
    std = np.nanstd(cum_min_dist, axis=1)

    xvals = np.linspace(0,2,cum_min_dist.shape[0])
        
    ax.plot(xvals,mu)
    ax.plot(xvals,mu+std,'k--')
    ax.plot(xvals,mu-std,'k--')
    ax.plot([0.0,1.0],[1.0,0.0],'-')
    
    ax.set_title(title, fontsize=label_font_size)
    ax.set_xlabel('Distance traveled / turning point distance', fontsize=label_font_size)
    ax.set_ylabel('Distance from home (fraction)', fontsize=label_font_size)
    ax.set_xlim(0,2)
    ax.set_ylim(0,1)
    plt.tight_layout()
    
    return fig, ax

def plot_distance_v_param(min_dists, min_dist_stds, distances, param_vals,
                          param_name,ylabel='Distance (steps)',
                          ax=None, label_font_size=11, unit_font_size=10,
                          title=None, xmin=10,xmax=10000, ymax=300,xticks=None,
                          reformat_legend=False):
    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=(nature_single, nature_single))

    colors = [cm.viridis(x) for x in np.linspace(0, 1, len(param_vals))]

    for i in range(len(param_vals)):
        noise = param_vals[i]
        mu = min_dists[i]
        sigma = min_dist_stds[i]
        if noise != 'Random':
            ax.semilogx(distances, mu, color=colors[i], label=noise, lw=1);
        else:
            ax.semilogx(distances, mu, color=colors[i], label='Random walk',
                        lw=1);
        ax.fill_between(distances,
                        [m+s for m, s in zip(mu, sigma)],
                        [m-s for m, s in zip(mu, sigma)],
                        facecolor=colors[i], alpha=0.2);

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(0, ymax)
    ax.set_title(title, fontsize=label_font_size)
    ax.tick_params(labelsize=unit_font_size)
    if xticks is not None :
        import matplotlib
        ax.set_xticks(xticks)
        # Cancel the formatting
        ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        
    ax.set_xlabel('Route length (steps)', fontsize=label_font_size)
    ax.set_ylabel(ylabel, fontsize=label_font_size)

    handles, labels = ax.get_legend_handles_labels()
    # Reformat labels here
    if reformat_legend :
        labels = [f'{val:.0f}%' for val in param_vals]
    
    l = ax.legend(handles,
                  labels,
                  loc='best',
                  fontsize=label_font_size,
                  #handlelength=0,
                  #handletextpad=0,
                  title=f'{param_name}:')
    l.get_title().set_fontsize(label_font_size)
    for i, text in enumerate(l.get_texts()):
        text.set_color(colors[i])
    #for handle in l.legendHandles:
    #    handle.set_visible(False)
    l.draw_frame(False)
    plt.tight_layout()
    return fig, ax

def plot_angular_distance_histogram(angular_distance, scale=1.0, ax=None, bins=36,
                                    color='b',labelname=''):
    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=(nature_single, nature_single))

    radii = np.histogram(angular_distance,
                         np.linspace(-np.pi - np.pi / bins,
                                     np.pi + np.pi / bins,
                                     bins + 2,
                                     endpoint=True))[0]
    radii[0] += radii[-1]
    radii = radii[:-1]
    radii = np.roll(radii, bins//2)
    radii = np.append(radii, radii[0])
    # Set all values to have at least a count of 1
    # Need this hack to get the plot fill to work reliably
    radii[radii == 0] = 1
    theta = np.linspace(0, 2 * np.pi, bins+1, endpoint=True)

    ax.plot(theta, scale*radii, color=color, alpha=0.5,label=labelname)
    if color:
        ax.fill_between(theta, 0, radii*scale, alpha=0.2, color=color)
    else:
        ax.fill_between(theta, 0, radii*scale, alpha=0.2)

    return fig, ax

def plot_angular_distances(noise_levels, angular_distances, bins=18, ax=None,
                           label_font_size=11, log_scale=False, title=None, scale=None):
    fig = None
    if ax is None:
        fig, ax = plt.subplots(subplot_kw=dict(projection='polar'),
                               figsize=(nature_single, nature_single))

    colors = [cm.viridis(x) for x in np.linspace(0, 1, len(noise_levels))]

    if scale is None :
        scale = [1.0]*len(noise_levels)
    for i in reversed(range(len(noise_levels))):
        plot_angular_distance_histogram(angular_distance=angular_distances[i],scale=scale[i],
                                        ax=ax, bins=bins, color=colors[i],labelname=noise_levels[i])

    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_rlabel_position(22)
    ax.set_title(title, y=1.08, fontsize=label_font_size)
    ax.legend(labelcolor='linecolor')
    if log_scale:
        ax.set_rscale('log')
        ax.set_rlim(0.0, 10001)  # What determines this?

    plt.tight_layout()
    return fig, ax