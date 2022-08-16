#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  3 14:47:12 2021

@author: dwinge
"""

import matplotlib.pyplot as plt
#import pandas as pd
import networkx as nx
import matplotlib.colors as mcolors
from pathlib import Path

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
        'size'   : 10}
plt.rc('font', **font)


def name_edges(weights, layers) :
    # Aquire the numerical indicies
    num_edges = weights.get_edges()
    # Loop through the numerical indicies and ask for names
    from_layer=weights.from_layer
    to_layer = weights.to_layer
    # Loop over channels and name edges
    named_edges = {}
    for key in num_edges : 
        edge_list = []
        for edge in num_edges[key]:
            down=edge[0]
            up = edge[1]
            weight = edge[2]
            down_name=layers[from_layer].get_node_name(down,from_layer)
            up_name=layers[to_layer].get_node_name(up,to_layer)
            edge = (down_name,up_name,weight)
            # Need to explicitly replace the value
            edge_list.append(edge)
        named_edges[key] = edge_list
    # return translated dictionary
    return named_edges

def name_nodes(layers) :
    nodes = {}
    for key in layers :
        node_names = layers[key].get_names(key)
        nodes[key] = node_names
    return nodes

def retrieve_G(layers, weights) :
    edges = {}
    for key in weights :
        edges[key] = name_edges(weights[key],layers)
    
    nodes = name_nodes(layers)
    
    # Construct a graph
    G = nx.DiGraph()
    for key in nodes :
        G.add_nodes_from(nodes[key], subset=key)
    for edge_set in edges.values() :
        for key in edge_set :
            G.add_weighted_edges_from(edge_set[key],color=key)
    
    return G
    
def visualize_network(layers, weights, exclude_nodes={}, 
                      exclude_layers=[], node_size=600,
                      layout='multipartite', show_edge_labels=True, 
                      shell_order=None, savefig=False, font_scaling=6,
                      arrow_size=20, **kwargs) :
    edges = {}
    for key in weights :
        edges[key] = name_edges(weights[key],layers)
    
    nodes = name_nodes(layers)
    # Remove specific nodes in specific layers
    for key in exclude_nodes :
        for node in exclude_nodes[key] :
            nodes[key].remove(node)
    # Remove layers using their key
    for key in exclude_layers:
        del nodes[key]
        # Search for corresponding edges
        edges_to_remove = [edge for edge in edges.keys() if key in edge]
        # Now we can delete the keys without changing what we loop over
        for edge_key in edges_to_remove :
            del edges[edge_key]
            
    
    # Construct a graph
    G = nx.DiGraph()
    for key in nodes :
        G.add_nodes_from(nodes[key], subset=key)
    for edge_set in edges.values() :
        for key in edge_set :
            G.add_weighted_edges_from(edge_set[key],color=key)

    val_map = {'I': 0.5,
               'H': 0.6,
               'O': 0.7}
    
    values = [val_map.get(node[0], 0.45) for node in G.nodes()]
    edge_labels=dict([((u,v,),f"{d['weight']:.1f}")
                     for u,v,d in G.edges(data=True)])
    edge_colors=['tab:'+d['color'] for u,v,d in G.edges(data=True)]
    edge_weights=[d['weight'] for u,v,d in G.edges(data=True)]
    
    # Try new way of constructing this
    #edge_labels = dict([((n1, n2), f'{n1}->{n2}')
    #                for n1, n2 in G.edges])
    
    #red_edges = [('I1','H0')]
    red_edges = []
    #edge_colors = ['black' if not edge in red_edges else 'red' for edge in G.edges()]
    black_edges = [edge for edge in G.edges() if edge not in red_edges]
    
    if layout=='multipartite' :
        pos=nx.multipartite_layout(G)
    elif layout=='spring' :
        pos=nx.spring_layout(G, iterations=10000, threshold=1e-5, **kwargs)
    elif layout=='circular' :
        pos=nx.circular_layout(G)
    elif layout=='spiral' :
        pos=nx.spiral_layout(G)
    elif layout=='kamada_kawai' :
        pos=nx.kamada_kawai_layout(G, **kwargs)
    elif layout=='shell' :
        nlist = []
        # Combine the output and input layer on the same circle
        if shell_order is None:
            # Number of layers
            P = len(nodes.keys())
            for key in nodes:
                if key < P-1 :
                    nlist.append(nodes[key])
                else :
                    nlist[0] += nodes[key]
            # Reverse list to have input + output as outer layer
            nlist = nlist[::-1]
            
        else :
            for entry in shell_order :
                if type(entry) is list:
                    nlist.append(nodes[entry[0]])
                    for k in range(1,len(entry)) :
                        nlist[-1] += nodes[entry[k]]
                else :
                    nlist.append(nodes[entry])
                
        pos=nx.shell_layout(G,nlist=nlist)
        
    else :  
        print('Sorry, layout not implemented, reverting back to multipartite')
        pos=nx.multipartite_layout(G)
        
    # Try simple scaling
    c = node_size/600
    
    nx.draw_networkx_nodes(G, pos, cmap=plt.get_cmap('Blues'), 
                           node_color = values, vmin=0., vmax=1.0,
                           node_size = node_size)
    nx.draw_networkx_labels(G, pos, font_size=(6+c*font_scaling))
    nx.draw_networkx_edges(G, pos, edgelist=red_edges, edge_color='r', 
                           arrows=True, arrowsize=arrow_size,node_size=node_size)
    nx.draw_networkx_edges(G, pos, edgelist=black_edges, edge_color=edge_colors,
                           arrows=True, arrowsize=arrow_size,node_size=node_size,
                           width=edge_weights,
                           connectionstyle='arc3,rad=.2')

                          # connectionstyle='arc3,rad=0.2')
    if show_edge_labels :
        nx.draw_networkx_edge_labels(G,pos,edge_labels=edge_labels)
    
    # There is an interface to graphviz .dot files provided by 
    #nx.drawing.nx_pydot.write_dot(G, 'graph.dot')
    # to generate a png, run dot -Tpng graph.dot > graph.png
    if savefig :
        #nx.drawing.nx_pydot.write_dot(G, 'network_layout.dot')
        plt.savefig('network_layout.png',dpi=300)
        plt.savefig('network_layout.svg')
    
    plt.show()
    
    return pos

def simple_paths(G,source, target) :
    paths = nx.all_simple_paths(G,source,target)
    return paths

def movie_maker(movie_series, layers, weights, exclude_nodes={}, node_size=600, layout='multipartite', show_edge_labels=True, shell_order=None) :
    from matplotlib.animation import FuncAnimation
    # Setup the network plot from the layers and weights
    edges = {}
    for key in weights :
        edges[key] = name_edges(weights[key],layers)
    
    nodes = name_nodes(layers)
    for key in exclude_nodes :
        for node in exclude_nodes[key] :
            nodes[key].remove(node)
        
    # Construct a graph
    G = nx.DiGraph()
    for key in nodes :
        G.add_nodes_from(nodes[key], subset=key)
    for edge_set in edges.values() :
        for key in edge_set :
            G.add_weighted_edges_from(edge_set[key],color=key)


    # Small loop here to set the colors
    # Fancy colors
    #val_map = {0: 'tab:blue',
    #           1: 'tab:red',
    #           2: 'tab:green'}
    # RGB colors
    val_map = {0: 'blue',
               1: 'red',
               2: 'green'}
    
    values=[]
    for node in G.nodes() :
        if node[0]=='H' :
            values.append(val_map[0])
        elif node[0]=='K' :
            values.append(val_map[1])
        else :
            values.append(val_map.get(int(node[1])))
    
    edge_labels=dict([((u,v,),f"{d['weight']:.1f}")
                     for u,v,d in G.edges(data=True)])
    # Fancy colors
    #edge_colors=['tab:'+d['color'] for u,v,d in G.edges(data=True)]
    # RGB colors
    edge_colors=[d['color'] for u,v,d in G.edges(data=True)]
    # Specified dynamically
    #edge_weights=[d['weight'] for u,v,d in G.edges(data=True)]
       
    if layout=='multipartite' :
        pos=nx.multipartite_layout(G)
    elif layout=='spring' :
        pos=nx.spring_layout(G)
    elif layout=='circular' :
        pos=nx.circular_layout(G)
    elif layout=='spiral' :
        pos=nx.spiral_layout(G)
    elif layout=='kamada_kawai' :
        pos=nx.kamada_kawai_layout(G)
    elif layout=='shell' :
        nlist = []
        # Combine the output and input layer on the same circle
        if shell_order is None:
            # Number of layers
            P = len(nodes.keys())
            for key in nodes:
                if key < P-1 :
                    nlist.append(nodes[key])
                else :
                    nlist[0] += nodes[key]
            # Reverse list to have input + output as outer layer
            nlist = nlist[::-1]
            
        else :
            for entry in shell_order :
                if type(entry) is list:
                    nlist.append(nodes[entry[0]])
                    for k in range(1,len(entry)) :
                        nlist[-1] += nodes[entry[k]]
                else :
                    nlist.append(nodes[entry])
                
        pos=nx.shell_layout(G,nlist=nlist)
        
    else :  
        print('Sorry, layout not implemented, reverting back to multipartite')
        pos=nx.multipartite_layout(G)
        
    # Try simple scaling
    c = node_size/600

    # Need a way to scale the alpha of each node depending on its activity
    # Start by renaming the DataFrame columns for easy access
    short_names = []
    for name in movie_series.columns :
        idx = name.find('-')
        if idx == -1 : idx = None
        short_names.append(name[:idx])
        
    changes=dict(zip(movie_series.columns,short_names))
    movie_series.rename(columns=changes,inplace=True)
    # Check maximum current in movies_series
    Imax=max(movie_series.max()) # .max() gives a column max

    # Now the alpha can be retrieved as a list
    tseries = movie_series['Time'] # easier to call
    idx = tseries.first_valid_index()
    alpha_P = []
    for node in G.nodes() :
        # Calculate transparancy normalized to 1.
        alpha_P.append(movie_series[node][idx]/Imax)
        

    # At this point we have G and pos which is what we need NEW:
    
    # Create a fixed size figure
    # fig, ax = plt.subplots(figsize=(5,5))
    fig = plt.figure(figsize=(5,7))
    ax1 = plt.subplot(411)
    ax2 = plt.subplot(4,1,(2,4),aspect=1.)
    
    def init() :
        movie_series.plot(x = 'Time', y = 'O0',ax=ax1)
        ax1.set_xlabel('Time (ns)')
        ax1.set_ylabel('O0-Pout (nA)')
        
    def update(idx) :
        ax2.clear()
        try :
            ax1.lines[1].remove()
        except :
            pass
        
        # Draw a dot to mark our point
        t = tseries.loc[idx]
        Pout = movie_series['O0'].loc[idx]
        ax1.plot(t,Pout,'ro',ms=5.)
                
        # Update our values of alpha 
        alpha_P = []
        for node in G.nodes() :
            # Calculate transparancy normalized to 1.
            alpha_P.append(max(movie_series[node][idx],0)/Imax)
            
        # Scale also edges by the activity of the sending node
        edge_weights=[d['weight']*movie_series[u][idx]/Imax for u,v,d in G.edges(data=True)]
        
        nx.draw_networkx_edges(G, pos, edgelist=G.edges(), edge_color=edge_colors,
                               arrows=True, arrowsize=5,node_size=node_size,
                               width=edge_weights,
                               connectionstyle='arc3,rad=.2')
        try :
            allnodes = nx.draw_networkx_nodes(G, pos, cmap=plt.get_cmap('Blues'), 
                                              node_color = values, vmin=0., vmax=1.0,
                                              node_size = node_size, alpha=alpha_P)
        except ValueError:
            print(f'Encountered error at t={float(tseries.loc[idx]):.1f} ns, idx={idx}')
            print('Values are:')
            print(values)
            print('Alphas are:')
            print(alpha_P)
            
        allnodes.set_edgecolor("black")
        nx.draw_networkx_labels(G, pos, font_size=(6+c*2))
        #nx.draw_networkx_edges(G, pos, edgelist=red_edges, edge_color='r', 
        #                       arrows=True, arrowsize=20,node_size=node_size)

    
                              # connectionstyle='arc3,rad=0.2')
        if show_edge_labels :
            nx.draw_networkx_edge_labels(G,pos,edge_labels=edge_labels)
    
        ax1.set_title(f't={float(tseries.loc[idx]):.1f} ns')
        
    # Create the animation 
    ani = FuncAnimation(fig, 
                        update, 
                        frames=range(tseries.first_valid_index(),tseries.last_valid_index()),
                        repeat=False,
                        init_func=init)

    ani.save('movie.mp4')
    
    # Show animation in the end    
    plt.show()

def bode_plot(device,plot_f_min=1e-3,plot_f_max=10,indicate_freq=None) :
    import numpy as np
    # Units are already in GHz so this creates a space from 1 MHz to 10 GHz
    Ns=100
    s_exp = np.linspace(-4,1,Ns)
    s = 1j*10**s_exp

    # Sample the gain function
    eigvals = device.setup_gain(device.gammas)
    G11, _ = device.gain(s,eigvals,device.gammas)

    # Amplitude (normalized) and phase data
    mag_G11 = np.absolute(G11) / np.absolute(G11[0])
    arg_G11 = np.angle(G11)

    # Use decibel units
    mag_G11_dB = 20*np.log10(mag_G11)

    # Produce Bode plots of the results
    fig, (ax1, ax2) = plt.subplots(2,1,figsize=(nature_double,nature_single),sharex=True)

    f_min = abs(s[0])

    ax1.plot(abs(s),mag_G11_dB)
    ax1.plot([f_min,plot_f_max],[-3,-3],'k--')
    ax1.grid('True')
    ax1.set_xscale('log')
    ax1.set_title('Bode plots for dynamic node')
    #ax1.set_xlabel('Frequency (GHz)')
    ax1.set_ylabel('|G11|/|G11[0]| (dB)')
    ax1.set_ylabel('|H| (dB)')
    ax1.set_xlim(plot_f_min,plot_f_max)
    ax1.set_ylim(-20,2)

    ax2.plot(abs(s),arg_G11*180/np.pi)
    ax2.grid('True')
    #ax2.set_xscale('log')
    ax2.set_xlabel('Frequency (GHz)')
    ax2.set_ylabel('Phase (radians)')
    ax2.set_ylim(-180,10)
    
    if indicate_freq is not None :
        ax1.plot([indicate_freq,indicate_freq],ax1.get_ylim(),'r--',lw=2)
        ax2.plot([indicate_freq,indicate_freq],ax2.get_ylim(),'r--',lw=2)
        
    plt.tight_layout()

    return fig, ax1, ax2

def bode_multi_plot(devices,plot_f_min=1e-3,plot_f_max=10,indicate_freq=None) :
    import numpy as np
    # MAKE SURE DEVICES IS A DICTIONARY indexed [-k, ..., -1, 0, 1... k]
    # Units are already in GHz so this creates a space from 1 MHz to 10 GHz
    Ns=100
    s_exp = np.linspace(-3,1,Ns)
    s = 1j*10**s_exp
    
    # Sample the gain function for all devices
    G11 = {}
    for k, device in devices.items() :
        eigvals = device.setup_gain(device.gammas)
        G11[k], _ = device.gain(s,eigvals,device.gammas)

    

    # Produce Bode plots of the results
    fig, (ax1, ax2) = plt.subplots(2,1,figsize=(nature_double,nature_single),sharex=True)

    f_min = abs(s[0])

    for k in devices :
        # Amplitude (normalized) and phase data
        mag_G11 = np.absolute(G11[k]) / np.absolute(G11[k][0])
        arg_G11 = np.angle(G11[k])

        # Use decibel units
        mag_G11_dB = 20*np.log10(mag_G11)
        if k == 0 :
            style = '-'
            my_lw=2
        else :
            style = '--'
            my_lw=1
            
        ax1.plot(abs(s),mag_G11_dB,style,color=colors[0],lw=my_lw)
        ax2.plot(abs(s),arg_G11*180/np.pi,style,color=colors[0],lw=my_lw)
        
    ax1.plot([f_min,plot_f_max],[-3,-3],'k--')
    ax1.grid('True')
    ax1.set_xscale('log')
    ax1.set_title('Bode plots for dynamic node')
    #ax1.set_xlabel('Frequency (GHz)')
    ax1.set_ylabel('|G11|/|G11[0]| (dB)')
    ax1.set_ylabel('|H| (dB)')
    ax1.set_xlim(plot_f_min,plot_f_max)
    ax1.set_ylim(-20,2)

    
    ax2.grid('True')
    #ax2.set_xscale('log')
    ax2.set_xlabel('Frequency (GHz)')
    ax2.set_ylabel('Phase (radians)')
    ax2.set_ylim(-180,10)
    
    if indicate_freq is not None :
        ax1.plot([indicate_freq,indicate_freq],ax1.get_ylim(),'r--',lw=2)
        ax2.plot([indicate_freq,indicate_freq],ax2.get_ylim(),'r--',lw=2)
        
    plt.tight_layout()

    return fig, ax1, ax2

def plot_weights(weights, colormap='viridis', savefig=False) :
    
    import numpy as np
    import matplotlib.cm as cm
    
    titles = weights.keys()
    Nplots = len(titles)
    Ncols = 4 # set max columns to 4
    Nrows = Nplots // Ncols + min(Nplots % Ncols,1)
    
    ticklabels = {'TL2': range(1, 17),
                  'CL1': range(1, 17),
                  'TB1': range(1, 9),
                  'TN2': ['L', 'R'],
                  'Rectifier': range(1,17),
                  'CPU4': range(1, 17),
                  'Pontine': range(1, 17),
                  'CPU1a': range(2, 16),
                  'CPU1b': range(8, 10)}
    
    fig, ax = plt.subplots(Nrows, Ncols, 
                           figsize=(nature_single*Nrows,nature_single*Ncols))
    
    for i, key in enumerate(weights):
        cax = ax[i // Ncols][i % Ncols]
        # Get some data, sources and targets
        weight_mat = np.copy(weights[key].W)
        # Normalize
        #print(key)
        norm = weight_mat.max()
        weight_mat /= norm
        
        source, target = key.split('->')
        
        p = cax.pcolor(weight_mat, cmap=colormap,vmin=-1, vmax=1)
        p.set_edgecolor('face')
        cax.set_aspect('equal')

        cax.set_xticks(np.arange(weight_mat.shape[1]) + 0.5)
        cax.set_xticklabels(ticklabels[source])

        cax.set_yticks(np.arange(weight_mat.shape[0]) + 0.5)
        cax.set_yticklabels(ticklabels[target])

        if i == 12:
            cax.set_title(source + ' to ' + target, y=1.41)
        else:
            cax.set_title(source + ' to ' + target)

        cax.set_xlabel(source + ' cell indices')
        cax.set_ylabel(target + ' cell indices')
        cax.tick_params(axis=u'both', which=u'both', length=0)

    # delete extra axes
    for k in range(i+1,Nrows*Ncols) :
        cax = ax[k // Ncols][k % Ncols]
        fig.delaxes(cax)

    #cbax = fig.add_axes([1.02, 0.05, 0.02, 0.9])
    cbax = fig.add_axes([0.85, 0.05, 0.02, 0.3])
    m = cm.ScalarMappable(cmap=colormap)
    m.set_array(np.linspace(-1.0, 1.0, 100))
    cb = fig.colorbar(m, cbax, ticks=[-1, -0.5, 0, 0.5, 1])
    cb.set_label('Connection Strength', labelpad=10)
    cb.ax.set_yticklabels(['-1.0','-0.5','0.0','0.5','1.0'])
    #cb.set_clim(0.,1.)
    
    plt.subplots_adjust(
        left = 0.05,  # the left side of the subplots of the figure
        right = 0.95,   # the right side of the subplots of the figure
        bottom = 0.05,  # the bottom of the subplots of the figure
        top = 0.95,     # the top of the subplots of the figure
        wspace = 0.3,  # the amount of width reserved for space between subplots,
                      # expressed as a fraction of the average axis width        
        hspace = 0.2,  # the amount of height reserved for space between subplots,
                      # expressed as a fraction of the average axis height)
                      )
    
    #plt.tight_layout()
    
    return fig, ax
    
def visualize_scaled_result(res, columns, scaling=None, time_interval=None) :
    # Make a copy to do nasty things to
    scaled  = res.copy()
    if scaling is not None :
        for k, col in enumerate(columns) :
            scaled[col] *= scaling[k]
            
    # Send back to visualize results
    visualize_dynamic_result(scaled,columns,time_interval)
    
def visualize_dynamic_result(res, columns, time_interval=None) :
    
    if time_interval is not None :
        select_res = res[(res["Time"]>=time_interval[0]) & (res["Time"]<=time_interval[1])]
    else : 
        select_res = res
    
    # Pretty generic plot function
    select_res.plot(x = 'Time', y = columns,
                    xlabel='Time (ns)', ylabel='Voltage/Current (V/nA)')
        
    plt.gca().grid(True)
    
def plot_sum_nodes(res, layers, quantity, time_interval=None) :
    import pandas as pd
    # First we select the correct time span
    if time_interval is not None :
        select_res = res[(res["Time"]>=time_interval[0]) & (res["Time"]<=time_interval[1])]
    else : 
        select_res = res
        
    # Pick out the time vector
    time = res['Time']
    
    # Construct a df for the wanted values
    df = pd.DataFrame(columns=layers)
    for char in layers :
        regex = char + '.\d?-' + quantity # Example 'H.\d?-Pout'
        df[char] = select_res.filter(regex=regex).sum(axis=1)
        
    df.insert(0,'Time',time)
    
    df.plot(x = 'Time', y = layers, xlabel='Time (ns)', ylabel='Voltage/Current (V/nA)',
            title=f'Summed {quantity} in specified layers')
    
    plt.gca().grid(True)
    
def subplot_input_output(target, res, channels) :
    # Check if channels key's are colors
    overlap = {name for name in channels if name in mcolors.CSS4_COLORS}
    use_channel_color = overlap == channels.keys()
    
    # Plot input currents for each channels
    columns_in = []
    columns_out = []
    colors_in = []

    for key in channels :
        columns_in.append('I{0}-Iout-'.format(channels[key]) + key)
        columns_out.append('O{0}-Iout-'.format(channels[key]) + key)
        colors_in.append(key)
        
    print(colors_in)
        
    if use_channel_color :
        res.plot(subplots=False, ax=target, legend=False, sharex=False, sharey=False,
                 x = 'Time', y = columns_in, color = colors_in,
                 xlabel='Time (ns)', ylabel='Input current (A)')
        ax2 = res.plot(subplots=False, ax=target, legend=False, sharex=False, sharey=False,
                       secondary_y=True,
                       x = 'Time', y = columns_out, color = colors_in, style = '--')
        
        # The plot wrapper actually spits out a new axis that we can use  
        ax2.set_ylabel('Output current (A)')
          
    else :
        res.plot(subplots=False, ax=target, legend=False, sharex=False, sharey=False,
                 x = 'Time', y = columns_in,
                 xlabel='Time (ns)', ylabel='Input current (A)')
        ax2 = res.plot(subplots=False, ax=target, legend=False, sharex=False, sharey=False,
                       secondary_y=True,
                       x = 'Time', y = columns_out, style = '--')
        
        # The plot wrapper actually spits out a new axis that we can use  
        ax2.set_ylabel('Output current (A)')
        
def subplot_node(target, res, node, plot_all=False) :
    # Add a dash to find the correct node
    node += '-'
    # Get all the labels for that node
    columns = [name for name in res.columns if node in name]

    if not plot_all :
        # Choose voltages and Iinh, Iexc, Pout, 6 columns
        columns = columns[0:5] + [columns[7]]


    # Plot voltages
    res.plot(subplots=False, ax=target, legend=True, sharex=True, sharey=False,
             x = 'Time', y = columns[:3], ylabel='Voltages (V)')
    
    # Plot currents
    ax2 = res.plot(subplots=False, ax=target, legend=True, sharex=True, sharey=False,
                   secondary_y=True, xlabel='Time (ns)',
                   x = 'Time', y = columns[3:], style = '--')
    
    
    
    # The plot wrapper actually spits out a new axis that we can use  
    ax2.set_ylabel('Currents (nA)')
    
    # mpl_axes_aligner is an optinal package
    try :
        from mpl_axes_aligner import align 
        align.yaxes(target,0,ax2,0)
    except:
        pass
    
    
    target.set_title('Node '+node[:-1]) # Now we exclude the dash again
    
def subplot_attr(target, res, attr) :
    """
    Generates a subplot for a specific attribute for all nodes

    Parameters
    ----------
    target : Axis object
        Target axis to plot onto.
    res : pandas DataFram
        Result data from dynamic simulation.
    attr : str
        Attribute matching a column in the DataFrame.

    Returns
    -------
    None.

    """
    # Get all the labels for that attribute
    columns = [name for name in res.columns if attr in name]
    short_names = [name[:2] for name in columns]
    
    if columns[0][3] == 'V' :
        ylabel = 'Voltage (V)' 
    else:
        ylabel = 'Current (A)'
    
    # Plot the attribute for all nodes
    res.plot(subplots=False, ax=target, legend=True, sharex=False, sharey=False,
             x = 'Time', y = columns, 
             xlabel='Time (ns)', ylabel=ylabel, label=short_names)
    
    target.set_title(attr)

def subplot_chain(target, res, nodes, data_label) :
    columns = [name for node in nodes for name in res.columns if (node in name and data_label in name)]
    short_names = [name[:2] for name in columns]
    
    res.plot(subplots=False, ax=target, legend=short_names, sharex=False, sharey=False,
             x = 'Time', y = columns, label=short_names,
             xlabel='Time (ns)', ylabel='Current (nA)')
    
def listToString(s) :
    str1 = "" 
    for ele in s:
        str1 += '-'
        str1 += ele 
    # return string  
    return str1
    
#TODO: Seems not to be working in Tutorial.py for example
def plot_chainlist(res, G, source, target, search_string='Pout', doublewidth=True) :
    paths = list(simple_paths(G, source, target))
    Npaths = len(paths)
    Nrows = Npaths // 3 + 1 # choose three in a row as max
    Ncols = min(3,Npaths)
    if doublewidth : 
        nature_width = nature_double 
    else :  
        nature_width = nature_single
        
    fig, axs = plt.subplots(Nrows, Ncols, 
                            figsize=(nature_width*Ncols, nature_single*Nrows))
    
    if Npaths > 1 :
        for k, ax in enumerate(axs) :
            subplot_chain(ax, res, paths[k], search_string)
            ax.set_title(search_string+'-chain'+listToString(paths[k]))
    else :
        subplot_chain(axs, res, paths[0], search_string)
        axs.set_title(search_string+'-chain'+listToString(paths[0]))
        
    plt.subplots_adjust(left=0.124, right=0.9, bottom=0.1, top=0.9, wspace=0.1)
    plt.tight_layout()
    
def plot_nodes(res, nodes, plot_all=False, onecolumn=False, doublewidth=True,
               time_interval=None) :
    
    N = len(nodes)
    Nrows = max( N // 3 + int(bool(N % 3)), 1 )  # choose three in a row as max
    Ncols = min(3,N)
    
    if onecolumn : Nrows = N ; Ncols = 1
    if doublewidth : 
        nature_width = nature_double 
    else :  
        nature_width = nature_single
        
    # Select the approperiate time interval
    if time_interval is not None :
        select_res = res[(res["Time"]>=time_interval[0]) & (res["Time"]<=time_interval[1])]
    else : 
        select_res = res
        
    fig, axs = plt.subplots(Nrows, Ncols, 
                            figsize=(nature_width*Ncols, nature_single*Nrows))
    
    if N > 1 :
        for k, ax in enumerate(axs.flatten()) :
            if k < N :
                subplot_node(ax, select_res, nodes[k], plot_all)
    else:
        subplot_node(axs, select_res, nodes[0], plot_all)
        
    plt.subplots_adjust(left=0.124, right=0.9, bottom=0.1, top=0.9, wspace=0.1)
    plt.tight_layout()
        

def plot_attributes(res, attr, onecolumn=False, doublewidth=True) :
    """
    Plots a set of chosen attributes for all nodes
    
    Parameters
    ----------
    res : pandas DataFrame
        Result data from dynamic simulation.
    attr : list
        List of attributes to be plotted, each generates a subplot

    Returns
    -------
    None.

    """
    
    N = len(attr)
    # An ugly discrete function but it does its job
    Nrows = max( N // 3 + int(bool(N % 3)), 1 )  # choose three in a row as max
    Ncols = min(3,N)
    
    if onecolumn : Nrows = N ; Ncols = 1
    if doublewidth : 
        nature_width = nature_double 
    else :  
        nature_width = nature_single
        
    fig, axs = plt.subplots(Nrows, Ncols, 
                            figsize=(nature_width*Ncols, nature_single*Nrows))
    
    # Check case when only one attribute is wanted
    if N > 1 :          
        for k, ax in enumerate(axs.flatten()) : # flatten in case of 2D array
            # Generate subplot through this specific function
            subplot_attr(ax, res, attr[k])
    else :
        subplot_attr(axs, res, attr[0])
    
    plt.subplots_adjust(left=0.124, right=0.9, bottom=0.1, top=0.9, wspace=0.1)
    plt.tight_layout()
    
def plot_devices(devices, Vleaks, noise=None, **kwargs) :
    """Function to show many devices. Could be extended with rows for added noise"""
    # Generate the figure    
    Nrows = 1  
    Ncols = len(devices)
    fig, axs = plt.subplots(Nrows, Ncols, 
                            figsize=(nature_single*Ncols, nature_single*Nrows))
     
    for k, key in enumerate(devices) : # flatten in case of 2D array
        # For each device we ask for the transistor example IV
        df_iv = devices[key].transistorIV_example(**kwargs)
        df_iv.plot(subplots=False, ax=axs[k], sharex=False, sharey=False,
                x = 'Vgate', y = 'Current', xlabel='Vgate (V)', ylabel='Current (uA)',
                title=key)
        axs[k].set_ylim(0,10)
    
    plt.subplots_adjust(left=0.124, right=0.9, bottom=0.1, top=0.9, wspace=0.1)
    plt.tight_layout()
    
def visualize_transistor(IV_examples,labels=None):
    """
    Visualizes the transistor function

    Parameters
    ----------
    handle : function
        Function handle of the transistor IV from the physics module.
    Vgate : numpy array
        Values of Vgate to be included in the plots.

    Returns
    -------
    None.

    """
    # Generate the pandas DataFrame    
    Nrows = 1  
    Ncols = 2
    fig, axs = plt.subplots(Nrows, Ncols, 
                            figsize=(nature_single*Ncols, nature_single*Nrows))
    
    yscale = ['linear','log']
    
    for k, ax in enumerate(axs) :
        if type(IV_examples) is list :
            for m in range(len(IV_examples)) :
                if labels is not None:
                    _label = labels[m]+yscale[k]
                else :
                    _label = yscale[k]
                    
                IV_examples[m].plot(subplots=False, ax=ax, sharex=False, sharey=False,
                                    x = 'Vgate', y = 'Current',label=_label,
                                    xlabel='Vgate (V)', ylabel='Current (uA)')
        else :
            if labels is not None:
                _label = labels+yscale[k]
            else :
                _label = yscale[k]
                
            IV_examples.plot(subplots=False, ax=ax, sharex=False, sharey=False,
                             x = 'Vgate', y = 'Current',label=_label,
                             xlabel='Vgate (V)', ylabel='Current (uA)')
        if k == 0 :
            ax.set_ylim(0,10)
        ax.set_yscale(yscale[k])
        ax.grid('True')
    
    #plt.legend()    
    plt.subplots_adjust(left=0.124, right=0.9, bottom=0.1, top=0.9, wspace=0.1)
    plt.tight_layout()
    
def visualize_LED_efficiency(example) :
    """
    Visualizes the internal quantum efficiency used by the model

    Parameters
    ----------
    handle : function
        Function that takes I as an input and generates an internal quantum 
        efficiency eta. 

    Returns
    -------
    None.

    """
    
    fig, ax = plt.subplots(figsize=(nature_single, nature_single))
    
    example.plot(subplots=False, ax=ax, sharex=False, sharey=False,
                 x = 'Current (uA)', y = 'eta, IQE')
                 #xlabel='Current (uA)', ylabel='Current (uA)')
                 
    #ax.set_ylim(0,10)
    ax.set_xscale('log')    
    ax.grid('True')
    plt.tight_layout()
               
def save_plot(fig, filename, plot_path=Path('plots/')):
    # Check for folder, otherwise create
    if not plot_path.is_dir() :
        plot_path.mkdir(parents=True)
    # Save the figure 
    fig.savefig(plot_path / (filename + '.png'),
                bbox_inches='tight', dpi=300)           