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
from mpl_axes_aligner import align

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
    
def visualize_network(layers, weights, exclude_nodes={}, node_size=600, layout='multipartite', show_edge_labels=True, shell_order=None, savefig=False) :
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

    val_map = {'I': 0.5,
               'H': 0.6,
               'O': 0.7}
    
    values = [val_map.get(node[0], 0.45) for node in G.nodes()]
    edge_labels=dict([((u,v,),f"{d['weight']:.1f}")
                     for u,v,d in G.edges(data=True)])
    edge_colors=[d['color'] for u,v,d in G.edges(data=True)]
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
    
    nx.draw_networkx_nodes(G, pos, cmap=plt.get_cmap('Blues'), 
                           node_color = values, vmin=0., vmax=1.0,
                           node_size = node_size)
    nx.draw_networkx_labels(G, pos, font_size=(6+c*6))
    nx.draw_networkx_edges(G, pos, edgelist=red_edges, edge_color='r', 
                           arrows=True, arrowsize=20,node_size=node_size)
    nx.draw_networkx_edges(G, pos, edgelist=black_edges, edge_color=edge_colors,
                           arrows=True, arrowsize=20,node_size=node_size,
                           width=edge_weights,
                           connectionstyle='arc3,rad=.2')

                          # connectionstyle='arc3,rad=0.2')
    if show_edge_labels :
        nx.draw_networkx_edge_labels(G,pos,edge_labels=edge_labels)
    
    # There is an interface to graphviz .dot files provided by 
    #nx.drawing.nx_pydot.write_dot(G, 'graph.dot')
    # to generate a png, run dot -Tpng graph.dot > graph.png
    if savefig :
        nx.drawing.nx_pydot.write_dot(G, 'network_layout.dot')
        plt.savefig('network_layout.png',dpi=300)
    
    return G

def simple_paths(G,source, target) :
    paths = nx.all_simple_paths(G,source,target)
    return paths

def save_movie_frame(t, layers, weights, node_size=600) :
    pass

def visualize_dynamic_result(res, columns) :
    
    # Pretty generic plot function
    res.plot(x = 'Time', y = columns,
             xlabel='Time (ns)', ylabel='Voltage/Current (V)')
        
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
    # Get all the labels for that node
    columns = [name for name in res.columns if node in name]

    if not plot_all :
        # Choose voltages and Iexc, Pout, 5 columns
        columns = columns[0:3] + [columns[4]] + [columns[7]]


    # Plot voltages
    res.plot(subplots=False, ax=target, legend=True, sharex=True, sharey=False,
             x = 'Time', y = columns[:3], ylabel='Voltages (V)')
    
    # Plot currents
    ax2 = res.plot(subplots=False, ax=target, legend=True, sharex=True, sharey=False,
                   secondary_y=True, xlabel='Time (ns)',
                   x = 'Time', y = columns[3:], style = '--')
    
    
    
    # The plot wrapper actually spits out a new axis that we can use  
    ax2.set_ylabel('Currents (nA)')
    
    align.yaxes(target,0,ax2,0,0.5)
    
    target.set_title('Node '+node)
    
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
    
def plot_chainlist(res, G, source, target, doublewidth=True) :
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
            subplot_chain(ax, res, paths[k], 'Iout')
            ax.set_title('Iout-chain'+listToString(paths[k]))
    else :
        subplot_chain(axs, res, paths[0], 'Iout')
        axs.set_title('Iout-chain'+listToString(paths[0]))
        
    plt.subplots_adjust(left=0.124, right=0.9, bottom=0.1, top=0.9, wspace=0.1)
    plt.tight_layout()
    
def plot_nodes(res, nodes, plot_all=False, onecolumn=False, doublewidth=True) :
    
    N = len(nodes)
    Nrows = max( N // 3 + int(bool(N % 3)), 1 )  # choose three in a row as max
    Ncols = min(3,N)
    
    if onecolumn : Nrows = N ; Ncols = 1
    if doublewidth : 
        nature_width = nature_double 
    else :  
        nature_width = nature_single
        
    fig, axs = plt.subplots(Nrows, Ncols, 
                            figsize=(nature_width*Ncols, nature_single*Nrows))
    
    if N > 1 :
        for k, ax in enumerate(axs.flatten()) :
            if k < N :
                subplot_node(ax, res, nodes[k], plot_all)
    else:
        subplot_node(axs, res, nodes[0], plot_all)
        
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
                          