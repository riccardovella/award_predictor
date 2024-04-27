import os

import pandas
import numpy

import networkx as nx
from networkx.algorithms import approximation


def avg_out_degree(G):
    degrees = G.out_degree()
    return sum([d for _, d in degrees])/float(len(G))

def diameter(G, approximate):
    if approximate:
        return approximation.diameter(G.to_undirected())
    return nx.diameter(G.to_undirected())

def avg_path_length(G):
    return nx.average_shortest_path_length(G)

def density(G):
    return nx.density(G)

def global_clustering_coeff(G):
    return nx.transitivity(G)

def avg_local_clustering_coeff(G):
    return nx.average_clustering(G)

def get_graph_measures(G, approximate):
    return [
        avg_out_degree(G),
        diameter(G, approximate),
        density(G),
        global_clustering_coeff(G),
        avg_local_clustering_coeff(G)
    ]

