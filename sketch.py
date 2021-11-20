
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from DPC import DPClustering, pairwise_distance

# G = nx.karate_club_graph()
# labels = dict(zip(G.nodes, G.nodes))
# node_color = [0 if G.nodes[v]['club'] == 'Mr. Hi' else 1 for v in G]

"""
email network
"""

with open('data/email_edges.txt', 'rb') as f:
    G = nx.read_adjlist(f, create_using=nx.DiGraph())
G.remove_edges_from(nx.selfloop_edges(G))
G.remove_nodes_from(list(nx.isolates(G)))


labels = {}
with open('data/email_labels.txt') as f:
    for l in f.readlines():
        k, v = l.split(' ')
        labels[k] = v[:-1]

# set weight by communities for plots only
for e in G.edges:
    G.edges[e]['weight'] = 0.1 if labels[e[0]] == labels[e[1]] else 0.001


labels_list = [int(_) for _ in set(labels.values())]
labels_list.sort()

# layout_wt = [G.edges[_]['weight'] for _ in G.edges]
pos = nx.spring_layout(G)
node_color = [int(labels[v]) for v in G]


def distF(i, j):
    """
    hop distances
    """
    return nx.shortest_path_length(G, i, j)


fig, ax = plt.subplots(2, figsize=(15, 15))
ec = nx.draw_networkx_edges(G, pos, alpha=0.2, width=1, ax=ax[0])
nc = nx.draw_networkx_nodes(G, pos,
                            nodelist=G.nodes,
                            node_color=node_color,
                            node_size=10,
                            cmap=plt.cm.jet,
                            alpha=0.8,
                            ax=ax[0])
#nx.draw_networkx_labels(G, pos, dict(zip(G.nodes, G.nodes)), font_color="whitesmoke", ax=ax[0])
ax[0].set_title('Truth')
fig.show()



# TODO ?? color same ->>> sort the order of the nodes!!!!
"""
DP clustering
"""
max_id = 33
in_data = np.array([pos[i] for i in range(max_id + 1)])
tags = node_color

DP = DPClustering(in_data, tags, pairwise_distance(max_id, distF))
DP.run(2)

node_color = [0 if G.nodes[v]['club'] == 'Mr. Hi' else 1 for v in G]
nx.draw_networkx_edges(G, pos, alpha=0.2, width=1, ax=ax[1])
nx.draw_networkx_nodes(G, pos,
                       nodelist=G.nodes,
                       node_color=DP.pred_tags,
                       node_size=400,
                       cmap=plt.cm.jet,
                       alpha=0.8,
                       ax=ax[1])
nx.draw_networkx_labels(G, pos, dict(zip(G.nodes, G.nodes)), font_color="whitesmoke", ax=ax[1])
ax[1].set_title('Prediction')
fig.show()


#
#
# def large_cluster(number_clusters=10):
#     with open('data/email_edges.txt', 'rb') as f:
#         G = nx.read_adjlist(f)
#     G.remove_edges_from(nx.selfloop_edges(G))
#     G.remove_nodes_from(list(nx.isolates(G)))
#
#     str_nodes = list(G.nodes)
#     str_nodes.sort(key=lambda x: int(x))
#     str_nodes_int = {k: v for v, k in enumerate(str_nodes)}
#     G = nx.relabel_nodes(G, str_nodes_int)
#
#     labels = {}
#     with open('data/email_labels.txt') as f:
#         for l in f.readlines():
#                 k, v = l.split(' ')
#                 if k in str_nodes_int:
#                     labels[str_nodes_int[k]] = int(v[:-1])
#
#     for e in G.edges:
#         G.edges[e]['weight'] = 0.1 if labels[e[0]] == labels[e[1]] else 0.001
#
#     pos = nx.spring_layout(G)
#     tags = [labels[v] for v in G]
#
#     def distF(i, j):
#         return nx.shortest_path_length(G, i, j)
#
#     # Density Peak Clustering
#     max_id = len(G.nodes) - 1
#     in_data = np.array([pos[i] for i in range(max_id + 1)])
#     DP = DPClustering(in_data, tags, pairwise_distance(max_id, distF, directed=True))
#     DP.run(number_clusters)
#     run_fig, run_ax = DP.plot(show=False)
#     run_fig.savefig('results/email_run.png')
#
#     fig, _ = cluster_plot(G, pos, tags, DP.pred_tags, node_size=10, label=False)
#     fig.savefig('results/email_large.png')
