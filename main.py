import numpy as np
import random
import scipy
import seaborn as sns
import networkx as nx
import matplotlib.pyplot as plt
from DPC import DPClustering, pairwise_distance

"""
Karate Club network (2 clusters)
"""


def cluster_plot(G, pos, true_tags, pred_tags, node_size=400, edge_width=1, label=True, show=True, **kwargs):
    # large plots
    fig, ax = plt.subplots(2, figsize=(15, 30))
    nx.draw_networkx_edges(G, pos, alpha=0.2, width=edge_width, ax=ax[0])
    nx.draw_networkx_nodes(G, pos,
                           nodelist=G.nodes,
                           node_color=true_tags,
                           node_size=node_size,
                           cmap=plt.cm.jet,
                           alpha=0.8,
                           ax=ax[0])
    if label:
        nx.draw_networkx_labels(G, pos, dict(zip(G.nodes, G.nodes)), font_color="whitesmoke", ax=ax[0])
    ax[0].set_title('Truth')
    nx.draw_networkx_edges(G, pos, alpha=0.2, width=edge_width, ax=ax[1])
    nx.draw_networkx_nodes(G, pos,
                           nodelist=G.nodes,
                           node_color=pred_tags,
                           node_size=node_size,
                           cmap=plt.cm.jet,
                           alpha=0.8,
                           ax=ax[1])
    if label:
        nx.draw_networkx_labels(G, pos, dict(zip(G.nodes, G.nodes)), font_color="whitesmoke", ax=ax[1])
    ax[1].set_title('Prediction')
    fig.savefig('results/Karate_large.png')
    if show:
        fig.show()
    return fig, ax


def SpectralDP(G, pos, tags, N_communities, desc='Karate', **kwargs):
    laplacian = scipy.sparse.csr_matrix.todense(nx.laplacian_matrix(G, weight=1))
    u, s, v = scipy.linalg.svd(laplacian)
    fig, ax = plt.subplots(figsize=(20, 10))
    sns.scatterplot(x=np.arange(len(s)), y=s[::-1], ax=ax)
    ax.set_title('Graph Spectrum')
    fig.show()

    u = u[:, -N_communities:-1]  # truncated
    delta_dist = pairwise_distance(len(tags) - 1, lambda x, y: np.linalg.norm(u[x] - u[y]))

    in_data = np.array([pos[i] for i in range(len(tags))])
    DP = DPClustering(in_data, tags)
    DP.run(rho_dist=delta_dist,
           delta_dist=delta_dist,
           assign_dist=delta_dist,
           max_clusters=N_communities,
           rho_threshold=5,
           delta_threshold=1,
           gamma=True,
           assign_gap=1)

    DP.align_prediction()

    run_fig, run_ax = DP.plot(show=False)
    if kwargs.get('plot_edge', True):
        nx.draw_networkx_edges(G, pos,
                               alpha=kwargs.get('alpha', 0.2),
                               width=kwargs.get('width', 1),
                               ax=run_ax[0, 0])
        nx.draw_networkx_edges(G, pos,
                               alpha=kwargs.get('alpha', 0.2),
                               width=kwargs.get('width', 1),
                               ax=run_ax[1, 0])
    run_fig.show()
    run_fig.savefig(f'results/{desc}_run.png')

    fig, _ = cluster_plot(G, pos, tags, DP.pred_tags, **kwargs)
    fig.savefig(f'results/{desc}_large.png')

    return DP


def karate_cluster():
    # G = nx.karate_club_graph()
    # tags = [0 if G.nodes[v]['club'] == 'Mr. Hi' else 1 for v in G]
    G = nx.random_partition_graph([20, 15, 10], 0.8, 0.1)
    tags = [G.nodes[v]['block'] for v in G]

    N_communities = len(set(tags))
    pos = nx.spring_layout(G)

    # Spectral Density Peak Clustering
    dp = SpectralDP(G, pos, tags, N_communities, desc='Karate')
    print(dp.eval())


def email_cluster():
    with open('data/email_edges.txt', 'rb') as f:
        G = nx.read_adjlist(f, create_using=nx.DiGraph())
    G = G.to_undirected(reciprocal=True)
    G.remove_edges_from(nx.selfloop_edges(G))
    G.remove_nodes_from(list(nx.isolates(G)))

    str_nodes = list(G.nodes)
    str_nodes.sort(key=lambda x: int(x))
    str_nodes_int = {k: v for v, k in enumerate(str_nodes)}
    G = nx.relabel_nodes(G, str_nodes_int)

    communities = {}
    with open('data/email_labels.txt') as f:
        c, tag_set = 0, {}
        for l in f.readlines():
            k, v = l.split(' ')
            if k in str_nodes_int:
                if v not in tag_set:
                    tag_set[v] = c
                    c += 1
                communities[str_nodes_int[k]] = tag_set[v]  # int(v[:-1])
    N_communities = len(set(communities.values()))

    # for plots only
    for e in G.edges:
        G.edges[e]['weight'] = 0.1 if communities[e[0]] == communities[e[1]] else 0.001

    pos = nx.spring_layout(G)
    tags = [communities[v] for v in G]

    # Spectral Density Peak Clustering
    dp = SpectralDP(G, pos, tags, N_communities, desc='Karate', node_size=10, label=False, plot_edge=False)
    print(dp.eval())


    # # delta_dist
    # max_id = len(tags) - 1
    # in_data = np.array([pos[i] for i in range(len(tags))])
    # import scipy
    # import seaborn as sns
    # laplacian = scipy.sparse.csr_matrix.todense(nx.laplacian_matrix(G, weight=1))
    # u, s, v = scipy.linalg.svd(laplacian)
    # fig, ax = plt.subplots(figsize=(20, 10))
    # sns.scatterplot(x=np.arange(len(s)), y=s[::-1], ax=ax)
    # ax.set_title('Graph Spectrum')
    # fig.show()
    # u = u[:, -N_communities:-1]  # truncated
    # delta_dist = pairwise_distance(max_id, lambda x, y: np.linalg.norm((u[x]-u[y])))
    #
    # # Density Peak Clustering
    # DP = DPClustering(in_data, tags)
    # DP.run(rho_dist=delta_dist,
    #        delta_dist=delta_dist,
    #        assign_dist=delta_dist,
    #        max_clusters=N_communities,
    #        rho_threshold=5,
    #        delta_threshold=1,
    #        gamma=True,
    #        assign_gap=1)
    # DP.align_prediction()
    #
    # run_fig, run_ax = DP.plot(show=False)
    # nx.draw_networkx_edges(G, pos, alpha=0.2, width=1, ax=run_ax[0, 0])
    # nx.draw_networkx_edges(G, pos, alpha=0.2, width=1, ax=run_ax[1, 0])
    # run_fig.savefig('results/email_run.png')
    # run_fig.show()
    #
    # fig, _ = cluster_plot(G, pos, tags, DP.pred_tags, node_size=10, label=False)
    # fig.savefig('results/email_large.png')


# def com_cluster():
#     """
#     too large...
#     :return:
#     """
#     with open('data/youtube5000.txt', 'rb') as f:
#         G = nx.read_adjlist(f)
#     G.remove_edges_from(nx.selfloop_edges(G))
#     G.remove_nodes_from(list(nx.isolates(G)))
#
#     # largest component
#     Gcc = sorted(nx.connected_components(G), key=len, reverse=True)
#     G = G.subgraph(Gcc[0])
#     str_nodes = list(G.nodes)
#     str_nodes.sort(key=lambda x: int(x))
#     str_nodes_int = {k: v for v, k in enumerate(str_nodes)}
#     G = nx.relabel_nodes(G, str_nodes_int)
#
#     communities = {}
#     with open('data/youtube5000.txt') as f:
#         for c, l in enumerate(f.readlines()):
#             for k in l[:-1].split('\t'):
#                 if k in str_nodes_int:
#                     communities[str_nodes_int[k]] = c
#
#     N_communities = len(set(communities.values()))
#     tags = [communities[v] for v in G]
#
#     def distF(i, j):
#         return nx.shortest_path_length(G, i, j)
#
#     # Density Peak Clustering
#     max_id = len(G.nodes) - 1
#     DP = DPClustering(np.zeros((2, 2)), tags)
#     dist_dict = pairwise_distance(max_id, distF)
#     DP.run(rho_threshold=100, delta_threshold=2,
#            rho_dist=dist_dict,
#            delta_dist=dist_dict,
#            assign_dist=dist_dict,
#            max_clusters=N_communities,
#            gamma=True)

# trans = np.array([[0, 1/2, 1/4, 0, 1/2],
#                   [1/3, 0, 1/4, 0, 0],
#                   [1/3, 1/2, 0, 1, 1/2],
#                   [0, 0, 1/4, 0, 0],
#                   [1/3, 0, 1/4, 0, 0]])
#
# s = np.array([0, 0, 0, 0, 1])
#
# for i in range(100):
#     s = trans.__matmul__(s)
