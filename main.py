import json
import random
import numpy as np
import scipy
import seaborn as sns
import networkx as nx
import matplotlib.pyplot as plt
from DPC import DPClustering, pairwise_distance


def shortest_path_len(G, i, j):
    return nx.shortest_path_length(G, i, j)


def add_edges(ax, G, pos, **kwargs):
    nx.draw_networkx_edges(G, pos,
                           alpha=kwargs.get('alpha', 0.1),
                           width=kwargs.get('width', 1),
                           ax=ax[0, 0])
    nx.draw_networkx_edges(G, pos,
                           alpha=kwargs.get('alpha', 0.1),
                           width=kwargs.get('width', 1),
                           ax=ax[1, 0])


def cluster_plot(G, pos, true_tags, pred_tags, node_size=400, edge_width=1, label=True, show=True, desc='', **kwargs):
    # large plots
    fig, ax = plt.subplots(1, 2, figsize=(30, 15))
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
    ax[1].set_title(f'Prediction {desc}')
    fig.savefig('results/Karate_large.png')
    if show:
        fig.show()
    return fig, ax


def SpectralDP(G, pos, tags, N_communities, desc='Karate', **kwargs):

    laplacian = scipy.sparse.csr_matrix.todense(nx.normalized_laplacian_matrix(G, weight=1))
    u, s, v = scipy.linalg.svd(laplacian)
    # fig, ax = plt.subplots(figsize=(15, 6))
    # sns.scatterplot(x=np.arange(len(s)), y=s[::-1], ax=ax)
    # ax.set_title('Graph Spectrum')
    # fig.show()

    u = u[:, -N_communities:-1]  # truncated
    delta_dist = pairwise_distance(len(tags) - 1, lambda x, y: np.linalg.norm(u[x] - u[y]))

    # hop_dist = pairwise_distance(len(tags) - 1, lambda x, y: shortest_path_len(G, x, y))

    in_data = np.array([pos[i] for i in range(len(tags))])
    DP = DPClustering(in_data, tags)
    DP.run(rho_dist=delta_dist, delta_dist=delta_dist, max_clusters=N_communities, **kwargs)
    # run_fig.savefig(f'results/{desc}.png')
    #
    entropy, purity = DP.eval()
    fig, _ = cluster_plot(G, pos, tags, [DP.align[_] if _ != -1 else -1 for _ in DP.pred_tags], desc=f'DP: entropy={entropy:.3f}, purity={purity:.3f}', **kwargs)
    fig.savefig(f'results/{desc}_large.png')

    return DP, u


def paper_cluster():
    """
    data: can use gamma=True for hint
    'spiral', 8, 5
    'jain', 11, 10
    'flame', 4, 6
    'aggregation', 12, 6
    """
    data, rho_threshold, delta_threshold = 'spiral', 10, 5
    # data, rho_threshold, delta_threshold = 'jain', 13, 10
    # data, rho_threshold, delta_threshold = 'flame', 9, 6
    # data, rho_threshold, delta_threshold = 'aggregation', 16, 5

    pos = np.loadtxt(f'data/{data}.txt')
    tags = [int(_)-1 for _ in pos[:, 2]]
    pos = pos[:, :2]
    # dist_dict = {(int(i)-1, int(j)-1): d for i, j, d in np.loadtxt('data/spiral_distance.txt')}
    dist_dict = pairwise_distance(len(pos) - 1, lambda x, y: np.linalg.norm(pos[x] - pos[y]))

    N_communities = len(set(tags))

    dp = DPClustering(pos, tags)
    dp.run(rho_dist=dist_dict,
           delta_dist=dist_dict,
           max_clusters=N_communities,
           rho_threshold=rho_threshold,
           delta_threshold=delta_threshold,
           gamma=True,
           gauss=True)
    entropy, purity = dp.eval()
    print(f'DP: entropy={entropy:.3f}, purity={purity:.3f}')

    dp.plot(show=True, threshold=True, desc=f'DP: entropy={entropy:.3f}, purity={purity:.3f}')

    # compare to k-means
    dp.k_means(pos, N_communities)
    entropy, purity = dp.eval()
    dp.plot(show=True, threshold=False, desc=f'K-means: entropy={entropy:.3f}, purity={purity:.3f}')


def rand_cluster():
    G = nx.random_partition_graph([5, 10, 15, 15, 15], 0.7, 0.1, seed=0)
    # G = nx.planted_partition_graph(5, 10, 0.5, 0.1, seed=0)
    tags = [G.nodes[v]['block'] for v in G]

    # for plots only
    for e in G.edges:
        G.edges[e]['weight'] = 1 if tags[e[0]] == tags[e[1]] else 0.4

    N_communities = len(set(tags))
    pos = nx.spring_layout(G)

    # Spectral Density Peak Clustering
    dp, embed = SpectralDP(G, pos, tags, N_communities, desc='rand',
                           gamma=False, rho_threshold=1, delta_threshold=0.2)
    print(dp.eval())

    entropy, purity = dp.eval()
    fig, ax = dp.plot(show=False, threshold=False, desc=f'DP: entropy={entropy:.3f}, purity={purity:.3f}')
    add_edges(ax, G, pos)
    fig.show()
    fig.savefig(f'results/rand_dp.png')

    # compare to k-means
    dp.k_means(embed, N_communities)
    entropy, purity = dp.eval()
    fig, ax = dp.plot(show=False, threshold=False, desc=f'K-means: entropy={entropy:.3f}, purity={purity:.3f}')
    add_edges(ax, G, pos)
    fig.show()
    fig.savefig(f'results/rand_k_means.png')

    # compare to k-means
    # dp.k_means(embed, N_communities)
    # entropy, purity = dp.eval()
    # fig, _ = cluster_plot(G, pos, tags, [dp.align[_] for _ in dp.pred_tags], desc=f'K-means: entropy={entropy:.3f}, purity={purity:.3f}')
    # fig.savefig(f'results/rand_k_means.png')


def karate_cluster():
    """
    Karate Club network (2 clusters)
    """
    G = nx.karate_club_graph()
    tags = [0 if G.nodes[v]['club'] == 'Mr. Hi' else 1 for v in G]

    N_communities = len(set(tags))
    pos = nx.spring_layout(G)

    # Spectral Density Peak Clustering
    dp, embed = SpectralDP(G, pos, tags, N_communities, desc='karate',
                           gamma=True, rho_threshold=5, delta_threshold=0.15)
    print(dp.eval())

    entropy, purity = dp.eval()
    fig, ax = dp.plot(show=False, threshold=False, desc=f'DP: entropy={entropy:.3f}, purity={purity:.3f}')
    add_edges(ax, G, pos)
    fig.show()
    fig.savefig(f'results/karate_dp.png')

    # compare to k-means
    dp.k_means(embed, N_communities)
    entropy, purity = dp.eval()
    fig, ax = dp.plot(show=False, threshold=False, desc=f'K-means: entropy={entropy:.3f}, purity={purity:.3f}')
    add_edges(ax, G, pos)
    fig.show()
    fig.savefig(f'results/karate_k_means.png')

    # compare to k-means
    dp.k_means(embed, N_communities)
    entropy, purity = dp.eval()
    fig, _ = cluster_plot(G, pos, tags, [dp.align[_] for _ in dp.pred_tags], desc=f'K-means: entropy={entropy:.3f}, purity={purity:.3f}')
    fig.savefig(f'results/karate_k_means.png')


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
        G.edges[e]['weight'] = 0.05 if communities[e[0]] == communities[e[1]] else 0.001

    pos = nx.spring_layout(G)
    tags = [communities[v] for v in G]

    # Spectral Density Peak Clustering
    dp, embed = SpectralDP(G, pos, tags, N_communities, desc='Email', node_size=10, label=False, plot_edge=False,
                           gamma=True,  percent=1., gauss=True, cut_gauss=True)
    print(dp.eval())

    entropy, purity = dp.eval()
    fig, ax = dp.plot(show=False, threshold=False, desc=f'DP: entropy={entropy:.3f}, purity={purity:.3f}')
    fig.show()
    fig.savefig(f'results/email_dp.png')

    # compare to k-means
    dp.k_means(embed, N_communities)
    entropy, purity = dp.eval()
    fig, ax = dp.plot(show=False, threshold=False, desc=f'K-means: entropy={entropy:.3f}, purity={purity:.3f}')
    fig.show()
    fig.savefig(f'results/email_k_means.png')

    # compare to k-means
    dp.k_means(embed, N_communities)
    entropy, purity = dp.eval()
    fig, _ = cluster_plot(G, pos, tags, [dp.align[_] for _ in dp.pred_tags],
                          desc=f'K-means: entropy={entropy:.3f}, purity={purity:.3f}', node_size=10, label=False)
    fig.savefig(f'results/email_k_means.png')


def facebook_cluster():
    """
    too large...
    :return:
    """
    with open('data/facebook_combined.txt', 'rb') as f:
        G = nx.read_adjlist(f)

    # relabel
    str_nodes = list(G.nodes)
    str_nodes.sort(key=lambda x: int(x))
    str_nodes_int = {k: v for v, k in enumerate(str_nodes)}
    G = nx.relabel_nodes(G, str_nodes_int)

    with open('data/fb_tags.txt', 'r') as f:
        content = f.read()
        communities = eval(content)

    N_communities = len(set(communities.values()))

    # for plots only
    for e in G.edges:
        G.edges[e]['weight'] = 0.05 if communities[e[0]] == communities[e[1]] else 0.001

    pos = nx.spring_layout(G)
    tags = [communities[v] for v in G]

    # Spectral Density Peak Clustering
    dp, embed = SpectralDP(G, pos, tags, N_communities, desc='facebook', node_size=10, label=False, plot_edge=False,
                           gamma=True, percent=2., gauss=True, cut_gauss=True)
    print(dp.eval())

    entropy, purity = dp.eval()
    fig, ax = dp.plot(show=False, threshold=False, desc=f'DP: entropy={entropy:.3f}, purity={purity:.3f}')
    fig.show()
    fig.savefig(f'results/facebook_dp.png')

    # compare to k-means
    dp.k_means(embed, N_communities)
    entropy, purity = dp.eval()
    fig, ax = dp.plot(show=False, threshold=False, desc=f'K-means: entropy={entropy:.3f}, purity={purity:.3f}')
    fig.show()
    fig.savefig(f'results/facebook_k_means.png')

    # compare to k-means
    dp.k_means(embed, N_communities)
    entropy, purity = dp.eval()
    fig, _ = cluster_plot(G, pos, tags, [dp.align[_] for _ in dp.pred_tags],
                          desc=f'K-means: entropy={entropy:.3f}, purity={purity:.3f}', node_size=10, label=False)
    fig.savefig(f'results/fb_k_means.png')
