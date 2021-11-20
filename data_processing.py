import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from DPC import DPClustering

g = nx.read_edgelist('data/facebook_combined.txt',
                     create_using=nx.Graph(), nodetype=int)

with open('data/egos.txt', 'r') as f:
    egos = [int(_) for _ in f.readline().split(' ')]

fig, ax = plt.subplots(1, figsize=(15, 15))

node_color = [10000 if v in egos else 1 for v in g]
node_size = [g.degree(v) for v in g]

pos = nx.spring_layout(g)
ec = nx.draw_networkx_edges(g, pos, alpha=0.2, width=1, ax=ax)
nc = nx.draw_networkx_nodes(g, pos,
                            nodelist=g.nodes,
                            node_color=node_color,
                            node_size=node_size,
                            cmap=plt.cm.jet,
                            alpha=0.8,
                            ax=ax)
fig.show()


G = nx.karate_club_graph()
fig, ax = plt.subplots(1, figsize=(15, 15))
node_color = [1 if G.nodes[v]['club'] == 'Mr. Hi' else 0 for v in G]
pos = nx.spring_layout(G)
ec = nx.draw_networkx_edges(G, pos, alpha=0.2, width=1, ax=ax)
nc = nx.draw_networkx_nodes(G, pos,
                            nodelist=G.nodes,
                            node_color=node_color,
                            node_size=100,
                            cmap=plt.cm.jet,
                            alpha=0.8,
                            ax=ax)
fig.show()





with open('fb_data/0.circles', 'r') as f:
    circles = [_[:-1].split('\t')[1:] for _ in f.readlines()]

# degrees = np.array(list(dict(g.degree).values()))
g.add_node(0)
for n in g.nodes():
    if n != 0:
        g.add_edge(0, n)
    g.nodes[n]['circle'] = -1


fig, ax = plt.subplots(1, figsize=(15, 15))

nodes = g.nodes()


for i, c in enumerate(circles):
    for node in c:
        if int(node) in g.nodes:
            g.nodes[int(node)]['circle'] = i

pos = nx.spring_layout(g)
colors = [g.nodes[n]['circle'] for n in g.nodes]
ec = nx.draw_networkx_edges(g, pos, alpha=0.2, width=1, ax=ax)
nc = nx.draw_networkx_nodes(g, pos, nodelist=nodes, node_color=colors, node_size=100, cmap=plt.cm.jet, ax=ax)
# nx.draw(g, pos=pos)
plt.show()

# with open('data/articles.tsv') as file:
#     tsv_file = csv.reader(file, delimiter="\t")
#     for line in tsv_file:
#         print(line)


betCent = nx.betweenness_centrality(G1, normalized=True, endpoints=True)
node_color = [20000.0 * G1.degree(v) for v in G1]
node_size = [v * 10000 for v in betCent.values()]
plt.figure(figsize=(20, 20))
nx.draw_networkx(G1, pos=pos, with_labels=False,
                 node_color=node_color,
                 node_size=node_size)
plt.axis('off')
