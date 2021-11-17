import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

G1 = nx.read_edgelist('fb_data/facebook_combined.txt',
                      create_using=nx.Graph(), nodetype=int)

degrees = np.array(list(dict(G1.degree).values()))

pos = nx.spring_layout(G1, k=0.1, weight=1)
nx.draw(G1, pos=pos, alpha=0.7)
plt.show()
# with open('data/articles.tsv') as file:
#     tsv_file = csv.reader(file, delimiter="\t")
#     for line in tsv_file:
#         print(line)
pos = nx.spring_layout(G1)
betCent = nx.betweenness_centrality(G1, normalized=True, endpoints=True)
node_color = [20000.0 * G1.degree(v) for v in G1]
node_size = [v * 10000 for v in betCent.values()]
plt.figure(figsize=(20, 20))
nx.draw_networkx(G1, pos=pos, with_labels=False,
                 node_color=node_color,
                 node_size=node_size)
plt.axis('off')
