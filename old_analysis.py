import matplotlib.pyplot as plt
import networkx as nx
import scipy
import random
import numpy as np
import seaborn as sns
import pandas as pd
import json
from scipy import stats
from SL_card import cards_encode_tensor

DATA_PATH = '/Users/fanglinjiajie/locals/datasets/CardData/'

# game data
data = pd.read_csv(DATA_PATH + 'tables_num.csv', dtype=str)
players = list(set(data[['user_1', 'user_2', 'user_3']].values.reshape(-1)))
network = np.zeros((len(players), len(players)))

for p1, p2, p3, num in data.values:
    network[players.index(p1)][players.index(p2)] += int(num)
    network[players.index(p1)][players.index(p3)] += int(num)
    network[players.index(p2)][players.index(p3)] += int(num)
network += network.transpose()

# density = 0.0064 ,
dens = len(network[network.nonzero()]) / network.size
play_edges = []
for i in range(len(players)):
    for j in range(i + 1, len(players)):
        if network[i][j] != 0:
            play_edges += [(players[i], players[j], network[i][j])]

G = nx.Graph()
G.add_nodes_from(players)
G.add_weighted_edges_from(play_edges)
# adjacency matrix
adj_mat = nx.to_numpy_matrix(G, weight=1)
fig, axe = plt.subplots(figsize=(20, 20))
sns.heatmap(adj_mat, ax=axe,cbar=False, cmap=plt.cm.binary)
plt.title('Adjacency Matrix')
plt.show()

# find cliques
laplacian = scipy.sparse.csr_matrix.todense(nx.laplacian_matrix(G,weight=1))
u, s, v = scipy.linalg.svd(laplacian)

fig, axe = plt.subplots(figsize=(20, 10))
s.sort()
sns.scatterplot(np.arange(len(s)), s)
plt.title('Spectral Clustering')
plt.show()

# network plot
edges, weights = zip(*nx.get_edge_attributes(G, 'weight').items())
weights = np.array(weights)
# reweight
weights = weights/weights.max()
weights = np.sqrt(weights)
degrees = np.array(list(dict(G.degree).values()))
nx.write_graphml(G, 'networl.graphml')

fig, axe = plt.subplots(figsize=(10, 10))
# Fruchterman-Reingold force-directed algorithm
pos = nx.spring_layout(G, k=0.1, weight=1)
nx.draw(G, pos=pos, node_color='b', edgelist=edges, width=5*weights**1.5, node_size=5*degrees**2, ax=axe, alpha = 0.7)
       # edge_color=weights*256, edge_cmap=plt.cm.Greys)
fig.show()

# G = nx.gnp_random_graph(10, 0.3)
# for u, v, d in G.edges(data=True):
#     d['weight'] = random.random()
#
# edges, weights = zip(*nx.get_edge_attributes(G, 'weight').items())
# weights = ([10*i for i in weights])
# pos = nx.spring_layout(G)
# nx.draw(G, pos,node_color='b', edgelist=edges, edge_color=weights, width=weights, edge_cmap=plt.cm.Blues)
# #plt.savefig('network.png')
# plt.show()


# card sampling
sample = []
seed = list(range(48))

# vector sample
All_CARD_num = []
for i in range(13):
    All_CARD_num += [i + 1] * 4
All_CARD_num.remove(12)
All_CARD_num.remove(13)
All_CARD_num.remove(13)
All_CARD_num.remove(13)
All_CARD_num = np.array(All_CARD_num)
for _ in range(10000):
    np.random.shuffle(seed)
    sample += [All_CARD_num[seed[:16]]]
sample = np.array(sample)
hand_mean = sample.mean(1).reshape(-1, 1)
hand_var = sample.var(1).reshape(-1, 1)
hand_skew = stats.skew(sample, axis=1).reshape(-1, 1)
hand_kurt = stats.kurtosis(sample, axis=1).reshape(-1, 1)
hand_sample = np.concatenate([hand_mean, hand_var, hand_skew, hand_kurt], axis=1)

# matrix sample
All_CARD = '3333444455556666777788889999TTTTJJJJQQQQKKKKAAA2'
N = 1000
for _ in range(N):
    np.random.shuffle(seed)
    idx = seed[:16]
    one_hand = ''
    for i in idx:
        one_hand += All_CARD[i]
    sample += [cards_encode_tensor(one_hand)]
sample = np.array(sample)
# vectorize
vec_sample = sample.sum(1)


# card quality
def sim(x, y):
    # weighted inner product
    # sim_weight = np.linspace(1, 2, 13)
    # x *= sim_weight
    # y *= sim_weight
    return (x*y).sum()/np.linalg.norm(x)/np.linalg.norm(y)
    # E_dist = np.linalg.norm(x-y)
    # return np.exp(-E_dist**2/2)

good_cards = []
good_score = []
with open(DATA_PATH+'card_good_final.csv') as file:
    good = json.load(file)
for card, score in good:
    good_cards += [cards_encode_tensor(card)]
    good_score += [int(score)]


bad_cards = []
bad_score = []
with open(DATA_PATH+'card_bad_final.csv') as file:
    bad = json.load(file)
for card, score in bad:
    bad_cards += [cards_encode_tensor(card)]
    bad_score += [int(score)]

cards = np.array(good_cards[:150] + bad_cards[:150])
card_score = np.array(good_score[:150] + bad_score[:150])

G = nx.Graph()
#G.add_nodes_from(np.arange(len(cards)))

simularities = []
similarity_matrix = np.zeros((len(cards), len(cards)))
for i in np.arange(len(cards)):
    for j in np.arange(i-1, len(cards)):
        simularity = sim(cards[i], cards[j])
        similarity_matrix[i][j] = simularity
        simularities += [simularity]
        G.add_node(i, weight=card_score[i])
        G.add_node(j, weight=card_score[j])
        if 0.85 < simularity < 0.9999999:
            G.add_weighted_edges_from([(i, j, simularity)])

similarity_matrix += similarity_matrix.transpose()
sns.distplot(simularities)
plt.show()


fig, axe = plt.subplots(figsize=(20, 20))
edges, weights = zip(*nx.get_edge_attributes(G, 'weight').items())
weights = np.array(weights)
colors = [G.nodes[n]['weight'] for n in G.nodes]
# Fruchterman-Reingold force-directed algorithm
pos = nx.spring_layout(G,k=0.1)
nx.draw(G, pos=pos, node_color=colors, edgelist=edges, width=weights, ax=axe, alpha=0.7, cmap=plt.cm.bwr)
       # edge_color=weights*256, edge_cmap=plt.cm.Greys)
fig.show()


