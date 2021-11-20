import networkx as nx
import numpy as np
import random
import seaborn as sns
from tqdm import trange
from matplotlib import pyplot as plt

"""
Density peak Clustering
"""


def distance(i, j):
    return abs(i - j)


def pairwise_distance(max_id, distFunc, directed=False):
    """
    undirected
    """
    res = {}
    for i in trange(max_id, desc='pairwise_dist'):
        for j in range(i + 1, max_id + 1):
            if directed:
                res[(i, j)] = distFunc(i, j)
                res[(j, i)] = distFunc(j, i)
            else:
                res[(i, j)] = res[(j, i)] = distFunc(i, j)
    return res


def spectral_c(G):
    # laplacian = scipy.sparse.csr_matrix.todense(nx.laplacian_matrix(G, weight=1))
    # u, s, v = scipy.linalg.svd(laplacian)
    s = nx.laplacian_spectrum(G)
    s.sort()
    sns.scatterplot(np.arange(len(s)), s)
    plt.title('Spectral Clustering')
    plt.show()


def local_density(max_id, dist_dict, dc=None):
    """
    dc = None will automatically choose dc as shown in the paper
    """
    if dc is None:
        d_max, d_min = max(dist_dict.values()), min(dist_dict.values())
        d_range = d_max - d_min
        while (d_max - d_min) / d_range > 0.001:  # binary search
            dc = (d_max + d_min) / 2
            avg_rho = sum([1 for v in dist_dict.values() if v < dc]) / (max_id + 1) ** 2
            if avg_rho < 0.1:
                d_min = dc
            elif avg_rho > 0.2:
                d_max = dc
            else:
                break
            assert d_max > d_min

    rho = [0] * (1 + max_id)
    for i in trange(max_id, desc='local_density'):
        for j in range(i + 1, max_id + 1):
            d = dist_dict[(i, j)]
            if d < dc:
                rho[i] += 1
                rho[j] += 1

    return rho, dc


def centrality(max_id, rho, dist_dict):
    """
    use sorts
    :param max_id:
    :param rho:
    :param dist_dict: d(i,j) for i < j
    :return:
    """
    delta = [-1] * (max_id + 1)
    sorted_ids = np.argsort(rho)

    # assign maximal values to the point with the largest density
    delta[sorted_ids[-1]] = max(dist_dict.values())

    for i_idx in trange(max_id, desc='centrality'):
        i = sorted_ids[i_idx]
        delta[i] = min(dist_dict[(i, j)] for j in sorted_ids[i_idx + 1:])
    return delta


def cluster(max_id, rho, delta, dist_dict, rho_threshold, delta_threshold, max_clusters=None, gamma=False,
            assign_gap=float('inf')):
    """
    pred_tags: default = -1
    """
    # TODO initial clusters
    pred_tags = [-1] * (max_id + 1)
    centers = {}
    rho = np.array(rho)
    sorted_ids = rho.argsort()[::-1]

    # use ranked gamma to select centers: gamma=True, max_clusters=?
    if gamma:
        gamma = np.array(rho) * np.array(delta)
        for c_, i_ in enumerate(gamma.argsort()[-max_clusters:]):
            pred_tags[i_] = c_
            centers[i_] = c_
    # use thresholds to select centers: rho_threshold=?, delta_threshold=?
    else:
        c_ = 0
        for i_ in sorted_ids:
            if rho[i_] >= rho_threshold and delta[i_] >= delta_threshold:
                pred_tags[i_] = c_
                centers[i_] = c_
                c_ += 1
                if max_clusters is not None and len(centers) >= max_clusters:
                    break

    # assign points same as nearest neighbor with higher densities
    for idx, i in enumerate(sorted_ids):
        if idx == 0:
            continue
        if pred_tags[i] < 0:  # smaller id must already be assigned
            neighbor = None
            min_dist = float('inf')
            for j in sorted_ids[:idx]:
                d = dist_dict[(i, j)]
                if d < min_dist:
                    neighbor = j
                    min_dist = d
            if min_dist <= assign_gap:
                pred_tags[i] = pred_tags[neighbor]

            # assert pred_tags[neighbor] >= 0
    # assert -1 not in pred_tags

    return pred_tags, centers


class DistanceFunc(object):
    def __init__(self):
        pass


class DensityFunc(object):
    def __init__(self, distF=None):
        pass


class DPClustering(object):
    def __init__(self, data: np.array, tags: list):
        """
        max_id: all points are [0, 1, 2, ..., max_id]
        """
        self.data = data  # N by p matrix
        self.tags = tags  # len(tags) == data.shape[0]
        self.num_true_clusters = len(set(tags))
        self.max_id = len(data) - 1

        self.rho = []
        self.delta = []
        self.dc = None
        self.num_pred_clusters = None
        self.rho_threshold = None
        self.delta_threshold = None
        self.pred_tags = []
        self.pred_centers = {}

    def run(self, rho_dist, delta_dist, assign_dist=None, **kwargs):
        """
        dist_dict: dist_dict[(i, j)] = d(i, j)
        """
        self.rho_threshold = kwargs.get('rho_threshold')
        self.delta_threshold = kwargs.get('delta_threshold')
        if assign_dist is None:
            assign_dist = rho_dist
        self.rho, self.dc = local_density(self.max_id, rho_dist)
        self.delta = centrality(self.max_id, self.rho, delta_dist)
        self.pred_tags, self.pred_centers = cluster(self.max_id, self.rho, self.delta, assign_dist, **kwargs)
                                                    # rho_threshold=rho_threshold,
                                                    # delta_threshold=delta_threshold,
                                                    # max_clusters=max_clusters)
        self.num_pred_clusters = len(self.pred_centers)
        return self.pred_tags, self.pred_centers

    def eval(self):
        c_pred_tags = [[] for _ in range(self.num_pred_clusters)]
        for i_, t_ in enumerate(self.pred_tags):
            c_pred_tags[t_] += [self.tags[i_]]
        total_entropy, total_purity = 0, 0
        for c in c_pred_tags:
            total_entropy += len(c) / (self.max_id + 1) * self.entropy(c)
            total_purity += len(c) / (self.max_id + 1) * self.purity(c)

        return total_entropy, total_purity

    def align_prediction(self):
        """
        it's computational difficult for large number of communities for permutation
        so I set priorities for the assignment by the cardinalities
        """
        true_clusters, pred_clusters = self.get_clusters()
        not_aligned = list(range(self.num_true_clusters))
        pred_clusters.sort(key=len, reverse=True)

        align = []
        for c in pred_clusters:
            pos = int(np.argmin([self.balanced_error_rate(c, true_clusters[n_]) for n_ in not_aligned]))
            align.append(not_aligned.pop(pos))
            if not not_aligned:  # if it predicts more clusters than needed -> keep extra labels
                assert self.num_pred_clusters >= self.num_true_clusters
                align += [len(align) + _ for _ in range(self.num_pred_clusters - self.num_true_clusters)]
                break

        for c_, t_ in zip(pred_clusters, align):
            for i_ in c_:
                self.pred_tags[i_] = t_

    def get_clusters(self):
        true_clusters = [[] for _ in range(self.num_true_clusters)]
        pred_clusters = [[] for _ in range(self.num_pred_clusters)]
        for idx, (tc, pc) in enumerate(zip(self.tags, self.pred_tags)):
            true_clusters[tc] += [idx]
            pred_clusters[pc] += [idx]
        return true_clusters, pred_clusters

    def plot(self, show=True, threshold=False):
        if self.data.shape[-1] > 2:
            from sklearn.manifold import TSNE
            data = TSNE(n_components=2).fit_transform(self.data)
            x_, y_ = data[:, 0], data[:, 1]
        else:
            x_, y_ = self.data[:, 0], self.data[:, 1]

        # plots
        fig, ax = plt.subplots(2, 2, figsize=(15, 15))
        sns.scatterplot(x=x_, y=y_, hue=[str(_) for _ in self.tags], ax=ax[0, 0])
        ax[0, 0].get_legend().remove()
        ax[0, 0].set_title('data')

        sns.scatterplot(x=x_, y=y_, hue=[str(_) for _ in self.pred_tags], ax=ax[1, 0])
        center_x = [x_[c] for c in self.pred_centers.keys()]
        center_y = [y_[c] for c in self.pred_centers.keys()]
        sns.scatterplot(x=center_x, y=center_y, ax=ax[1, 0], marker='+', color='black', s=400)
        ax[1, 0].get_legend().remove()
        ax[1, 0].set_title(f'pred |clusters|={self.num_pred_clusters}, dc={self.dc:.5f}')

        sns.scatterplot(x=self.rho, y=self.delta, hue=[str(_) for _ in self.tags], ax=ax[0, 1])
        ax[0, 1].set_xlabel('rho (density)')
        ax[0, 1].set_ylabel('delta (centrality)')
        if threshold:
            ax[0, 1].axvline(x=self.rho_threshold, color='b', linestyle='--')
            ax[0, 1].axhline(y=self.delta_threshold, color='b', linestyle='--')
        # plot selected centers:
        centers = list(self.pred_centers.keys())
        sns.scatterplot(x=[self.rho[p] for p in centers], y=[self.delta[p] for p in centers], marker='+',
                        ax=ax[0, 1], color='black', s=400)
        ax[0, 1].get_legend().remove()
        ax[0, 1].set_title('delta-rho')

        gamma = np.array(self.rho) * np.array(self.delta)
        gamma.sort()
        sns.scatterplot(x=range(len(gamma)), y=gamma[::-1], ax=ax[1, 1])
        ax[1, 1].set_title('gamma')
        if show:
            fig.show()
        return fig, ax

    @staticmethod
    def entropy(group: list):
        if len(set(group)) <= 1:
            return 0
        res = 0
        N_ = len(group)
        for c_ in set(group):
            res -= group.count(c_) / N_ * np.log2(group.count(c_) / N_)
        return res

    @staticmethod
    def purity(group: list):
        if len(set(group)) <= 1:
            return 1
        mode = max(set(group), key=group.count)
        return group.count(mode) / len(group)

    @staticmethod
    def balanced_error_rate(g1, g2):
        p1 = len([_ for _ in g1 if _ not in g2]) / len(g1)
        p2 = len([_ for _ in g2 if _ not in g1]) / len(g2)
        return 0.5 * (p1 + p2)

#
# if __name__ == '__main__':
#     def distF(i, j):
#         return ((x[i] - x[j]) ** 2 + (y[i] - y[j]) ** 2) ** 0.5
#
#
#     length = 100
#     x = [random.gauss(0, 1) for _ in range(length // 2)] + \
#         [random.gauss(3, 1) for _ in range(length)] + \
#         [random.gauss(5, 1) for _ in range(length * 2)]
#     y = [random.gauss(0, 1) for _ in range(length // 2)] + \
#         [random.gauss(3, 1) for _ in range(length)] + \
#         [random.gauss(0, 1) for _ in range(length * 2)]
#     tag_ = ['0'] * (length // 2) + ['1'] * length + ['2'] * length * 2
#
#     in_data = np.array([x, y]).transpose()
#     DP = DPClustering(in_data, tag_, pairwise_distance(len(x) - 1, distF))
#     DP.run(3)
#     DP.plot()

#     import random
#     import seaborn as sns
#     from matplotlib import pyplot as plt
#
#     length = 100
#
#     x = [random.gauss(0, 1) for _ in range(length//2)] + \
#         [random.gauss(3, 1) for _ in range(length)] + \
#         [random.gauss(5, 1) for _ in range(length*2)]
#     y = [random.gauss(0, 1) for _ in range(length//2)] + \
#         [random.gauss(3, 1) for _ in range(length)] + \
#         [random.gauss(0, 1) for _ in range(length*2)]
#     t_ = ['0'] * (length//2) + ['1'] * length + ['2'] * length*2
#     maxima_id = len(x) - 1
#
#     fig, ax = plt.subplots(2, 2, figsize=(15, 15))
#
#     sns.scatterplot(x=x, y=y, hue=t_, ax=ax[0, 0])
#     ax[0, 0].get_legend().remove()
#     ax[0, 0].set_title('data')
#
#

#
#     d_ = pairwise_distance(maxima_id, distF)
#     rho_, dc_ = local_density(maxima_id, d_)
#     delta_ = centrality(maxima_id, rho_, d_)
#     pre_t, centers = cluster(maxima_id, rho_, delta_, d_, num_clusters=3)
#
#
#     # plots
#     sns.scatterplot(x=rho_, y=delta_, hue=t_, ax=ax[0, 1])
#     ax[0, 1].get_legend().remove()
#     ax[0, 1].set_title('delta-rho')
#
#     gamma = np.array(rho_) * np.array(delta_)
#     gamma.sort()
#     sns.scatterplot(x=range(len(gamma)), y=gamma[::-1], ax=ax[1, 1])
#     ax[1, 1].set_title('gamma')
#
#     sns.scatterplot(x=x, y=y, hue=[str(_) for _ in pre_t], ax=ax[1, 0])
#     center_x = [x[c] for c in centers.keys()]
#     center_y = [y[c] for c in centers.keys()]
#     sns.scatterplot(x=center_x, y=center_y, ax=ax[1, 0], marker='+', color='black', s=400)
#     ax[1, 0].get_legend().remove()
#     ax[1, 0].set_title(f'pred, dc={dc_}')
#     fig.show()
#
# TODO: create graph version? automatically find # of clusters?
# TODO: how to initialize centers?  density cutoff -> gauss?
