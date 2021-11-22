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


def range_dc_select(max_id, dist_dict, avg_rho_range=(0.01, 0.02)):
    avg_rho, dc = 0, -1
    d_max, d_min = max(dist_dict.values()), min(dist_dict.values())
    d_range = d_max - d_min
    while (d_max - d_min) / d_range > 0.001:  # binary search
        dc = (d_max + d_min) / 2
        avg_rho = sum([1 for v in dist_dict.values() if v < dc]) / (max_id + 1) ** 2
        if avg_rho < avg_rho_range[0]:
            d_min = dc
        elif avg_rho > avg_rho_range[1]:
            d_max = dc
        else:
            break
        assert d_max > d_min
    print('avg_rho=', avg_rho, 'dc=', dc)
    return dc


def local_density(max_id, dist_dict, gauss=True):
    """
    dc = None will automatically choose dc as shown in the paper
    """
    # if dc is None:
    #     range_dc_select(max_id, dist_dict, avg_rho_range)
    percent = 2.0
    position = int(max_id * (max_id + 1) / 2 * percent / 100)
    dc = sorted(dist_dict.values())[position * 2 + max_id]
    print('dc=', dc)

    rho = [0] * (1 + max_id)
    for i in trange(max_id, desc='local_density'):
        for j in range(i + 1, max_id + 1):
            d = dist_dict[(i, j)]
            if d < dc:
                if gauss:
                    rho[i] += np.exp(- (d / dc) ** 2)
                    rho[j] += np.exp(- (d / dc) ** 2)
                else:
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
    delta, neighbor = [-1] * (max_id + 1), [-1] * (max_id + 1)
    sorted_ids = np.argsort(rho)

    # assign maximal values to the point with the largest density
    delta[sorted_ids[-1]] = max(dist_dict.values())

    for i_idx in trange(max_id, desc='centrality'):
        i = sorted_ids[i_idx]
        nearest = i_idx + 1 + np.argmin([dist_dict[(i, j)] for j in sorted_ids[i_idx + 1:]])
        neighbor[i] = sorted_ids[nearest]
        delta[i] = dist_dict[(i, sorted_ids[nearest])]
        assert min(dist_dict[(i, j)] for j in sorted_ids[i_idx + 1:]) == delta[i]
    return delta, neighbor


def cluster(max_id, rho, delta, neighbor, rho_threshold, delta_threshold, max_clusters=None, gamma=False,
            assign_gap=float('inf'), **kwargs):
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

    assert centers

    # assign points same as nearest neighbor with higher densities
    for idx, i in enumerate(sorted_ids):
        if idx == 0:
            continue
        if pred_tags[i] < 0:  # smaller id must already be assigned
            pred_tags[i] = pred_tags[neighbor[i]]

    # for idx, i in enumerate(sorted_ids):
    #     if idx == 0:
    #         continue
    #     if pred_tags[i] < 0:  # smaller id must already be assigned
    #         neighbor = None
    #         min_dist = float('inf')
    #         for j in sorted_ids[:idx]:
    #             d = dist_dict[(i, j)]
    #             if d < min_dist:
    #                 neighbor = j
    #                 min_dist = d
    #         if min_dist <= assign_gap:
    #             pred_tags[i] = pred_tags[neighbor]

            # assert pred_tags[neighbor] >= 0
    assert -1 not in pred_tags

    return pred_tags, centers


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
        self.neighbor = []
        self.dc = None
        self.rho_dist = {}
        self.delta_dist = {}
        # self.assign_dist = {}
        self.num_pred_clusters = None
        self.rho_threshold = None
        self.delta_threshold = None
        self.pred_tags = []
        self.pred_centers = {}
        self.avg_rho_range = (0.05, 0.1)
        self.align = {}
        self.use_km = False

    def reset(self):
        self.rho = []
        self.delta = []
        self.neighbor = []
        self.dc = None
        self.num_pred_clusters = None
        self.rho_threshold = None
        self.delta_threshold = None
        self.pred_tags = []
        self.pred_centers = {}
        self.avg_rho_range = (0.05, 0.1)
        self.align = {}

    def run(self, rho_dist, delta_dist, **kwargs):
        """
        dist_dict: dist_dict[(i, j)] = d(i, j)
        """
        # if assign_dist is None:
        #     assign_dist = rho_dist

        self.reset()
        self.rho_dist = rho_dist
        self.delta_dist = delta_dist
        # self.assign_dist = assign_dist
        self.rho_threshold = kwargs.get('rho_threshold')
        self.delta_threshold = kwargs.get('delta_threshold')
        self.avg_rho_range = kwargs.get('avg_rho_range', (0.05, 0.1))

        self.rho, self.dc = local_density(self.max_id, rho_dist, gauss=kwargs.get('gauss', True))
        self.delta, self.neighbor = centrality(self.max_id, self.rho, delta_dist)
        self.pred_tags, self.pred_centers = cluster(self.max_id, self.rho, self.delta, self.neighbor, **kwargs)
        self.num_pred_clusters = len(self.pred_centers)
        self.use_km = False
        self.align_prediction()
        return self.pred_tags, self.pred_centers

    def k_means(self, data, n_clusters):
        """
        data: no.array of shape Nxp
        """
        # self.reset()
        from sklearn.cluster import KMeans
        km = KMeans(n_clusters=n_clusters).fit(data)
        self.pred_tags = km.labels_.tolist()

        self.pred_centers = {np.argmin([np.linalg.norm(x - c) for x in data]): v
                             for v, c in enumerate(km.cluster_centers_)}
        self.num_pred_clusters = len(self.pred_centers)

        self.use_km = True
        self.align_prediction()
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
        c_ids = np.argsort([len(_) for _ in pred_clusters])[::-1]
        # pred_clusters.sort(key=len, reverse=True)

        align = {v: v for v in range(self.num_pred_clusters)}
        for c in c_ids:
            pos = int(np.argmin([self.balanced_error_rate(pred_clusters[c], true_clusters[n_]) for n_ in not_aligned]))
            align[c] = not_aligned.pop(pos)
            if not not_aligned:  # if it predicts more clusters than needed -> keep extra labels
                assert self.num_pred_clusters >= self.num_true_clusters
                # align += [len(align) + _ for _ in range(self.num_pred_clusters - self.num_true_clusters)]
                break

        self.align = align

        # for c_, t_ in zip(pred_clusters, align):
        #     for i_ in c_:
        #         self.pred_tags[i_] = t_

    def get_clusters(self):
        true_clusters = [[] for _ in range(self.num_true_clusters)]
        pred_clusters = [[] for _ in range(self.num_pred_clusters)]
        for idx, (tc, pc) in enumerate(zip(self.tags, self.pred_tags)):
            true_clusters[tc] += [idx]
            pred_clusters[pc] += [idx]
        return true_clusters, pred_clusters

    def plot(self, show=True, threshold=False, desc=''):
        if self.data.shape[-1] > 2:
            from sklearn.manifold import TSNE
            data = TSNE(n_components=2).fit_transform(self.data)
            x_, y_ = data[:, 0], data[:, 1]
        else:
            x_, y_ = self.data[:, 0], self.data[:, 1]

        # plots
        fig, ax = plt.subplots(2, 2, figsize=(15, 15))
        fig.suptitle(desc)
        sns.scatterplot(x=x_, y=y_, hue=[str(_) for _ in self.tags], ax=ax[0, 0])
        ax[0, 0].get_legend().remove()
        ax[0, 0].set_title('data')

        sns.scatterplot(x=x_, y=y_, hue=[str(self.align[_]) for _ in self.pred_tags], ax=ax[1, 0])
        center_x = [x_[c] for c in self.pred_centers.keys()]
        center_y = [y_[c] for c in self.pred_centers.keys()]
        sns.scatterplot(x=center_x, y=center_y, ax=ax[1, 0],
                        marker='+', color='black' if not self.use_km else 'red', s=400)
        ax[1, 0].get_legend().remove()
        ax[1, 0].set_title(
            f'pred |clusters|={self.num_pred_clusters} ' + f'dc={self.dc}' if self.dc is not None else '')

        sns.scatterplot(x=self.rho, y=self.delta, hue=[str(_) for _ in self.tags], ax=ax[0, 1])
        ax[0, 1].set_xlabel('rho (density)')
        ax[0, 1].set_ylabel('delta (centrality)')
        if threshold:
            ax[0, 1].axvline(x=self.rho_threshold, color='b', linestyle='--')
            ax[0, 1].axhline(y=self.delta_threshold, color='b', linestyle='--')
        # plot selected centers:
        centers = list(self.pred_centers.keys())
        sns.scatterplot(x=[self.rho[p] for p in centers], y=[self.delta[p] for p in centers], ax=ax[0, 1],
                        marker='+', color='black' if not self.use_km else 'red', s=400)
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

# TODO: create graph version? automatically find # of clusters?
# TODO: how to initialize centers?  density cutoff -> gauss?
