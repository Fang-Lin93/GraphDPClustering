import numpy as np
import random
import seaborn as sns
from matplotlib import pyplot as plt

"""
Density peak Clustering
"""


def distance(i, j):
    return abs(i - j)


def pairwise_distance(max_id, distFunc):
    """
    undirected
    """
    res = {}
    for i in range(max_id):
        for j in range(i + 1, max_id + 1):
            res[(i, j)] = res[(j, i)] = distFunc(i, j)
    return res


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
            elif avg_rho > 0.15:
                d_max = dc
            else:
                break
            assert d_max > d_min

    rho = [0] * (1 + max_id)
    for i in range(max_id):
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

    delta[sorted_ids[-1]] = max(dist_dict.values())

    for i_idx in range(max_id):
        i = sorted_ids[i_idx]
        delta[i] = min(dist_dict[(i, j)] for j in sorted_ids[i_idx + 1:])
    return delta


def cluster(max_id, rho, delta, dist_dict, num_clusters):
    """
    tags: defualt = -1
    """

    # TODO initial clusters
    tags = [-1] * (max_id + 1)
    centers = {}
    rho = np.array(rho)

    gamma = np.array(rho) * np.array(delta)
    for c_, i_ in enumerate(gamma.argsort()[-num_clusters:]):
        tags[i_] = c_
        centers[i_] = c_

    # assign points same as nearest neighbor with higher densities
    sorted_ids = rho.argsort()[::-1]

    while -1 in tags:
        for idx, i in enumerate(sorted_ids):
            if tags[i] < 0:
                neighbor = -1
                min_dist = float('inf')
                for j in sorted_ids[:idx]:
                    if tags[j] >= 0:
                        d = dist_dict[(i, j)] if i < j else dist_dict[(j, i)]
                        if d < min_dist:
                            neighbor = j
                            min_dist = d

                # assert neighbor >= 0
                tags[i] = tags[neighbor]

    return tags, centers


class DistanceFunc(object):
    def __init__(self):
        pass


class DensityFunc(object):
    def __init__(self, distF=None):
        pass


class DPClustering(object):
    def __init__(self, data: np.array, tags: list, distFunc):
        """
        max_id: all points are [0, 1, 2, ..., max_id]
        distF: it can give d(i, j) for any i, j in [0, 1, 2, ..., max_id]
        """
        self.data = data  # N by p matrix
        self.tags = tags  # len(tags) == data.shape[0]
        self.max_id = len(data) - 1
        self.distFunc = distFunc

        self.dist_dict = {}
        self.rho = []
        self.delta = []
        self.dc = None
        self.num_clusters = None
        self.pred_tags = []
        self.pred_centers = {}

    def run(self, num_clusters: int = 3):
        self.num_clusters = num_clusters
        self.dist_dict = pairwise_distance(self.max_id, self.distFunc)
        self.rho, self.dc = local_density(self.max_id, self.dist_dict)
        self.delta = centrality(self.max_id, self.rho, self.dist_dict)
        self.pred_tags, self.pred_centers = cluster(self.max_id, self.rho, self.delta, self.dist_dict,
                                                    num_clusters=num_clusters)
        return self.pred_tags, self.pred_centers

    def plot(self):
        if self.data.shape[-1] > 2:
            from sklearn.manifold import TSNE
            data = TSNE(n_components=2).fit_transform(self.data)
            x_, y_ = data[:, 0], data[:, 1]
        else:
            x_, y_ = self.data[:, 0], self.data[:, 1]

        # plots
        fig, ax = plt.subplots(2, 2, figsize=(15, 15))
        sns.scatterplot(x=x_, y=y_, hue=self.tags, ax=ax[0, 0])
        ax[0, 0].get_legend().remove()
        ax[0, 0].set_title('data')

        sns.scatterplot(x=self.rho, y=self.delta, hue=self.tags, ax=ax[0, 1])
        ax[0, 1].set_xlabel('rho (density)')
        ax[0, 1].set_ylabel('delta (centrality)')
        ax[0, 1].get_legend().remove()
        ax[0, 1].set_title('delta-rho')

        gamma = np.array(self.rho) * np.array(self.delta)
        gamma.sort()
        sns.scatterplot(x=range(len(gamma)), y=gamma[::-1], ax=ax[1, 1])
        ax[1, 1].set_title('gamma')

        sns.scatterplot(x=x_, y=y_, hue=[str(_) for _ in self.pred_tags], ax=ax[1, 0])
        center_x = [x_[c] for c in self.pred_centers.keys()]
        center_y = [y_[c] for c in self.pred_centers.keys()]
        sns.scatterplot(x=center_x, y=center_y, ax=ax[1, 0], marker='+', color='black', s=400)
        ax[1, 0].get_legend().remove()
        ax[1, 0].set_title(f'pred, dc={self.dc:.5f}')
        fig.show()



if __name__ == '__main__':

    def distF(i, j):
        # return np.exp((x[i] - x[j]) ** 2 + (y[i] - y[j]) ** 2)
        return ((x[i] - x[j]) ** 2 + (y[i] - y[j]) ** 2) ** 0.5
    length = 100
    x = [random.gauss(0, 1) for _ in range(length // 2)] + \
        [random.gauss(3, 1) for _ in range(length)] + \
        [random.gauss(5, 1) for _ in range(length * 2)]
    y = [random.gauss(0, 1) for _ in range(length // 2)] + \
        [random.gauss(3, 1) for _ in range(length)] + \
        [random.gauss(0, 1) for _ in range(length * 2)]
    t_ = ['0'] * (length // 2) + ['1'] * length + ['2'] * length * 2

    in_data = np.array([x, y]).transpose()
    DP = DPClustering(in_data, t_, distF)
    DP.run(3)
    DP.plot()




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
      # TODO: create graph version? automatically find # of clusters? density cutoff -> gauss
      # TODO: how to initialize centers?
