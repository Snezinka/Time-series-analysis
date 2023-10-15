import numpy as np
import copy
import math
from modules.utils import *
from modules.metrics import *


class BestMatchFinder:
    """
    Base Best Match Finder.

    Parameters
    ----------
    query : numpy.ndarrray
        Query.

    ts : numpy.ndarrray
        Time series.

    excl_zone_denom : float, default = 1
        The exclusion zone.

    top_k : int, default = 3
        Count of the best match subsequences.

    normalize : bool, default = True
        Z-normalize or not subsequences before computing distances.

    r : float, default = 0.05
        Warping window size.
    """

    def __init__(self, ts, query, exclusion_zone=1, top_k=3, normalize=True, r=0.05):

        self.query = copy.deepcopy(np.array(query))
        self.ts = ts
        self.excl_zone_denom = exclusion_zone
        self.top_k = top_k
        self.normalize = normalize
        self.r = r
        self.bestmatch = {"index": None, "distance": None}

    def _apply_exclusion_zone(self, a, idx, excl_zone):
        """
        Apply an exclusion zone to an array (inplace).

        Parameters
        ----------
        a : numpy.ndarrray
            The array to apply the exclusion zone to.

        idx : int
            The index around which the window should be centered.

        excl_zone : int
            Size of the exclusion zone.

        Returns
        -------
        a: numpy.ndarrray
            The array which is applied the exclusion zone.
        """

        zone_start = max(0, idx - excl_zone)
        zone_stop = min(a.shape[-1], idx + excl_zone)
        a[zone_start: zone_stop + 1] = np.inf

        return a

    def _top_k_match(self, distances, bsf):
        """
        Find the top-k match subsequences.

        Parameters
        ----------
        distances : list
            Distances between query and subsequences of time series.

        m: int
            Subsequence length.

        bsf : float
            Best-so-far.

        excl_zone : int
            Size of the exclusion zone.

        Returns
        -------
        best_match_results: dict
            Dictionary containing results of algorithm.
        """

        distances = np.copy(distances)
        top_k_match_idx = []
        top_k_match_dist = []

        if (self.excl_zone_denom is None):
            self.excl_zone_denom = 0
        else:
            self.excl_zone_deno = int(np.ceil(len(distances) / self.excl_zone_denom))

        for i in range(self.top_k):
            min_idx = np.argmin(distances)
            min_dist = distances[min_idx]

            if (np.isnan(min_dist)) or (np.isinf(min_dist)):
                continue
            else:
                distances = self._apply_exclusion_zone(distances, min_idx, self.excl_zone_deno)
                top_k_match_idx.append(min_idx)
                top_k_match_dist.append(min_dist)

        return {'index': top_k_match_idx, 'distance': top_k_match_dist}

    def perform(self):

        raise NotImplementedError


class NaiveBestMatchFinder(BestMatchFinder):
    """
    Naive Best Match Finder.
    """

    def __init__(self, ts=None, query=None, exclusion_zone=1, top_k=3, normalize=True, r=0.05):
        super().__init__(ts, query, exclusion_zone, top_k, normalize, r)

    def perform(self):
        """
        Perform the best match finder using the naive algorithm.

        Returns
        -------
        best_match_results: dict
            Dictionary containing results of the naive algorithm.
        """


        bsf = float("inf")

        if (self.excl_zone_denom is None):
            excl_zone = 0
        else:
            excl_zone = int(np.ceil(len(self.query) / self.excl_zone_denom))

        distances = []
        index = []

        for i in range(len(self.ts) - len(self.query)):
            subsequence = self.ts[i:i + len(self.query)]
            subsequence_index = np.arange(i, i + len(self.query))
            dist = DTW_distance(self.query, subsequence, r=self.r)
            if dist < bsf:
                bsf = dist
                distances.append(dist)
                index.append(subsequence_index)

        top_k_dist = self._top_k_match(distances, bsf, excl_zone)

        self.bestmatch["index"] = [index[i] for i in top_k_dist["index"]]
        self.bestmatch["distance"] = top_k_dist["distance"]

        return self.bestmatch


class UCR_DTW(BestMatchFinder):
    """
    UCR-DTW Match Finder.
    """

    def __init__(self, ts=None, query=None, exclusion_zone=1, top_k=3, normalize=True, r=0.05):
        super().__init__(ts, query, exclusion_zone, top_k, normalize, r)

    def _LB_Kim(self, subs1, subs2):
        """
        Compute LB_Kim lower bound between two subsequences.

        Parameters
        ----------
        subs1 : numpy.ndarrray
            The first subsequence.

        subs2 : numpy.ndarrray
            The second subsequence.

        Returns
        -------
        lb_Kim : float
            LB_Kim lower bound.
        """

        lb_Kim = ED_distance(subs1[0], subs2[0]) + ED_distance(subs1[-1], subs2[-1])

        return lb_Kim

    def _LB_Keogh(self, subs1, subs2, r):
        """
        Compute LB_Keogh lower bound between two subsequences.

        Parameters
        ----------
        subs1 : numpy.ndarrray
            The first subsequence.

        subs2 : numpy.ndarrray
            The second subsequence.

        r : float
            Warping window size.

        Returns
        -------
        lb_Keogh : float
            LB_Keogh lower bound.
        """

        lb_Keogh = 0

        r = math.floor(len(subs1) * r)

        for i in range(len(subs1)):
            k_low = i - r
            k_high = i + r

            if k_low < 0:
                k_low = 0

            if k_high > len(subs1) - 1:
                k_high = len(subs1) - 1

            u = max(subs2[k_low:k_high])
            l = min(subs2[k_low:k_high])

            if subs1[i] > u:
                lb_Keogh += pow(subs1[i] - u, 2)
            elif subs1[i] < l:
                lb_Keogh += pow(subs1[i] - l, 2)
            else:
                lb_Keogh += 0

        return lb_Keogh

    def perform(self):
        """
        Perform the best match finder using UCR-DTW algorithm.

        Returns
        -------
        best_match_results: dict
            Dictionary containing results of UCR-DTW algorithm.
        """

        bsf = float("inf")
        if (self.normalize):
            self.ts = z_normalize(self.ts)

        distances = []
        index = []

        self.lb_Kim_num = 0
        self.lb_KeoghQC_num = 0
        self.lb_KeoghCQ_num = 0

        for i in range(len(self.ts) - len(self.query)):
            subsequence = self.ts[i:i + len(self.query)]
            subsequence_index = np.arange(i, i + len(self.query))

            if self._LB_Kim(self.query, subsequence) < bsf:

                if self._LB_Keogh(self.query, subsequence, self.r) < bsf:

                    if self._LB_Keogh(subsequence, self.query, self.r) < bsf:

                        dist = DTW_distance(self.query, subsequence, r=self.r)

                        if dist < bsf:
                            bsf = dist
                            distances.append(dist)
                            index.append(subsequence_index)

                    else:
                        self.lb_KeoghCQ_num += 1

                else:
                    self.lb_KeoghQC_num += 1

            else:
                self.lb_Kim_num += 1

        top_k_dist = self._top_k_match(distances, bsf)

        self.bestmatch["index"] = [index[i] for i in top_k_dist["index"]]
        self.bestmatch["distance"] = top_k_dist["distance"]

        return {'index': self.bestmatch['index'],
                'distance': self.bestmatch['distance'],
                'lb_Kim_num': self.lb_Kim_num,
                'lb_KeoghCQ_num': self.lb_KeoghCQ_num,
                'lb_KeoghQC_num': self.lb_KeoghQC_num
                }
