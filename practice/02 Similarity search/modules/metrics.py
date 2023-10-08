import numpy as np
import math

def ED_distance(ts1, ts2):
    """
    Calculate the Euclidean distance.

    Parameters
    ----------
    ts1 : numpy.ndarray
        The first time series.

    ts2 : numpy.ndarray
        The second time series.

    Returns
    -------
    ed_dist : float
        Euclidean distance between ts1 and ts2.
    """
    
    ed_dist = 0

    # INSERT YOUR CODE

    return ed_dist


def norm_ED_distance(ts1, ts2):
    """
    Calculate the normalized Euclidean distance.

    Parameters
    ----------
    ts1 : numpy.ndarray
        The first time series.

    ts2 : numpy.ndarray
        The second time series.

    Returns
    -------
    norm_ed_dist : float
        The normalized Euclidean distance between ts1 and ts2.
    """

    norm_ed_dist = 0

    # INSERT YOUR CODE

    return norm_ed_dist


def DTW_distance(ts1, ts2, r=0.05):
    """
    Calculate DTW distance.

    Parameters
    ----------
    ts1 : numpy.ndarray
        The first time series.

    ts2 : numpy.ndarray
        The second time series.

    r : float
        Warping window size.
    
    Returns
    -------
    dtw_dist : float
        DTW distance between ts1 and ts2.
    """

    dtw_dist = 0

    m = len(ts1)
    D = np.zeros((m + 1, m + 1))
    D[:, :] = float('Inf')
    D[0, 0] = 0

    r = math.floor(len(ts1) * r)

    for i in range(1, m + 1):

        for j in range(max(1, i - r), min(m, i + r) + 1):
            dist = (ts1[i - 1] - ts2[j - 1]) ** 2

            D[i, j] = dist + min(D[i - 1, j],
                                 D[i, j - 1],
                                 D[i - 1, j - 1])


    dtw_dist = D[m, m]

    return dtw_dist