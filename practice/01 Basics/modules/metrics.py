import numpy as np


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

    ed_dist = np.sqrt(sum(pow(ts1-ts2, 2) for ts1, ts2 in zip(ts1, ts2)))
    
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


def DTW_distance(ts1, ts2, r=None):
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

    m = ts1.shape[0] + 1
    D = np.zeros((m, m))
    D[0, 1:] = float('Inf')
    D[1:, 0] = float('Inf')
    D[0, 0] = 0

    for i in range(1, m):

        for j in range(1, m):
            dist = (ts1[i - 1] - ts2[j - 1]) ** 2
            D[i, j] = dist + min(D[i - 1, j],
                                 D[i, j - 1],
                                 D[i - 1, j - 1])

    return D[-1, -1]
 