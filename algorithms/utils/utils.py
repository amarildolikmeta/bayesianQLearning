import numpy as np

def moving_average(data, window_width):
    cumsum_vec = np.cumsum(np.insert(data, 0, 0))
    ma_vec = (cumsum_vec[window_width:] - cumsum_vec[:-window_width]) / window_width
    return ma_vec

def argmaxrand(a):
    '''
    If a has only one max it is equivalent to argmax, otehrwise it uniformly random selects a maximum
    '''

    indeces = np.where(np.array(a) == np.max(a))[0]
    return np.random.choice(indeces)

def cummean(a):
    '''
    Performs the cumulative mean of a vector
    '''

    return np.cumsum(a) / (np.arange(len(a)) + 1)