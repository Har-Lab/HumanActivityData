import numpy as np
from pyts.metrics import dtw
from scipy.interpolate import interp1d
from scipy.signal import correlate

''' Array Manipulations '''

def interpolate(array, length):
    if type(array) != np.array:
        array = np.array(array)
    array_length = len(array)
    return interp1d(np.arange(0, array_length), array)(np.linspace(0.0, array_length-1, length))

def reinterpolate(array, window_length):
    if type(array) != np.array:
        array = np.array(array)
    length = len(array)
    return np.concatenate([np.tile(array, window_length//length),array[:window_length%length]])

def pad(array, length):
    if type(array) != np.array:
        array = np.array(array)
    return np.pad(array, (0,length - len(array)), 'constant')

def center_moving_average(array, period):
    if type(array) != np.array:
        array = np.array(array)
    ret = np.cumsum(array)
    ret[period:] = ret[period:] - ret[:-period]
    return ret[period - 1:] / period

''' Distance Metrics '''

def cross_correlation(arr1, arr2, method = 'avg'):
    if type(arr1) != np.array:
        arr1 = np.array(arr1)
        arr2 = np.array(arr2)
    cases = {'avg': np.mean, 'max' : np.max, 'min' : np.min}
    return cases[method](correlate(arr1, arr2))

def euclidean_distance(arr1, arr2):
    if type(arr1) != np.array:
        arr1 = np.array(arr1)
        arr2 = np.array(arr2)
    return np.linalg.norm(arr1-arr2)

def dynamic_time_warping(arr1, arr2):
    return dtw(arr1, arr2)


''' Barycenter Averaging '''

def average_barycenter(X):
    if type(X) != np.ndarray:
        X = np.array(X)
    return np.mean(X, axis = 0)

def interpolated_average(X):
    if type(X) != np.array:
        X = np.array(X, dtype = object)
    length = np.mean(np.array(list(map(len, X))), dtype = int)
    interpolated_candidates = [interpolate(arr, length) for arr in X]
    return average_barycenter(interpolated_candidates)


''' Scoping Maps '''

metrics  = {'euclidean' : euclidean_distance,
            'correlation' : cross_correlation,
            'dtw' : dynamic_time_warping,
            }

barycenters = {'interpolated' : interpolated_average,
               'average' : average_barycenter
               }

manipulations = {'interpolate' : interpolate,
                 'reinterpolate' : reinterpolate,
                 'pad' : pad,
                 'moving_average' : center_moving_average,
                 }