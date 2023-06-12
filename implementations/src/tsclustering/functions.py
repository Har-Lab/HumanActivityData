import numpy as np
from pyts.metrics import dtw
from scipy.interpolate import interp1d
from scipy.signal import correlate

''' Array Manipulations '''

def interpolate(array, length):
    return interp1d(np.arange(0, len(array)), array)(np.linspace(0.0, len(array)-1, length))

def reinterpolate(array, window_length):
    array = list(array)
    length = len(array)
    return np.array(array*(window_length//length)+array[:window_length%length])

def pad(array, length):
    return np.pad(array, (0,length - len(array)), 'constant')

def center_moving_average(array, period):
    ret = np.cumsum(np.array(array))
    ret[period:] = ret[period:] - ret[:-period]
    return ret[period - 1:] / period

''' Distance Metrics '''

def cross_correlation(arr1, arr2):
    return np.max(correlate(arr1, arr2))

def euclidean_distance(arr1, arr2):
    return np.sqrt(sum((np.array(arr1)-np.array(arr2))**2))

def dynamic_time_warping(arr1, arr2):
    return dtw(arr1, arr2)


''' Barycenter Averaging '''

def euclidean_barycenter(X):
    return np.mean(np.array(X), axis = 0)

def interpolated_barycenter(X, size = 'avg'):
    cases = {'max' : np.max, 'min' : np.min, 'avg' : np.mean}
    length = cases[size](np.array([len(arr) for arr in X]), dtype = int)
    interpolated_candidates = [interpolate(arr, length) for arr in X]
    return euclidean_barycenter(interpolated_candidates)


''' Scoping Maps '''

metrics  = {'euclidean' : euclidean_distance,
            'correlation' : cross_correlation,
            'dtw' : dynamic_time_warping,
            }

barycenters = {'interpolated' : interpolated_barycenter,
               'euclidean' : euclidean_barycenter
               }

manipulations = {'interpolate' : interpolate,
                 'reinterpolate' : reinterpolate,
                 'pad' : pad,
                 'moving_average' : center_moving_average,
                 }