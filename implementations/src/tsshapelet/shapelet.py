from tsshapelet.functions import metrics, barycenters, manipulations, np

class Shapelet:

    def __init__ (self, series, metric = 'dtw'):
        self.series = np.array(series)
        self.original = self.series
        self.metric = metric

    ''' Preprocessing '''

    def quantile_normalization(self, quantile = 0.5):
        self.series = self.series - np.quantile(self.series, quantile)

    def smooth(self, period):
        self.series = manipulations['moving_average'](self.series, period)

    def phase_sync(self, thres = .9, min_dist = 60):
        ind = manipulations['peak_utils'](self.series, thres = thres, min_dist = min_dist)
        if len(ind) > 3:
            self.series = self.series[ind[2]:]

    def interpolate(self, factor):
        self.series = manipulations['interpolate'](self.series, int(len(self.series)*factor))
        
    ''' Candidate Extraction '''

    def peak_analysis(self, min_dist = 60, thres = 0.75):
        self.indices = manipulations['peak_utils'](self.series, thres = thres, min_dist = min_dist)

    # Uses peakutils to extract subsequences between peak as candidates
    def peak_extraction(self, min_dist = 60, thres = 0.6, max_dist = 150):
        self.peak_analysis(min_dist = min_dist, thres = thres)
        self.candidates = []
        start = 0
        for i in self.indices: 
            candidate = self.series[start:i]
            start = i 
            if min_dist <= len(candidate) <= max_dist:
                self.candidates.append(candidate)

    # Selects qty of subsequences of random length within parameters from time series
    def random_extraction(self, qty, min_dist = 60, max_dist = 150):
        self.candidates = []
        for q in range(qty):
            index = np.random.randint(max_dist, len(self.series)-max_dist)
            length = np.random.randint(min_dist, max_dist) if min_dist != max_dist else max_dist
            self.candidates.append(self.series[index-length//2 : index+length//2])

    # runs the random extraction method with statistically derived parameters
    def normal_extraction(self, qty = None, min_dist = None, thres = None):
        if min_dist != None and thres != None:
            self.peak_analysis(min_dist = min_dist, thres = thres)
        else:
            self.peak_analysis()
        lengths = np.diff(self.indices)
        mean = np.mean(lengths, dtype = int)
        std = np.std(lengths, dtype = int)
        qty = len(self.series) // mean if qty == None else qty
        self.random_extraction(qty, min_dist = mean - std, max_dist = mean + std)

    # Slides window along series and extracts each subsequence as a candidate
    def windowed_extraction(self, window_length = 80, step = 1):
        self.candidates = []
        for i in np.arange(0, len(self.series) - window_length, step):
            self.candidates.append(self.series[i:i+window_length])

    ''' Shapelet Creation '''
    
    # Returns the cost of comparing the given array to each other candidate elementwise
    def _compare_array_to_shapelets(self, array):
        cost = 0
        for shapelet in self.shapelets:
            cost += metrics[self.metric](shapelet, array)
        return cost

    # Returns the cost of comparing the given array to the original series
    def _compare_array_to_series(self, array, step = 1):
        size = len(array)
        cost = 0
        for i in range(size, len(self.series) - size, step):
            cost += metrics[self.metric](array, self.series[i:i+size])
        return cost
    
    # Returns the cost of comparing the given array to each other candidate elementwise
    def _compare_array_to_candidates(self, array):
        cost = 0
        for candidate in self.candidates:
            cost += metrics[self.metric](candidate, array)
        return cost

    # Assigns the candidate with the minimum distance to each other candidate as the shapelet
    def order_candidates(self, comparison = 'candidates'):
        distances = {}
        comparisons = {'candidates': self._compare_array_to_candidates,
                       'series' : self._compare_array_to_series,
                       'geometric' : self._compare_array_to_shapelets
                      }
        for candidate in self.candidates:
            distances[comparisons[comparison](candidate)] = candidate
        self.candidates = []
        for distance in sorted(distances):
            self.candidates.append(distances[distance])

    # Refits all shapelets to a given window size
    def reinterpolate_shapelets(self, size):
        shapelets = []
        for shapelet in self.shapelets:
            shapelets.append(manipulations['reinterpolate'](shapelet, size))
        self.shapelets = shapelets  

    ''' Control Flow ''' 

    # Control flow extract candidates from series
    def candidate_extraction(self, extraction = 'peak', min_dist = 60, thres = 0.8, max_dist = 120, sample = 100, window_length = 100, step = 1):
        if extraction == 'peak':
            self.peak_extraction(min_dist = min_dist, thres = thres, max_dist = max_dist)
        if extraction == 'normal':
            self.normal_extraction(qty = sample)
        if extraction == 'random':
            self.random_extraction(sample, min_dist = min_dist, max_dist = max_dist)
        if extraction == 'windowed':
            self.windowed_extraction(window_length = window_length, step = step)
    
    # Control flow for generating shapelets from candidates
    def shapelet_selection(self, barycenter = None, qty = 1,comparison = 'candidates'):
        if barycenter != None:
            self.shapelets = [barycenters[barycenter](self.candidates)]
        else:
            self.order_candidates(comparison = comparison)
            self.shapelets = self.candidates[:qty]