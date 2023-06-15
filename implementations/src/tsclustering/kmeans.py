from tsclustering.functions import metrics, np, barycenters

class KMeans():
    
    def __init__ (self, n_init = 5, k_clusters = 3, max_iter = 100, centroids = [], metric = 'dtw', averaging = 'interpolated'):
        self.k_clusters = k_clusters
        self.n_init = n_init
        self.max_iter = max_iter
        self.centroids = centroids
        self.metric = metric
        self.method = averaging
        
    # assigns each instance to its geometrically closest centroid
    def _assign_clusters(self, X):
        return [np.argmin(np.array([metrics[self.metric](x, centroid)**2 for centroid in self.centroids])) for x in X]
    
    # randomly initializes k centroids to an instance of x
    def _initialize_centroids(self, k_centroids):
        centroids = [self.X[np.random.randint(0, self.X.shape[0])] for k in range(k_centroids)]
        return np.array(centroids, dtype = self.dtype)

    # checks to make sure there are no duplicate centroids
    def _check_centroid_duplicates(self):
        if len(self.centroids) == 0:
            return False
        for k1 in range(len(self.centroids)):
            for k2 in range(len(self.centroids)):
                if k1 != k2 and np.array_equal(self.centroids[k1], self.centroids[k2]):
                    return False
        return True 

    # sets the centroids to the mean length and the mean value of each index, in each cluster
    def _update_centroids(self, X):
        new_centroids = []
        for k in range(len(self.centroids)):  
            cluster = X[np.where(self.clusters==k)[0]]
            if cluster.shape[0] == 0:
                new_centroids.append(self.centroids[k])
            elif cluster.shape[0] == 1:
                new_centroids.append(cluster[0])
            else:
                new_centroids.append(barycenters[self.method](cluster))
        return np.array(new_centroids, dtype = self.dtype)

    # returns True if each centroid has not changed upon centroid update
    def _check_solution(self, new_centroids):
        return np.all([np.array_equal(self.centroids[i], new_centroids[i]) for i in range(len(self.centroids))])
    
    def _get_inertia(self):
        return sum([metrics[self.metric](self.X[i], self.centroids[self.clusters[i]])**2 for i in range(len(self.X))])

    # solves the local cluster problem with randomly assigned centroids
    def local_kmeans(self):
        if len(self.centroids) < self.k_clusters:
            self.centroids = self._initialize_centroids(self.k_clusters)
        for i in range(self.max_iter):
            self.clusters = self._assign_clusters(self.X)
            new_centroids = self._update_centroids(self.X)
            if self._check_solution(new_centroids):
                break
            else:
                self.centroids = new_centroids
        self.inertia = self._get_inertia()

    # solves the local cluster problem n_init times and saves the result with the lowest inertia
    def sample_kmeans(self):
        costs = {}
        for n in range(self.n_init):
            self.centroids = []
            self.clusters = []
            self.local_kmeans()
            costs[self.inertia] = self.clusters, self.centroids
        self.inertia = min(costs)
        self.clusters = costs[self.inertia][0]
        self.centroids = costs[self.inertia][1]

    # Fits centroids and assigns clusters n_init times according to Lloy'ds algorithm
    def fit(self, X):
        self.dtype = object if np.any(np.diff(list(map(len, X)))!=0) else 'float64'
        self.X = np.array(X, dtype = self.dtype)
        self.sample_kmeans()

    # Assigns an out of sample X to each of the nearest centroids
    def predict(self, X):
        dtype = object if np.any(np.diff(list(map(len, X)))!=0) else 'float64'
        X = np.array(X, dtype = dtype)
        return self._assign_clusters(X)

    # Computes the distance of each instance of X to each centroid
    def soft_cluster(self):
        soft_clusters = []
        for centroid in self.centroids:
            distances = []
            for i in range(len(self.X)):
                distances.append(metrics[self.metric](self.X[i], centroid))
            soft_clusters.append(distances)
        a = np.array(soft_clusters)
        a = a.reshape(a.shape[1], a.shape[0]);
        return a
