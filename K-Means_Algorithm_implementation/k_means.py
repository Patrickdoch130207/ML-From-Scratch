import numpy as np

def euclidian_distance(X, centroids):
    """
    Calcule la distance euclidienne entre chaque point et chaque centroïde.

    Utilise le mécanisme de broadcasting de NumPy pour vectoriser le calcul
    et éviter les boucles explicites, optimisant ainsi les performances.

    Args:
        X (np.ndarray): Matrice de données de forme (n_samples, n_features).
        centroids (np.ndarray): Matrice des centroïdes de forme (k, n_features).

    Returns:
        np.ndarray: Matrice des distances de forme (n_samples, k).
    """
    distances = np.sqrt(np.sum((X[:, np.newaxis, :] - centroids)**2, axis=2))
    return distances

def manhattan_distance(X, centroids):
    """
    Calcule la distance de Manhattan (L1) entre chaque point et chaque centroïde.

    Args:
        X (np.ndarray): Matrice de données de forme (n_samples, n_features).
        centroids (np.ndarray): Matrice des centroïdes de forme (k, n_features).

    Returns:
        np.ndarray: Matrice des distances de forme (n_samples, k).
    """
    distance = np.sum(abs(X[:, np.newaxis, :] - centroids), axis=2)
    return distance

class KMeans:
    """
    Algorithme de clustering K-Means.

    Cette implémentation supporte différentes méthodes d'initialisation, notamment 
    K-Means++, et permet l'utilisation de métriques de distance personnalisées.

    Attributes:
        k (int): Nombre de clusters.
        max_iter (int): Nombre maximum d'itérations.
        init_method (str): Méthode d'initialisation ('forgy', 'random_partition', 'kmeans++').
        distance_metric (function): Fonction utilisée pour calculer la distance.
        tol (float): Seuil de tolérance pour déclarer la convergence.
        random_state (int, optional): Graine pour la reproductibilité des résultats.
        centroids (np.ndarray): Coordonnées des centres de clusters après entraînement.
        labels (np.ndarray): Indices des clusters assignés à chaque point d'entraînement.
    """

    def __init__(self, k=2, max_iter=100, init_method="forgy", 
                 distance_metric=euclidian_distance, tol=0.01, random_state=None):
        """Initialise les hyperparamètres du modèle KMeans."""
        self.k = k
        self.max_iter = max_iter
        self.init_method = init_method
        self.distance_metric = distance_metric
        self.random_state = random_state
        self.tol = tol 
        self.centroids = None
        self.labels = None

    def initialiaze_centroids(self, X):
        """
        Initialise les centroïdes selon la stratégie spécifiée.

        Stratégies disponibles :
        - 'forgy' : Sélectionne aléatoirement k points distincts du dataset.
        - 'random_partition' : Assigne chaque point à un cluster au hasard et calcule les moyennes.
        - 'kmeans++' : Sélectionne des points distants les uns des autres pour accélérer la convergence.

        Args:
            X (np.ndarray): Matrice de données de forme (n_samples, n_features).

        Returns:
            np.ndarray: Centroïdes initiaux de forme (k, n_features).

        Raises:
            ValueError: Si 'init_method' ne correspond pas aux méthodes implémentées.
        """
        X = np.array(X)
        np.random.seed(self.random_state)
        n_samples, n_features = X.shape
        
        if self.init_method == 'forgy':
            indices = np.random.choice(n_samples, self.k, replace=False)
            centroids = X[indices]
            return centroids
            
        elif self.init_method == 'random_partition':
            random_labels = np.random.randint(0, self.k, size=n_samples)
            centroids = np.zeros((self.k, n_features)) 
            for i in range(self.k):
                cluster_points = X[random_labels == i]
                if len(cluster_points) > 0:
                    centroids[i] = np.mean(cluster_points, axis=0)
                else:
                    centroids[i] = X[np.random.choice(n_samples)]
            return centroids
        
        elif self.init_method == 'kmeans++':
            first_index = np.random.randint(0, n_samples)
            centroids = [X[first_index]]
            for j in range(1, self.k):
                dists = np.array([min([np.sum((x - c)**2) for c in centroids]) for x in X])
                probs = dists / dists.sum()
                cumulative = np.cumsum(probs)
                r = np.random.rand()
                next_idx = np.searchsorted(cumulative, r)
                centroids.append(X[next_idx])
            return np.array(centroids)
            
        else:
            raise ValueError(f"Méthode d'initialisation '{self.init_method}' non reconnue.")

    def fit(self, X):
        """
        Entraîne le modèle sur le jeu de données X.

        L'algorithme itère entre l'assignation des points au centroïde le plus proche 
        et la mise à jour de la position des centroïdes jusqu'à convergence (stabilité) 
        ou atteinte du nombre maximum d'itérations.

        Args:
            X (np.ndarray): Données d'entraînement de forme (n_samples, n_features).

        Returns:
            self: L'instance entraînée.
        """
        X = np.array(X)
        self.centroids = self.initialiaze_centroids(X)
        
        stable_count = 0
        stable_required = 5

        for i in range(self.max_iter):
            distances = self.distance_metric(X, self.centroids)
            self.labels = np.argmin(distances, axis=1)
            
            nouveaux_centroids = np.zeros_like(self.centroids)
            for k in range(self.k):
                points_k = X[self.labels == k]
                if len(points_k) > 0:
                    nouveaux_centroids[k] = np.mean(points_k, axis=0)
                else:
                    nouveaux_centroids[k] = self.centroids[k] 

            ecart = np.max(np.sqrt(np.sum((nouveaux_centroids - self.centroids)**2, axis=1)))            
            self.centroids = nouveaux_centroids
            
            print(f"Itération {i+1}/{self.max_iter} | Écart : {ecart:.4f} | Stable : {stable_count}/{stable_required}")

            if ecart < self.tol:
                stable_count += 1
                if stable_count > stable_required:
                    print(f"\nEntraînement terminé — convergence en {i+1} itération(s)")
                    break
            else: 
                stable_count = 0
        else:
            print(f"\nEntraînement terminé — maximum de {self.max_iter} itérations atteint")

        return self

    def predict(self, X):
        """
        Assigne chaque point de X au cluster le plus proche selon les centroïdes appris.

        Args:
            X (np.ndarray): Nouvelles données de forme (n_samples, n_features).

        Returns:
            np.ndarray: Indices des clusters assignés.
        """
        X = np.array(X)
        distances = self.distance_metric(X, self.centroids)
        return np.argmin(distances, axis=1)

    def calculate_inertia(self, X):
        """
        Calcule l'inertie (Within-Cluster Sum of Squares) du modèle.

        L'inertie mesure la compacité des clusters. Plus elle est faible, plus 
        les points sont proches de leur centre.

        Args:
            X (np.ndarray): Données de forme (n_samples, n_features).

        Returns:
            float: Somme des distances au carré entre chaque point et son centroïde.
        """
        X = np.array(X)
        inertia = 0
        for k in range(self.k):
            points_k = X[self.labels == k]
            if len(points_k) > 0:
                inertia += np.sum((points_k - self.centroids[k]) ** 2)
        return inertia

    def calculate_silhouette_score(self, X):
        """
        Calcule le score de silhouette moyen du clustering.

        Le score est compris entre -1 et 1 :
        - Proche de 1 : Les points sont bien classés et loin des clusters voisins.
        - Proche de 0 : Les points sont proches de la frontière entre clusters.
        - Proche de -1 : Les points sont probablement assignés au mauvais cluster.

        Args:
            X (np.ndarray): Données de forme (n_samples, n_features).

        Returns:
            float: Moyenne des coefficients de silhouette de tous les échantillons.

        Raises:
            ValueError: Si le modèle n'a pas été entraîné ou si k < 2.
        """
        X = np.array(X)
        
        if self.labels is None or self.centroids is None:
            raise ValueError("Le modèle doit être entraîné avant le calcul du silhouette score.")
        if self.k < 2:
            raise ValueError("Le score de silhouette nécessite au moins 2 clusters.")
        
        n_samples, n_features = X.shape
        silhouette_scores = np.zeros(n_samples)
        
        for i in range(n_samples):
            cluster_i = self.labels[i]
            same_cluster = X[self.labels == cluster_i]
            
            if len(same_cluster) <= 1:
                silhouette_scores[i] = 0
                continue
                
            dists_same = np.sqrt(np.sum((same_cluster - X[i])**2, axis=1))
            a_i = np.sum(dists_same) / (len(same_cluster) - 1)
            
            b_i = np.inf
            for j in range(self.k):
                if j == cluster_i:
                    continue
                other_cluster = X[self.labels == j]
                if len(other_cluster) == 0:
                    continue
                dists_other = np.sqrt(np.sum((other_cluster - X[i])**2, axis=1))
                mean_dist_j = np.mean(dists_other)
                if mean_dist_j < b_i:
                    b_i = mean_dist_j
                
            silhouette_scores[i] = (b_i - a_i) / max(a_i, b_i)
                
        return np.mean(silhouette_scores)