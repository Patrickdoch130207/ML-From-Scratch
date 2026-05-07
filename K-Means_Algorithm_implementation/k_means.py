import numpy as np

def euclidian_distance (X,centroids):
    """
    Calcule la distance euclidienne entre chaque point et chaque centroïde.
    Utilise le broadcasting NumPy pour optimiser le calcul.
    
    """
    distances = np.sqrt(np.sum((X[:,np.newaxis,:] - centroids)**2, axis=2))
    return distances
    

def manhattan_distance(X,centroids):
    distance = np.sum(abs(X[:,np.newaxis,:]-centroids),axis=2)
    return distance


class KMeans:
    """
    Implémentation de l'algorithme de clustering K-Means.
    
    Attributes:
        k (int): Nombre de clusters cibles.
        max_iter (int): Nombre maximal d'itérations pour la convergence.
        init_method (str): Méthode d'initialisation ('forgy' ou 'random_partition').
        distance_metric (function): Fonction de calcul de distance.
        tol (float): Tolérance pour le critère d'arrêt (convergence).
    """
    
    def __init__(self,k=2,max_iter = 100,init_method="forgy",distance_metric = euclidian_distance,tol = 0.01,random_state = None):
        self.k = k
        self.max_iter = max_iter
        self.init_method = init_method
        self.distance_metric = distance_metric
        self.random_state = random_state
        self.tol = tol 
        self.centroids = None
        self.labels = None
        
        
        
        
    def initialiaze_centroids(self,X):
        """
        Initialise les centroïdes selon la méthode choisie.
        
        **Methode 'forgy'**:
            Tire k indices aléatoires distincts (replace=False = sans remise) et retourne les points correspondants.
            Rapide et simple, mais les centroïdes peuvent se retrouver très proches 
            
        **Methode 'random_partition'**:
            Assigne aléatoirement chaque point à un cluster (0..k-1), puis prend la moyenne de chaque groupe
            comme centroïde. Le cas len == 0 (cluster vide) est géré en tirant un point aléatoire — robuste.
            
        **Methode 'kmeans++'**:
            Basé sur une distribution de probabilité cumulée et une méthode de tri par dichotomie,permettant d'avoir des clusters éloignés les un des autres
        
        Args:
            X (np.ndarray): Matrice des données d'entrée de shape(n_samples, n_features).
            
        Returns:
            np.ndarray: Centroïdes initiaux (k, n_features).
        
        **Methode 'kmeans++'**: 
        
        
        
        """
        X = np.array(X)
        np.random.seed(self.random_state)
        n_samples,n_features = X.shape
        
        if self.init_method == 'forgy':                                         # Choisit k points aléatoires existants dans X
            indices = np.random.choice(n_samples,self.k,replace=False)
            centroids = X[indices]
            return centroids
            
        
        elif self.init_method == 'random_partition':                     
            random_labels = np.random.randint(0,self.k,size=n_samples)
            centroids = np.zeros((self.k,n_features)) 
            
            for i in range (self.k):
                cluster_points = X[random_labels == i]

                if len(cluster_points) > 0 :
                    centroids[i] = np.mean(cluster_points, axis=0)
                else : 
                    centroids[i] = X[np.random.choice(n_samples)]
                
            return centroids
        
        elif self.init_method == 'kmeans++':
            first_index = np.random.randint(0,n_samples)
            centroids = [X[first_index]] #Choix totalement aléatoire du premier centroide
            
            for j in range(1,self.k):
                # Version correcte
                dists = np.array([min([np.sum((x - c)**2) for c in centroids]) for x in X]) # Calcul de la distance au carré par rapport au centroide le plus proche
                probs = dists / dists.sum() #Ramène chaque distance à une échelle de probabilité. Plus la proba est forte plus le point est loin du centroide le plus proche de lui
                cumulative = np.cumsum(probs) # Permet d'ordonner les probas de facon ce croissante(fonction de répartition)
                r=np.random.rand()
                next_idx = np.searchsorted(cumulative, r) #utilise le tri  dichotomique et renvoie l'index 
                centroids.append(X[next_idx])
            
            return np.array(centroids)
            
        else : 
            raise ValueError(f"Cette méthode d'initiation de centroides n'est pas définie.")
            
            
            
    
    
    def fit(self,X):
        """
        Boucle principale d'entraînement. Quatre étapes par itération 
        Calcul des distances : distance_metric(X, centroids) → matrice (n, k)
        Assignation : argmin sur axis=1 → label du centroïde le plus proche pour chaque point
        Mise à jour des centroïdes : moyenne des points de chaque cluster
        Convergence : si ‖nouveaux_centroids − centroids‖ < tol, on stoppe
        
        """
        X = np.array(X)
        self.centroids = self.initialiaze_centroids(X)
        
        stable_count = 0          # ← compteur d'itérations stables
        stable_required = 5       # ← nombre d'itérations stables requises

        for i in range(self.max_iter):
            distances = self.distance_metric(X,self.centroids)
            self.labels = np.argmin(distances, axis=1)
            
            
            nouveaux_centroids = np.zeros_like(self.centroids)
            for k in range(self.k):
                points_k = X[self.labels == k]
                if len(points_k) > 0:
                    nouveaux_centroids[k] = np.mean(points_k, axis=0)
                else:
                    nouveaux_centroids[k] = self.centroids[k] 
            # 4. Condition d'arrêt (Convergence)
            ecart = np.max(np.sqrt(np.sum((nouveaux_centroids - self.centroids)**2, axis=1)))            
            self.centroids = nouveaux_centroids
            
            # ─── En cours ───
            print(f"Itération {i+1}/{self.max_iter} | Écart : {ecart:.4f} | Stable : {stable_count}/{stable_required}")

            if ecart < self.tol:
                stable_count += 1
                if stable_count > stable_required:
                    print(f"\n Entraînement terminé — convergence en {i+1} itération(s)")
                    break
            else: 
                stable_count = 0
        else:
            print(f"\n Entraînement terminé — maximum de {self.max_iter} itérations atteint")

        
        return self

    
    def predict(self, X):
        """Assigne de nouveaux points aux clusters déjà définis."""
        X = np.array(X)
        distances = self.distance_metric(X, self.centroids)
        return np.argmin(distances, axis=1)
    
    

    def calculate_inertia(self, X):
        """Somme des distances au carré de chaque point à son centroïde."""
        X = np.array(X)
        inertia = 0
        for k in range(self.k):
            points_k = X[self.labels == k]
            if len(points_k) > 0:
                inertia += np.sum((points_k - self.centroids[k]) ** 2)
        return inertia
    
    
    
    def calculate_silhouette_score(self,X):
        """
        Calcule le score de silhouette moyen.
    
    Pour chaque point i :
        a(i) = distance moyenne aux autres points du même cluster
        b(i) = distance moyenne minimale aux points d'un autre cluster
        s(i) = (b(i) - a(i)) / max(a(i), b(i))
    
    Score global = moyenne de tous les s(i)
    Interprétation : proche de 1 → bon clustering, proche de -1 → mauvais
    
    Args:
        X (np.ndarray): Données d'entrée de shape (n_samples, n_features).
    
    Returns:
        float: Score de silhouette moyen entre -1 et 1.
        """
        X = np.array(X)
        
        if self.labels is None or self.centroids is None:
            raise ValueError("Le modèle doit etre entrainé avant le calcul du silhouette score. Utilisez .fit()")
        if self.k < 2:
            raise ValueError("Le score de silhouette nécessite au moins 2 clusters.")
        
        n_samples,n_features = X.shape
        
        silhouette_scores = np.zeros(n_samples)
        
        for i in range(n_samples):
            cluster_i = self.labels[i]
            
            #a(i):cohésion intra-cluster
            
            same_cluster = X[self.labels == cluster_i]
            
            if len(same_cluster) <= 1:
                silhouette_scores[i] = 0
                continue
                
            dists_same = np.sqrt(np.sum((same_cluster - X[i])**2,axis=1))
            a_i = np.sum(dists_same) / (len(same_cluster) - 1)  # Exclure le point lui-même (distance = 0)
            
            #b(i);séparation inter-cluster
            b_i = np.inf #recoit l'infini positif pour faciliter la comparaison
            for j in range(self.k):
                if j == cluster_i:
                    continue # On saute le cluster du point
                
                other_cluster = X[self.labels == j]
                if len(other_cluster) == 0:
                    continue
                
                dists_other = np.sqrt(np.sum((other_cluster - X[i])**2,axis=1))
                mean_dist_j = np.mean(dists_other)
                
                if mean_dist_j < b_i:
                    b_i = mean_dist_j
                
            # s(i)
            silhouette_scores[i] = (b_i - a_i)/max(a_i,b_i)
                
        return np.mean(silhouette_scores)