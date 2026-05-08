import numpy as np
from collections import Counter

def euclidian_distance(x1, x2):
    """Calcule la distance euclidienne entre deux vecteurs.

    Args:
        x1 (np.ndarray): Premier vecteur de données.
        x2 (np.ndarray): Deuxième vecteur de données.

    Returns:
        float: La distance géométrique entre les deux points.
    """
    distance = np.sqrt(np.sum((x1 - x2)**2,axis = 1 ))
    return distance

def manhattan_distance(x1, x2):
    """Calcule la distance de Manhattan entre deux vecteurs.

    Args:
        x1 (np.ndarray): Premier vecteur de données.
        x2 (np.ndarray): Deuxième vecteur de données.

    Returns:
        float: La somme des valeurs absolues des différences de coordonnées.
    """
    distance = np.sum(np.abs(x1 - x2),axis = 1)
    return distance

class KNN:
    """Implémentation de l'algorithme des K-Plus Proches Voisins (K-Nearest Neighbors).

    Cette classe permet de réaliser des tâches de classification ou de régression
    en utilisant différentes métriques de distance.

    Attributes:
        task (str): Le type de tâche à réaliser ('Classification' ou 'LinearRegression').
        k (int): Le nombre de voisins à considérer.
        distance_metric (function): La fonction utilisée pour calculer la distance.
        X_train (np.ndarray): Les données d'entraînement.
        y_train (np.ndarray): Les étiquettes ou valeurs cibles d'entraînement.
    """
    
    Metrics = {
        "euclidian": euclidian_distance,
        "manhattan": manhattan_distance
    }
    Tasks = ["Classification", "LinearRegression"]
    
    def __init__(self, task, k=3, distance_metric="euclidian"):
        """Initialise le modèle avec vérification des paramètres."""
        if task not in self.Tasks:
            raise ValueError(f"Tâche invalide. Choisissez parmi : {self.Tasks}")
        self.task = task
        self.k = k
        
        if distance_metric not in self.Metrics:
            raise ValueError(f"Métrique invalide. Choisissez parmi : {list(self.Metrics.keys())}")
        self.distance_metric = self.Metrics[distance_metric]

    def fit(self, X, y):
        """Mémorise les données d'entraînement.

        Args:
            X (np.ndarray): Matrice des caractéristiques d'entraînement.
            y (np.ndarray): Vecteur des cibles (labels ou valeurs numériques).
        """
        self.X_train = np.array(X)
        self.y_train = np.array(y)
        
    def predict(self, X):
        """Prédit les classes ou les valeurs pour un ensemble de données.

        Args:
            X (np.ndarray): Matrice des données à prédire.

        Returns:
            list: Liste des prédictions générées pour chaque échantillon de X.
        """
        self.X_test = np.array(X)
        predictions = [self._predict(x) for x in X]
        return predictions
    
    def _predict(self, x):
        """Méthode interne pour prédire la valeur d'un seul échantillon.

        Args:
            x (np.ndarray): Un seul vecteur de données.

        Returns:
            int/float: La classe majoritaire ou la moyenne des k-voisins.
        """
        # Calculer les distances entre la donnée de test et toutes les données d'entraînement
        distances = self.distance_metric(x, self.X_train)
    
        # Obtention des indices des k plus proches voisins
        k_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = self.y_train[k_indices]

        # Logique de prédiction selon la tâche
        if self.task == "Classification":
            most_common = Counter(k_nearest_labels).most_common(1)
            return most_common[0][0]
        else :
            # Calcul de la moyenne pour la régression
            return np.mean(k_nearest_labels)