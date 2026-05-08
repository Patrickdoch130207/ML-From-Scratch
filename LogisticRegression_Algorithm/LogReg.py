import numpy as np

def sigmoid(x):
    """
    Applique la fonction sigmoïde élément par élément.

    Transforme toute valeur réelle en une probabilité comprise entre 0 et 1,
    ce qui permet d'interpréter la sortie du modèle comme une probabilité de
    classe.

    Args:
        x (np.ndarray): Valeurs d'entrée, typiquement la prédiction linéaire
                        du modèle.

    Returns:
        np.ndarray: Valeurs transformées, comprises dans l'intervalle (0, 1).
    """
    result = 1 / (1 + np.exp(-x))
    return result


class LogisticRegression():
    """
    Régression logistique binaire entraînée par descente de gradient batch.

    Le modèle apprend un vecteur de poids et un biais en minimisant la
    binary cross-entropy via des mises à jour itératives des paramètres.

    Attributes:
        lr (float): Taux d'apprentissage (learning rate).
        n_iters (int): Nombre d'itérations de la descente de gradient.
        weights (np.ndarray): Poids appris, de forme (n_features,).
        biais (float): Biais appris.
    """

    def __init__(self, lr=0.001, n_iters=1000):
        """
        Initialise les hyperparamètres du modèle.

        Args:
            lr (float): Taux d'apprentissage. Par défaut 0.001.
            n_iters (int): Nombre d'itérations d'entraînement. Par défaut 1000.
        """
        self.lr = lr
        self.n_iters = n_iters
        self.weights = None
        self.biais = None

    def fit(self, X, y):
        """
        Entraîne le modèle sur les données X et les étiquettes y.

        À chaque itération, la méthode effectue un forward pass pour calculer
        les probabilités prédites, puis un backward pass pour calculer les
        gradients de la loss par rapport aux poids et au biais. Les paramètres
        sont ensuite mis à jour dans le sens opposé au gradient.

        Args:
            X (np.ndarray): Matrice des features d'entraînement, de forme
                            (n_samples, n_features).
            y (np.ndarray): Vecteur des étiquettes binaires (0 ou 1), de forme
                            (n_samples,).

        Returns:
            self: L'instance entraînée.
        """
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.biais = 0

        for _ in range(self.n_iters):
            # Forward pass
            linear_pred = np.dot(X, self.weights) + self.biais
            predictions = sigmoid(linear_pred)

            # Calcul des gradients
            error = predictions - y
            dw = (1 / n_samples) * np.dot(X.T, error)
            db = (1 / n_samples) * np.sum(error)

            # Mise à jour des paramètres
            self.weights -= self.lr * dw
            self.biais   -= self.lr * db

        return self

    def predict(self, X):
        """
        Prédit la classe binaire pour chaque observation de X.

        Calcule la probabilité via le modèle linéaire et la fonction sigmoïde,
        puis applique un seuil à 0.5 pour produire les étiquettes finales.

        Args:
            X (np.ndarray): Matrice des features à prédire, de forme
                            (n_samples, n_features).

        Returns:
            list[int]: Liste des classes prédites (0 ou 1) pour chaque
                       observation.
        """
        linear_pred = np.dot(X, self.weights) + self.biais
        y_pred = sigmoid(linear_pred)
        class_pred = [0 if y <= 0.5 else 1 for y in y_pred]
        return class_pred