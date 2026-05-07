import numpy as np
from collections import Counter


def euclidian_distance (x1,x2):
    distance = np.sqrt(np.sum((x1 - x2)**2))
    return distance

def manhattan_distance(x1,x2):
    distance = np.sum(abs(x1 - x2))
    return distance



    
class KNN:
    
    Metrics = {
        "euclidian": euclidian_distance,
        "manhattan": manhattan_distance
    }
    Tasks = ["Classification","LinearRegression"]
    
   
    def __init__(self,task,k=3,distance_metric = "euclidian"):
        
        if task not in self.Tasks:
            raise ValueError(
                f"'{task}' n'est pas une tâche valide que cet algorithme peut réaliser"
                f"Veuillez choisir parmi les tâches suivantes : {self.Tasks[0 : ]}"
            )
            
        else :
            self.task = task
        
        self.k = k
        
        #Vérification de l'existence de la métrique dans le tableau des métriques disponibles
        
        if distance_metric not in self.Metrics:
            options = ",".join(self.Metrics.keys())
            raise ValueError(
                f"'{distance_metric}' n'est pas une métrique valide \n"
                f"\n Veuillez choisir parmi les métriques suivantes : {options}"
            )
        else:
            self.distance_metric = self.Metrics[distance_metric]
        
        
    
    
    def fit(self,X,y):
        self.X_train = X
        self.y_train = y
        
    
    def predict(self,X):
        print("Début des prédictions")
        
        self.X_test = X
        predictions = [self._predict(x) for x in X]
        
        return predictions
        
        print("Fin des prédictions ....")
    
    
    def _predict(self,x):
        
        #Calculer les distances entre chaque donnée de test et les données d'entrainement
        print("Calcul des distances...")
        
        distances = [self.distance_metric(x,x_train) for x_train in self.X_train]
    
        #Obtention des k plus proches voisins
        
        k_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = [self.y_train[i] for i in k_indices]

        #Obtention de la valeur prédite selon le cas de tache 
        if self.task == "Classification":
            most_common = Counter(k_nearest_labels).most_common()
            return most_common[0][0]
        
        else :
            predicted_value = np.sum(k_nearest_labels)/len((k_nearest_labels))
            return predicted_value
            

