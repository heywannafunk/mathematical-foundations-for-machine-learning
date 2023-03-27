import numpy as np
from sklearn.datasets import make_blobs
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
import pandas as pd

def knn_crossdimensional_cv(k_neighbours:list=[3,7,15], k_folds:int=10, 
                            d_list:list=[2,100,200,300,400,500], d_range:tuple=None) -> dict:
    
    result = {}
    for k in k_neighbours:
        result[k] = np.empty((0,))
        
    for d in d_list:
        mu1 = np.full((d,), 2/np.sqrt(d))
        mu2 = np.full((d,), -2/np.sqrt(d))
        mu = np.vstack([mu1, mu2])
        
        X, y = make_blobs(n_samples=10000, centers=mu, cluster_std=1., n_features=d, random_state=0)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
        
        # Создание экземпляра классификатора KNeighborsClassifier
        knn = KNeighborsClassifier()
        
        # Определение диапазона значений гиперпараметров для GridSearchCV
        param_grid = {'n_neighbors': [3, 7, 15]}
        
        # Создание объекта GridSearchCV
        grid = GridSearchCV(knn, param_grid, cv=k_folds)
        # Обучение объекта GridSearchCV на обучающих данных
        grid.fit(X_train, y_train)
        
        # Фиксация результата
        results_df = pd.DataFrame(grid.cv_results_)
        for key in result:
            result[key] = np.append(result[key], results_df[results_df['param_n_neighbors'] == key]['mean_test_score'].iloc[0])
        
    return result