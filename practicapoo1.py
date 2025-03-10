# -*- coding: utf-8 -*-
"""
Editor de Spyder

Este es un archivo temporal.
"""

from abc import ABC, abstractclassmethod
import numpy as np
import sklearn.datasets

iris = sklearn.datasets.load_iris() #diccionari
print(iris.DESCR)

X, y = iris.data, iris.target

ratio_train, ratio_test = 0.7, 0.3

num_samples, num_features = X.shape

idx = np.random.permutation(range(num_samples))
#devuelve lista números ordenador aleatoriamente 

num_samples_train = int(num_samples*ratio_train)
num_samples_test = int(num_samples*ratio_test)

idx_train = idx[:num_samples_train]
idx_test = idx[num_samples_train: num_samples_train+num_samples_test]
X_train, y_train = X[idx_train], y[idx_train]
X_test, y_test = X[idx_test], y[idx_test]

class RandomForestClassifier():
    def __init__(self, num_trees: int, min_size: int, max_depth: int, ratio_samples:float, num_random_features: float, criterion: str):
        if not isinstance(num_trees, int):
            raise TypeError("Num_trees debe ser un entero")
        self._num_trees = num_trees
        if not isinstance(min_size, int):
            raise TypeError("min_size debe ser un entero")
        self._min_size = min_size
        if not isinstance(max_depth, int):
            raise TypeError("max_depth debe ser un entero")
        self._max_depth = max_depth
        if not isinstance(ratio_samples, float):
            raise TypeError("Ratio_samples debe ser un float")
        self._ratio_samples = ratio_samples
        if not isinstance(num_random_features, float):
            raise TypeError("Num_random_features debe ser un float")
        self._num_random_features = num_random_features
        if not isinstance(criterion, str):
            raise TypeError("Criterion debe ser un str")
        self._criterion = criterion
        
    def fit(self, X: float[][], y: float[]):
    
    def predict(self, X: float[][], y: float[]):

class Node(ABC):  
    @abstractclassmethod
    def predict(self, X: float[][], y: float[]):
        """This method must be implemented by all subclasses."""
        pass

class Leaf(Node):
    def __init__(self, label: int):
        if not isinstance(label, int):
            raise TypeError("Label debe tener un valor entero")
        self._label = label
        
        
    
class Parent(Node):
    def __init__(self, feature_index: int, threshold: float):
        if not isinstance(feature_index, int):
            raise TypeError("Feature_index debe tener un valor entero")
        self._feature_index = feature_index
    
    


if __name__=="__main__":
    max_depth = 10 # maximum number of levels of a decision tree
    min_size_split = 5 # if less, do not split a node
    ratio_samples = 0.7 # sampling with replacement
    num_trees = 10 # number of decision trees
    num_random_features = int(np.sqrt(num_features))
    # number of features to consider at
    # each node when looking for the best split
    criterion = 'gini'

