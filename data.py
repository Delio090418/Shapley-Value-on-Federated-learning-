import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
import random
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import copy
import math
from itertools import combinations
from itertools import permutations
from sklearn import preprocessing


random.seed(5)
np.random.seed(5)
torch.manual_seed(5)

def make_client_points():
    blobs_specs = [
        {
            "centers": [1, 1],
            "cluster_std": .1,
        },
        {
            "centers": [-1, 1],
            "cluster_std": .1,
        },
        {
            "centers": [-1, -1],
            "cluster_std": .1,
        },
        {
            "centers": [1, -1],
            "cluster_std": .1,
        },
        
    ]

    clients_specs = [
        
         [500,0,0, 500],
         [500,0,0, 500],
         [0,500,0, 500],
         [0,0,500, 500],
         [500,500,500, 500]
        # [300, 10, 500, 0],
        # #[100, 100, 100, 0],
        # #[100, 100, 100, 0],
        # [300, 10, 500, 10],
        # #[150, 50, 10, 1],
        # [10, 300, 10, 500],
        # [400, 400, 400, 400],
        

    ]

    centers = []
    cluster_std = []
    for label in range(len(blobs_specs)):
        centers.append(blobs_specs[label]["centers"])
        cluster_std.append(blobs_specs[label]["cluster_std"])

    clients_data = []
    for n_samples in clients_specs:
        X, y = make_blobs(
            n_samples=n_samples,
            n_features=2,
            centers=centers,
            cluster_std=cluster_std,
        )
        #X=preprocessing.normalize(X,axis=0)
        clients_data.append([torch.from_numpy(X.astype(np.float32)),torch.LongTensor(y.astype(np.float32))])

    return clients_data

if __name__ == "__main__":
    clasif=make_client_points()
    for client in range(len(clasif)):
        blobs, blob_labels = clasif[client]
        plt.scatter(blobs[:, 0], blobs[:, 1], c=blob_labels)
        plt.show()

    ####Compare the weights between the clients 0 and 1.