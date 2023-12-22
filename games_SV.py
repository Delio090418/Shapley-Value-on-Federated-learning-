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

from data import make_client_points
from model import MLP

from main import Federation
from main import Clientclasi

random.seed(5)
np.random.seed(5)
torch.manual_seed(5)

class games_and_values(Federation):
    def __init__(self,neurons, num_clientes) -> None:
        super( ).__init__(neurons, num_clientes)
        self.number_clients=num_clientes
        self.clientes=list(range(self.number_clients))

    


    def game_last_round(self, conjunto_clientes, model=None):
        if len(conjunto_clientes) == 0:
            return 0
        conj_clientes=[self.lista_clientes[i] for i in conjunto_clientes]

        if model is None:
            model = MLP(neurons)
        params = self.aggregate(conj_clientes)
        Federation.set_parameters(model, params)
        #predictions= model(torch.from_numpy(self.X_test))#predictions= self.model(self.X_test)
        worth = model.evaluation(self.X_test, self.y_test)
        return worth
    
    def retrainin_game(self,con_clientes, iterations, epochs=100):
        if len(con_clientes) == 0:
            return 0
        self.modelo.reset_parameters()
        self.federated_averaging(iterations,con_clientes,num_epochs=epochs)
        #predictions= self.model(torch.from_numpy(self.X_test))
        worth = self.evaluacion()
        return worth
    
    def Shapley_value_last_round(self, iteration=1):
        self.federated_averaging(iteration,[0,1,2])
        self.number_clients =len(self.clients_points)
        S = np.zeros(self.number_clients)

        model = MLP(neurons)
        for client_idx in range(self.number_clients):
            other_clients_idx = list(range(self.number_clients))
            other_clients_idx.remove(client_idx)
            #data = []
            for subset_len in range(len(other_clients_idx)+1):
                ss = 0
                coef = math.comb(self.number_clients - 1, subset_len)
                for subset_idx in combinations(other_clients_idx, subset_len):
                    a = (self.game_last_round(list(subset_idx)+[client_idx], model)-self.game_last_round(list(subset_idx), model))/coef
                    ss += a
                S[client_idx] += ss
            S[client_idx] /= self.number_clients  
        return S
    
    def Shapley_value_retraing(self, lista_clientes,iteration=1, epochs=100):
        self.lis_clientes=lista_clientes
        self.number_clients =len(self.clients_points)
        S = np.zeros(self.number_clients)
        for client_idx in self.lis_clientes:
            other_clients_idx = [j for j in self.lis_clientes if j!=client_idx]

            # other_clients_idx = self.lis_clientes
            # other_clients_idx.remove(client_idx)
     
            for subset_len in range(len(other_clients_idx)+1):
                ss = 0
                coef = math.comb(len(self.lis_clientes) - 1, subset_len)
                for subset_idx in combinations(other_clients_idx, subset_len):
                    a = (self.retrainin_game(list(subset_idx)+[client_idx], iteration, epochs=epochs)-self.retrainin_game(list(subset_idx),iteration, epochs=epochs))/coef
                    ss += a
                S[client_idx] += ss
            S[client_idx] /= len(self.lis_clientes)
        return S

    
    def coalitional_values(self, iteration=1,epochs=100):
        d = dict()
        for subset_len in range(self.number_clients+1):
            for subset_idx in combinations(self.clientes, subset_len):
                    d[subset_idx] = self.retrainin_game(list(subset_idx), iteration,epochs=epochs)
        return d
    

if __name__ == "__main__":
    neurons=[2,15,4]
    modelo_federation=games_and_values(neurons,4)
    retra=modelo_federation.Shapley_value_retraing([0,1,2,3],10, 30)#something is offfffffffffff
    print(retra)
   
      
    
