import matplotlib.pyplot as plt
import random
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import copy
import math
from sklearn import preprocessing
from data import make_client_points
from model import MLP

random.seed(5)
np.random.seed(5)
torch.manual_seed(5)

class Clientclasi:
    def __init__(self, neurons, X=None, y=None, id=0) -> None:
        super( ).__init__() 
        self.neurons=neurons
        self.model = MLP(neurons, id)

        self.id= id

    
        self.local_train_X, self.local_train_y = (X, y)
    
        
    def fit(self,num_epochs = 200, plot=False):
        self.model.fit(self.local_train_X, self.local_train_y,num_epochs, plot=plot)

    def evaluation(self,X,y):
        return self.model.evaluation(X, y)
    
    def get_model_grads(self):
        self.model.get_model_grads()
        
    def state_dict(self):
        return self.model.state_dict()
    
    def reset_parameters(self):
        self.model.reset_parameters()

    def load_state_dict(self, weights):
        self.model.load_state_dict(weights)
    
    def parameters(self):
        return self.model.parameters()
    

class Federation(Clientclasi):
    def __init__(self, neurons, num_clientes) -> None:
        super( ).__init__(neurons) 
        self.neurons=neurons
        self.num_clientes=num_clientes
        self.modelo = MLP(neurons)
        
        
        self.datos=make_client_points()
        self.clients_points = self.datos[:self.num_clientes]
        self.X_test, self.y_test = self.datos[self.num_clientes]

        self.lista_clientes = [Clientclasi(neurons,self.clients_points[i][0], self.clients_points[i][1]) for i in range(len(self.clients_points))]
    
    def clientess(self):
        return self.lista_clientes

    def aggregate(self,List_clients):
        pesos_cliente = [client.parameters() for client in List_clients]            
        data = []
        for params_clients in zip(*pesos_cliente):
            data.append(sum(param.data for param in params_clients)/len(List_clients))
        return data
    
    def set_parameters(modelo, params):
        for model_parametro, param in zip(modelo.parameters(), params):
            model_parametro.data = param

    def federated_averaging(self,num_iterations,conjunto_clientes, num_epochs=20, plot=False):
        conj_clientes=[self.lista_clientes[i] for i in conjunto_clientes]
        iteration = 0

        acuuracy_data = []
        while iteration <=num_iterations-1:
            
            for client in conj_clientes:
                weights_modelo=self.modelo.state_dict()
                client.load_state_dict(weights_modelo)
                client.fit(num_epochs=num_epochs)
                
            params = self.aggregate(conj_clientes)
            Federation.set_parameters(self.modelo, params)

            if plot:
                accuracy = self.evaluacion()
                acuuracy_data.append(accuracy)

            iteration += 1

        if plot:
            self.plot_data(acuuracy_data, title=f"accuracy for cleints {conjunto_clientes}")

        return self.modelo.parameters
    
    def evaluacion(self):
        return self.modelo.evaluation(self.X_test, self.y_test)
    
    def plot_data(self, data, title=""):
        self.modelo.plot_data(data, title)
    
    
if __name__ == "__main__":
    neurons = [2,15,4]
    clasif=make_client_points()
    
    federacion=Federation(neurons, 4)
    federacion.federated_averaging(5,[0,2,3], 15, plot=True)
    evalu=federacion.evaluacion()
    people=federacion.clientess()

    #print(people[2].evaluation(clasif[len(clasif)-1][0],clasif[len(clasif)-1][1]))
    print(evalu)
    #plt.show()
        