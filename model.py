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


class MLP(nn.Module):
    def __init__(self, neurons, id=0):  #'self' is required
        super(MLP, self).__init__()  # <- This is required

        # Define all layers as a special list
        self.layers = nn.Sequential()
        #normalizer = nn.BatchNorm1d(neurons[0], affine=False, track_running_stats=False)
        # normalizer = nn.LayerNorm(neurons[0])
        #self.layers.append(normalizer)
        # This way, we just iteratively append all layers
        for i in range(len(neurons)-2):
            self.layers.append(nn.Linear(neurons[i], neurons[i+1]))
            self.layers.append(nn.ReLU())
        
        # Unfortunately, the last layer has a different activation function
        #(an 'if' statement could alse be introduced in the above loop)
        self.layers.append(nn.Linear(neurons[i+1], neurons[i+2]))
        self.layers.append(nn.Softmax(dim=1))
       
        


        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.01, weight_decay=0.001)

        self.id= id

    def forward(self, x):
        # x = self.normalizer(x)
        x = self.layers(x)
        return x
    
    def fit(self,X,y,num_epochs = 100, plot=False):
        # Training loop
        #history of the loss
        loss_data = []
        accuracy_data = []
        for epoch in range(num_epochs):
            self.train()
            self.optimizer.zero_grad()
            output = self(X)
            loss = self.criterion(output, y)
            loss.backward()
            self.optimizer.step()
            loss_data.append(loss.detach().item())
            accuracy = self.evaluation(X, y)
            accuracy_data.append(accuracy)
        
        if plot:
            self.plot_data(loss_data, title=f"loss for client{self.id}")
            self.plot_data(accuracy_data, title=f"accuracy for cleint{self.id}")

        # plt.plot(loss_data)
        # plt.show()
            #print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item()}')

    def plot_data(self, data, title=""):
        plt.figure()
        plt.plot(data)
        plt.title(title)
        #plt.show()


    @torch.no_grad()
    def evaluation(self,X,y):
        self.eval()
        output = self(X)
        predicted_labels = torch.argmax(output, dim=1)
        correct = (predicted_labels == y).sum().item()
        accuracy = correct / len(y) * 100
        return accuracy
    
    def get_model_grads(self):
        # client_parameters = self.parameters()
        # return client_parameters.grad
        return [layer.grad for layer in self.layers]
    
    
    def reset_parameters(self):
        # self.normalizer.reset_parameters()
        # self.layers.reset_parameters()
        for layer in self.children():
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()
    
   
    
if __name__ == "__main__":
    neurons = [2,15,4]
    clasif=make_client_points()

    # Create an instance of the MLP
    mlp = MLP(neurons)
    mlp.fit(clasif[1][0],clasif[1][1],200, plot=True)
    eva=mlp.evaluation(clasif[len(clasif)-1][0],clasif[len(clasif)-1][1])
    print(eva)


