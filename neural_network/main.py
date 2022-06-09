import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import random


class NeuralNetwork:
    def __init__(self):
        # partidos ganados, partidos perdidos, puntos a favor, puntos en contra, resultado
        self.inputs = [0 for i in range(4)]
        self.weights = [1 for i in range(4)]

    def set_values(self, inputs):
        self.inputs = inputs

    def train(self, inputs, expected_output):
        self.inputs = inputs
        if expected_output != self.compute_output_with_activation_function(self.weights):
            # Algoritmo genetico
            population = [[random.uniform(-1, 1), random.uniform(-1, 1), random.uniform(-1, 1),
                           random.uniform(-1, 1)] for i in range(10)]
            for n in range(50):
                # ordenar
                if expected_output == 1:
                    population = [x[1] for x in sorted(zip([self.compute_output(i) for i in population],
                                         population), key=lambda x: x[0], reverse=True)]
                else:
                    population = [x[1] for x in sorted(zip([self.compute_output(i) for i in population],
                                         population), key=lambda x: x[0])]
                #Generar hijos
                children = []
                for i in range(1, 5):
                    children1, children2 = [0 for i in range(4)], [0 for i in range(4)]
                    alpha = random.random()
                    for j in range(4):
                        children1[j] = population[0][j] * (1 - alpha) + population[i][j] * alpha
                        children2[j] = population[i][j] * (1 - alpha) + population[0][j] * alpha
                    children.append(children1)
                    children.append(children2)
                # remplazar
                population[4:] = children
                # ordenar
                if expected_output == 1:
                    population = [x[1] for x in sorted(zip([self.compute_output(i) for i in population],
                                                           population), key=lambda x: x[0], reverse=True)]
                else:
                    population = [x[1] for x in sorted(zip([self.compute_output(i) for i in population],
                                                           population), key=lambda x: x[0])]
            # Seleccionar a que peso afectar
            if(self.compute_output_with_activation_function([self.weights[0], self.weights[1],
                                                             self.weights[2], population[0][3]])) == expected_output:
                self.weights[3] = population[0][3]
            elif(self.compute_output_with_activation_function([self.weights[0], self.weights[1],
                                                             population[0][2], self.weights[3]])) == expected_output:
                self.weights[2] = population[0][2]
            elif(self.compute_output_with_activation_function([self.weights[0], population[0][1],
                                                             self.weights[2], self.weights[3]])) == expected_output:
                self.weights[1] = population[0][1]
            elif (self.compute_output_with_activation_function([population[0][0], self.weights[1],
                                                                self.weights[2], self.weights[3]])) == expected_output:
                self.weights[0] = population[0][0]
            else:
                self.weights = population[0]


    def compute_output_with_activation_function(self, weights):
        return 1 if self.compute_output(weights) > 0 else 0

    def compute_output(self, weights):
        output = 0
        for index in range(len(self.inputs)):
            output += self.inputs[index] * weights[index]
        return output





if __name__ == "__main__":
    games = pd.read_csv('partidos.csv')
    neural_network: NeuralNetwork = NeuralNetwork()
    games_np = games.to_numpy()
    for i in range(20):
        neural_network.train(games_np[i, 1:5], games_np[i, 5])
    for i in range(21, 30):
        neural_network.set_values(games_np[i, 1:5])
        print(neural_network.compute_output_with_activation_function(neural_network.weights))
        print(games_np[i, 5])
        print()

