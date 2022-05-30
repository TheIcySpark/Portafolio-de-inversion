import random
import csv
import pandas as pd
import numpy as np
from datetime import datetime


class Chromosome:
    def __init__(self):
        self.genes = np.array(self.generateRandomGenes(10))

    def generateRandomGenes(self, n: int) -> [float]:
        randomValues: [float] = [random.random() for x in range(n)]
        sumatory: float = sum(randomValues)
        return [x / sumatory for x in randomValues]


class Population:
    def __init__(self):
        self.chromosome = []
        self.fitness = []
        for i in range(10):
            self.chromosome.append(Chromosome())
            self.fitness.append(0)

    def generateRandomPopulation(self):
        for chromosome in self.chromosome:
            chromosome.randomizeChromosome()

    def computeFitness(self):
        pass

    def printChromosomes(self):
        for chromosome in self.chromosome:
            chromosome.printChromosome()


class PortfolioData:
    def __init__(self):
        self.assets: pd.DataFrame() = pd.read_csv('Portafolio.csv')
        # self.returnsAssets =

    def computeAssetsReturns(self, assets):
        pass


if __name__ == '__main__':
    portfolio: PortfolioData = PortfolioData()
    print(portfolio.assets.hist())
