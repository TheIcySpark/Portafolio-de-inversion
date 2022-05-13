import random
import csv
from datetime import datetime


class Chromosome:
	def __init__(self):
		self.gen = [0] * 10

	
	def randomizeChromosome(self):
		i = 0
		remainingInversion = 1
		while i < len(self.gen):
			self.gen[i] = random.uniform(0, remainingInversion)
			remainingInversion -= self.gen[i]
			i += 1
		self.gen[-1] = remainingInversion

	
	def printChromosome(self):
		for gen in self.gen:
			print(str(round(gen, 4)), end=' ')
		print()

	

	
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
		self.assetsNames = []
		self.assetsValuesByMonth = []
		self.assetsReturn = []
		self.assetsReturnProm = []


	def fetchFileData(self):
		with open('Portafolio.csv') as csvPortfolioData:
			csvReader = csv.reader(csvPortfolioData, delimiter = ',')
			lineCount = 0
			for row in csvReader:
				if lineCount == 0:
					self.assetsNames = row
				else:
					self.assetsValuesByMonth.append(row)


	def computeAssetsReturn(self):
		i = 0
		while i < len(self.assetsValuesByMonth):
			j = 1
			self.assetReturns.append([])
			while j < len(self.assetsValuesByMonth[i]):
				self.assetsReturn[i].append((self.assetsValuesByMonth[i][j] - self.assetsValuesByMonth[i][j - 1])
											/ self.assetsValuesByMonth[i][j - 1])
				j += 1
			i += 1
	
	def computeAssetsReturnProm(self):
		for assetReturn in self.assetsReturn:
			self.assetsReturnProm.append([sum(x)/len(x) for x in self.assetsReturn[i]])
	

	def fetchAndComputeData(self):
		self.fetchFileData()
		self.computeAssetsReturn()
		self.computeAssetsReturnProm()



if __name__ == '__main__':
	portfolioData = PortfolioData()
	portfolioData.fetchAndComputeData()
	print(portfolioData.assetsNames)
	print(portfolioData.assetsValuesByMonth)
		