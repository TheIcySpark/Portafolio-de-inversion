import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import random


class Chromosome:
    def __init__(self):
        self.genes = np.array(self.__generate_random_genes())

    @staticmethod
    def __generate_random_genes():
        n = 10
        random_values = [random.random() for x in range(n)]
        summation = sum(random_values)
        return [x / summation for x in random_values]

    def normalize(self):
        self.genes = [x / sum(self.genes) for x in self.genes]


class PortfolioData:
    def __init__(self):
        self.assets = pd.read_csv('portfolio.csv')
        self.returns = pd.DataFrame(self.__compute_returns(), columns=self.assets.columns)
        self.average_expected_return = pd.DataFrame(self.__compute_average_expected_return(),
                                                    columns=self.assets.columns)
        self.covariance = np.cov(self.returns.to_numpy(copy=True), rowvar=False)

    def __compute_average_expected_return(self):
        returns = self.returns.to_numpy(copy=True)
        return np.array([[sum(returns[:, x]) / len(returns) for x in range(len(returns[0]))]])

    def __compute_returns(self):
        assets = self.assets.to_numpy(copy=True)
        returns = self.assets.to_numpy(copy=True)
        r = 1
        while r < len(returns):
            c = 0
            while c < len(returns[r]):
                returns[r][c] = (assets[r][c] - assets[r - 1][c]) / assets[r - 1][c]
                c += 1
            r += 1
        return returns[1:, ]


class Population:
    def __init__(self):
        self.chromosomes = np.array([Chromosome() for i in range(100)])
        self.fitness = np.array([])

    def compute_risk(self, chromosome, covariance):
        risk = 0
        for i in range(0, len(chromosome.genes)):
            for j in range(0, len(chromosome.genes)):
                risk += chromosome.genes[i] * chromosome.genes[j] * covariance[i][j]
        return risk

    def compute_chromosome_return(self, chromosome, average_expected_returns):
        return sum([average_expected_returns[0][i] * chromosome.genes[i] for i in range(len(
                average_expected_returns[0]))])

    def compute_fitness(self, average_expected_returns, covariance):
        for chromosome in self.chromosomes:
            numerator = self.compute_chromosome_return(chromosome, average_expected_returns)
            denominator = self.compute_risk(chromosome, covariance)
            self.fitness = np.append(self.fitness, [numerator / denominator])

    def sort(self):
        sorted_pairs = sorted(zip(self.fitness, self.chromosomes), key=lambda x: x[0])
        self.fitness = [i[0] for i in sorted_pairs]
        self.chromosomes = [i[1] for i in sorted_pairs]

    def __cross(self, f1_index, f2_index, alpha):
        h1, h2 = Chromosome(), Chromosome()
        for i in range(len(h1.genes)):
            h1.genes[i] = \
                self.chromosomes[f1_index].genes[i] * (1 - alpha) + self.chromosomes[f2_index].genes[i] * alpha
            h2.genes[i] = \
                self.chromosomes[f2_index].genes[i] * (1 - alpha) + self.chromosomes[f1_index].genes[i] * alpha
        h1.normalize()
        h2.normalize()
        return [h1, h2]

    def __select_different_chromosome_index(self, index_list: list):
        index2 = index_list[0]
        while index2 in index_list:
            index2 = random.randint(0, len(self.chromosomes) - 1)
        return index2

    def __select_by_tournament(self):
        children = np.array([])
        for i in range(40):
            alpha = random.random()
            f1_index = random.randint(0, len(self.chromosomes) - 1)
            f2_index = self.__select_different_chromosome_index([f1_index])
            f3_index = self.__select_different_chromosome_index([f1_index, f2_index])
            sorted_index_list = sorted([f1_index, f2_index, f3_index])
            f1_index = sorted_index_list[1]
            f2_index = sorted_index_list[2]
            children = np.append(children, self.__cross(f1_index, f2_index, alpha))
        return children

    @staticmethod
    def __mutate(children):
        mutation_population_size = random.randint(1, 4)
        mutation_index_list = np.random.choice([x for x in range(80)], mutation_population_size)
        for i in mutation_index_list:
            children[i].genes[random.randint(0, 9)] = random.random()
            children[i].normalize()

    def __replacement(self, children):
        i = 0
        while i < 80:
            self.chromosomes[i] = children[i]
            i += 1

    def next_generation(self):
        children = self.__select_by_tournament()
        self.__mutate(children)
        self.__replacement(children)
        self.__prepare_for_next_iteration()

    def next_generation_by_differential_evolution(self):
        mutation_vector = np.array([])
        for i in range(len(self.chromosomes)):
            c1 = self.__select_different_chromosome_index([i])
            c2 = self.__select_different_chromosome_index([i, c1])
            f = 0.63
            # mutation_vector = np.append(mutation_vector, [[x for x in ]])

    def __prepare_for_next_iteration(self):
        self.fitness = np.array([])


if __name__ == '__main__':
    portfolio = PortfolioData()
    print(portfolio.assets.to_string())
    print(portfolio.returns.to_string())
    print(portfolio.average_expected_return.to_string())
    population = Population()
    population.compute_fitness(portfolio.average_expected_return.to_numpy(copy=True), portfolio.covariance)
    population.sort()
    print("start")
    print(population.fitness[-1])
    best_fitness_history = [population.fitness[-1]]
    enough_convergence = 80
    while enough_convergence != 0:
        population.next_generation()
        population.compute_fitness(portfolio.average_expected_return.to_numpy(copy=True), portfolio.covariance)
        population.sort()
        best_fitness_history.append(population.fitness[-1])
        if best_fitness_history[-1] == best_fitness_history[-2]:
            enough_convergence -= 1
        else:
            enough_convergence = 80
    print(population.fitness[-1])
    plt.plot([i for i in range(len(best_fitness_history))], best_fitness_history)
    plt.show()
    print(population.chromosomes[-1].genes)
    print(pd.DataFrame([population.chromosomes[-1].genes], columns=portfolio.assets.columns).to_string())
    plt.pie(population.chromosomes[-1].genes, labels=portfolio.assets.columns)
    plt.show()
    chromosome_return = population.compute_chromosome_return(population.chromosomes[-1],
                                                             portfolio.average_expected_return.to_numpy())
    chromosome_risk = population.compute_risk(population.chromosomes[-1], portfolio.covariance)
    plt.bar(['Return', 'Risk'], [chromosome_return, chromosome_risk])
    plt.show()
