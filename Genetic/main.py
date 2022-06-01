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

    def compute_risk(self, covariance):
        risk = 0
        for i in range(0, len(self.genes)):
            for j in range(0, len(self.genes)):
                risk += self.genes[i] * self.genes[j] * covariance[i][j]
        return risk

    def compute_chromosome_return(self, average_expected_returns):
        return sum([average_expected_returns[0][i] * self.genes[i] for i in range(len(
                average_expected_returns[0]))])


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
    def __init__(self, average_expected_returns, covariance):
        self.average_expected_returns = average_expected_returns
        self.covariance = covariance
        self.chromosomes = np.array([Chromosome() for i in range(100)])
        self.fitness = np.array([])
        self.__compute_fitness()
        self.__sort()

    def next_generation(self):
        children = self.__select_by_tournament()
        self.__mutate(children)
        self.__replacement(children)
        self.__compute_fitness()
        self.__sort()

    def next_generation_by_differential_evolution(self):
        mutation_vectors = self.__generate_mutation_vectors()
        children = self.__differential_cross(mutation_vectors)
        self.__differential_replacement(children)
        self.__compute_fitness()
        self.__sort()

    def __differential_replacement(self, children):
        for chromosome_index in range(len(self.chromosomes)):
            if self.__target_function(children[chromosome_index]) > self.__target_function(
                    self.chromosomes[chromosome_index]):
                self.chromosomes[chromosome_index] = children[chromosome_index]

    def __target_function(self, chromosome):
        numerator = chromosome.compute_chromosome_return(self.average_expected_returns)
        denominator = chromosome.compute_risk(self.covariance)
        return numerator / denominator

    def __generate_mutation_vectors(self):
        mutation_vectors = np.array([])
        for i in range(len(self.chromosomes)):
            c1 = self.__select_different_chromosome_index([i])
            c2 = self.__select_different_chromosome_index([i, c1])
            f = random.random()
            mutation_vectors = np.append(mutation_vectors, [self.__generate_mutation_vector_de_current(i, c1, c2, f)])
        return mutation_vectors

    def __differential_cross(self, mutation_vectors):
        children = np.array([])
        for chromosome_index in range(len(self.chromosomes)):
            cr = random.random()
            new_child: Chromosome = Chromosome()
            for gen_index in range(len(self.chromosomes[chromosome_index].genes)):
                if random.random() == cr or \
                        random.randint(0, len(self.chromosomes[chromosome_index].genes)) == gen_index:
                    new_child.genes[gen_index] = mutation_vectors[chromosome_index].genes[gen_index]
                else:
                    new_child.genes[gen_index] = self.chromosomes[chromosome_index].genes[gen_index]
            new_child.normalize()
            children = np.append(children, [new_child])
        return children

    def __generate_mutation_vector_de_current(self, current, c1, c2, f):
        mutation_vector = Chromosome()
        for i in range(len(mutation_vector.genes)):
            mutation_vector.genes[i] = abs(self.chromosomes[current].genes[i] + f * (self.chromosomes[c1].genes[i] -
                                                                                     self.chromosomes[c2].genes[i]))
        mutation_vector.normalize()
        return mutation_vector

    def __compute_fitness(self):
        self.fitness = np.array([])
        for chromosome in self.chromosomes:
            self.fitness = np.append(self.fitness, [self.__target_function(chromosome)])

    def __sort(self):
        sorted_pairs = sorted(zip(self.fitness, self.chromosomes), key=lambda x: x[0])
        self.fitness = [i[0] for i in sorted_pairs]
        self.chromosomes = [i[1] for i in sorted_pairs]

    def __cross(self, f1_index, f2_index, alpha):
        children1, children2 = Chromosome(), Chromosome()
        for i in range(len(children1.genes)):
            children1.genes[i] = \
                self.chromosomes[f1_index].genes[i] * (1 - alpha) + self.chromosomes[f2_index].genes[i] * alpha
            children2.genes[i] = \
                self.chromosomes[f2_index].genes[i] * (1 - alpha) + self.chromosomes[f1_index].genes[i] * alpha
        children1.normalize()
        children2.normalize()
        return [children1, children2]

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


def use_genetic():
    convergence_iterations = 100
    population: Population = Population(portfolio.average_expected_return.to_numpy(copy=True), portfolio.covariance)
    print("start")
    print(population.fitness[-1])
    best_fitness_history = [population.fitness[-1]]
    current_iterations = convergence_iterations
    while current_iterations > 0:
        population.next_generation()
        best_fitness_history.append(population.fitness[-1])
        if best_fitness_history[-1] == best_fitness_history[-2]:
            current_iterations -= 1
        else:
            current_iterations = convergence_iterations
    print(population.fitness[-1])
    plt.plot([i for i in range(len(best_fitness_history))], best_fitness_history)
    plt.show()
    print(pd.DataFrame([population.chromosomes[-1].genes], columns=portfolio.assets.columns).to_string())
    plt.pie(population.chromosomes[-1].genes, labels=portfolio.assets.columns)
    plt.show()
    chromosome_return = population.chromosomes[-1].compute_chromosome_return(
        portfolio.average_expected_return.to_numpy())
    chromosome_risk = population.chromosomes[-1].compute_risk(portfolio.covariance)
    plt.bar(['Return', 'Risk'], [chromosome_return, chromosome_risk])
    plt.show()


def use_differential():
    convergence_iterations = 50
    population: Population = Population(portfolio.average_expected_return.to_numpy(copy=True), portfolio.covariance)
    print("start")
    print(population.fitness[-1])
    best_fitness_history = [population.fitness[-1]]
    current_iterations = convergence_iterations
    while current_iterations > 0:
        population.next_generation_by_differential_evolution()
        best_fitness_history.append(population.fitness[-1])
        print(population.fitness[-1])
        if best_fitness_history[-1] == best_fitness_history[-2]:
            current_iterations -= 1
        else:
            current_iterations = convergence_iterations
    print(population.fitness[-1])
    plt.plot([i for i in range(len(best_fitness_history))], best_fitness_history)
    plt.show()
    print(pd.DataFrame([population.chromosomes[-1].genes], columns=portfolio.assets.columns).to_string())
    plt.pie(population.chromosomes[-1].genes, labels=portfolio.assets.columns)
    plt.show()
    chromosome_return = population.chromosomes[-1].compute_chromosome_return(
        portfolio.average_expected_return.to_numpy())
    chromosome_risk = population.chromosomes[-1].compute_risk(portfolio.covariance)
    plt.bar(['Return', 'Risk'], [chromosome_return, chromosome_risk])
    plt.show()


if __name__ == '__main__':
    portfolio = PortfolioData()
    print(portfolio.assets.to_string())
    print(portfolio.returns.to_string())
    print(portfolio.average_expected_return.to_string())
    use_differential()
    use_genetic()

