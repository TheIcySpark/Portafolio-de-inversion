import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import random
from enum import Enum


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
        # Se almacenan los retornos en promedio
        self.average_expected_returns = average_expected_returns
        # se almacena la tabla se covarianza
        self.covariance = covariance
        # Se crea una poblacion de 100 cromosomas
        self.chromosomes = np.array([Chromosome() for i in range(100)])
        # se utiliza un array para el calculo del fitness
        self.fitness = np.array([])
        # Al iniciar se calcula el fitness
        self.__compute_fitness()
        # Se ordena la poblacion en base al fitness
        self.__sort()

    # Esta funcion genera la siguiente generacion de la poblacion usando el algoritmo genetico
    def next_generation(self):
        children = self.__select_by_tournament()
        self.__mutate(children)
        self.__replacement(children)
        self.__compute_fitness()
        self.__sort()

    # Esta funcion genera la siguente generacion de la poblacion usando el algoritmo diferencial
    def next_generation_by_differential_evolution(self):
        mutation_vectors = self.__generate_mutation_vectors()
        children = self.__differential_cross(mutation_vectors)
        self.__differential_replacement(children)
        self.__compute_fitness()
        self.__sort()

    # Remplazo usando el algoritmo diferencial
    def __differential_replacement(self, children):
        for chromosome_index in range(len(self.chromosomes)):
            if self.__target_function(children[chromosome_index]) > self.__target_function(
                    self.chromosomes[chromosome_index]):
                self.chromosomes[chromosome_index] = children[chromosome_index]

    # Retorna el valor de la funcion objetivo con un cromosoma especifico
    def __target_function(self, chromosome):
        numerator = chromosome.compute_chromosome_return(self.average_expected_returns)
        denominator = chromosome.compute_risk(self.covariance)
        return numerator / denominator

    # Genera los vectores de mutacion
    def __generate_mutation_vectors(self):
        mutation_vectors = np.array([])
        for i in range(len(self.chromosomes)):
            c1 = self.__select_different_chromosome_index([i])
            c2 = self.__select_different_chromosome_index([i, c1])
            f = random.random()
            mutation_vectors = np.append(mutation_vectors, [self.__generate_mutation_vector_de_current(i, c1, c2, f)])
        return mutation_vectors

    # Realiza la crusa en el algoritmo diferencial
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

    # Genera vectores de mutacion usando De/current
    def __generate_mutation_vector_de_current(self, current, c1, c2, f):
        mutation_vector = Chromosome()
        for i in range(len(mutation_vector.genes)):
            mutation_vector.genes[i] = abs(self.chromosomes[current].genes[i] + f * (self.chromosomes[c1].genes[i] -
                                                                                     self.chromosomes[c2].genes[i]))
        mutation_vector.normalize()
        return mutation_vector

    # Calcula el fitness de la poblacion
    def __compute_fitness(self):
        self.fitness = np.array([])
        for chromosome in self.chromosomes:
            self.fitness = np.append(self.fitness, [self.__target_function(chromosome)])

    # Ordena la poblacion en base al fitness
    def __sort(self):
        sorted_pairs = sorted(zip(self.fitness, self.chromosomes), key=lambda x: x[0])
        self.fitness = [i[0] for i in sorted_pairs]
        self.chromosomes = [i[1] for i in sorted_pairs]

    # Realiza la cruza en el algoritmo genetico
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

    # Selecciona un indice que no este en la lista index_list
    def __select_different_chromosome_index(self, index_list: list):
        index2 = index_list[0]
        while index2 in index_list:
            index2 = random.randint(0, len(self.chromosomes) - 1)
        return index2

    # Realiza la seleccion por torneo
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

    # Realiza la mutacion en el algoritmo genetico
    @staticmethod
    def __mutate(children):
        mutation_population_size = random.randint(1, 4)
        mutation_index_list = np.random.choice([x for x in range(80)], mutation_population_size)
        for chromosome_index in mutation_index_list:
            children[chromosome_index].genes[random.randint(0, 9)] = random.random()
            children[chromosome_index].normalize()

    # Realiza el remplazo en el algoritmo genetico
    def __replacement(self, children):
        i = 0
        while i < 80:
            self.chromosomes[i] = children[i]
            i += 1


class EvolutionaryAlgorithm(Enum):
    GENETIC = 1
    DIFFERENTIAL = 2


def use_genetic(population: Population, best_fitness_history):
    convergence_iterations = 500
    current_iterations = convergence_iterations
    while current_iterations > 0:
        population.next_generation()
        best_fitness_history.append(population.fitness[-1])
        if best_fitness_history[-1] == best_fitness_history[-2]:
            current_iterations -= 1
        else:
            current_iterations = convergence_iterations


def use_differential(population: Population, best_fitness_history):
    convergence_iterations = 50
    current_iterations = convergence_iterations
    while current_iterations > 0:
        population.next_generation_by_differential_evolution()
        best_fitness_history.append(population.fitness[-1])
        if best_fitness_history[-1] == best_fitness_history[-2]:
            current_iterations -= 1
        else:
            current_iterations = convergence_iterations


def use_evolutionary_algorithm(evolutionary_algorithm: EvolutionaryAlgorithm):
    population: Population = Population(portfolio.average_expected_return.to_numpy(copy=True), portfolio.covariance)
    print(population.fitness[-1])
    best_fitness_history = [population.fitness[-1]]
    if evolutionary_algorithm == EvolutionaryAlgorithm.GENETIC:
        use_genetic(population, best_fitness_history)
    elif evolutionary_algorithm == EvolutionaryAlgorithm.DIFFERENTIAL:
        use_differential(population, best_fitness_history)
    print(best_fitness_history[-1])
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


def show_portfolio_info(portfolio: PortfolioData):
    print(portfolio.assets.to_string())
    print(portfolio.returns.to_string())
    print(portfolio.average_expected_return.to_string())

if __name__ == '__main__':
    portfolio = PortfolioData()
    show_portfolio_info(portfolio)
