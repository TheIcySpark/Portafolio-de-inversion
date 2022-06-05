import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import random

def calcular_retornos():
    act, ret = activos.to_numpy(copy=True), activos.to_numpy(copy=True)
    r = 1
    while r < len(ret):
        c = 0
        while c < len(ret[r]):
            ret[r][c] = (act[r][c] - act[r - 1][c]) / act[r - 1][c]
            c += 1
        r += 1
    return ret[1:, ]

activos = pd.read_csv('portfolio.csv')
retornos = pd.DataFrame(calcular_retornos(), columns=activos.columns)
numero_retornos = len(retornos.to_numpy(copy=True))
aux = []
aux = retornos.to_numpy(copy=True)
promedio_retornos = [sum(aux[:, x]) / len(aux) for x in range(len(aux[0]))]
covarianza = np.cov(retornos.to_numpy(copy=True), rowvar=False)
print(promedio_retornos)

def normalizar(cromosoma):
    suma = sum(cromosoma)
    for i in range(10):
        cromosoma[i] = cromosoma[i] / suma
    return cromosoma

def crear_cromosoma():
    valores = []
    for i in range(10):
        valores.append(random.random())
    normalizar(valores)
    return valores

def crear_poblacion():
    p = []
    for i in range(100):
        p.append(crear_cromosoma())
    return p

poblacion = crear_poblacion()
aptitud = []

def calcular_riesgo(cromosoma):
    global covarianza
    risk = 0
    for i in range(10):
        for j in range(10):
            risk += cromosoma[i] * cromosoma[j] * covarianza[i][j]
    return risk

def calcular_retorno(cromosoma):
    global promedio_retornos
    return sum([promedio_retornos[i] * cromosoma[i] for i in range(len(promedio_retornos))])




def next_generation():
    children = select_by_tournament()
    mutate(children)
    replacement(children)
    compute_fitness()
    sort()

def next_generation_by_differential_evolution():
    mutation_vectors = generate_mutation_vectors()
    children = differential_cross(mutation_vectors)
    differential_replacement(children)
    compute_fitness()
    sort()

def differential_replacement( children):
    global poblacion
    for chromosome_index in range(100):
        if target_function(children[chromosome_index]) > target_function(
                poblacion[chromosome_index]):
            poblacion[chromosome_index] = children[chromosome_index]

def target_function(chromosome):
    numerator = calcular_retorno(chromosome)
    denominator = calcular_riesgo(chromosome)
    return numerator / denominator

def generate_mutation_vectors():
    mutation_vectors = []
    for i in range(100):
        c1 = select_different_chromosome_index([i])
        c2 = select_different_chromosome_index([i, c1])
        f = random.random()
        mutation_vectors.append(generate_mutation_vector_de_current(i, c1, c2, f))
    return mutation_vectors

def differential_cross(mutation_vectors):
    global poblacion
    children = []
    for chromosome_index in range(100):
        cr = random.random()
        new_child = crear_cromosoma()
        for gen_index in range(10):
            if random.random() == cr or \
                    random.randint(0, 9) == gen_index:
                new_child[int(gen_index)] = mutation_vectors[int(chromosome_index)][int(gen_index)]
            else:
                new_child[gen_index] = poblacion[chromosome_index][gen_index]
        normalizar(new_child)
        children.append(new_child)
    return children

def generate_mutation_vector_de_current(current, c1, c2, f):
    global poblacion
    mutation_vector = crear_cromosoma()
    for i in range(10):
        mutation_vector[i] = abs(poblacion[current][i] + f * (poblacion[c1][i] -
                                                                                 poblacion[c2][i]))
    mutation_vector = normalizar(mutation_vector)
    return mutation_vector

def compute_fitness():
    global poblacion, aptitud
    aptitud = []
    for cromosoma in poblacion:
        aptitud.append(target_function(cromosoma))

def sort():
    global poblacion, aptitud
    sorted_pairs = sorted(zip(aptitud, poblacion), key=lambda x: x[0])
    aptitud = [i[0] for i in sorted_pairs]
    poblacion = [i[1] for i in sorted_pairs]

def cross(f1_index, f2_index, alpha):
    global poblacion
    children1 = crear_cromosoma()
    children2 = crear_cromosoma()
    for i in range(10):
        children1[i] = \
            poblacion[f1_index][i] * (1 - alpha) + poblacion[f2_index][i] * alpha
        children2[i] = \
            poblacion[f2_index][i] * (1 - alpha) + poblacion[f1_index][i] * alpha
    normalizar(children1)
    normalizar(children2)
    return children1, children2

def select_different_chromosome_index(index_list: list):
    index2 = index_list[0]
    while index2 in index_list:
        index2 = random.randint(0, 99)
    return index2

def select_by_tournament():
    children = []
    for i in range(40):
        alpha = random.random()
        f1_index = random.randint(0, 99)
        f2_index = select_different_chromosome_index([f1_index])
        f3_index = select_different_chromosome_index([f1_index, f2_index])
        sorted_index_list = sorted([f1_index, f2_index, f3_index])
        f1_index = sorted_index_list[1]
        f2_index = sorted_index_list[2]
        a1, a2 = cross(f1_index, f2_index, alpha)
        children.append(a1)
        children.append(a2)
    return children

def mutate(children):
    mutation_index = random.randint(0, 39)
    children[mutation_index][random.randint(0, 9)] = random.random()
    normalizar(children[mutation_index])

def replacement(children):
    i = 0
    while i < 80:
        poblacion[i] = children[i]
        i += 1

best_fitness_history = []
for i in range(600):
    next_generation_by_differential_evolution()
    best_fitness_history.append(aptitud[-1])
plt.plot([i for i in range(len(best_fitness_history))], best_fitness_history)
plt.show()
plt.pie(poblacion[-1], labels=activos.columns)
plt.show()
print(calcular_retorno(poblacion[-1]))
print(calcular_riesgo(poblacion[-1]))
plt.bar(['Return', 'Risk'], [calcular_retorno(poblacion[-1]), calcular_riesgo(poblacion[-1])])
plt.show()

best_fitness_history = []
for i in range(3000):
    next_generation()
    best_fitness_history.append(aptitud[-1])
plt.plot([i for i in range(len(best_fitness_history))], best_fitness_history)
plt.show()

plt.bar(['Return', 'Risk'], [calcular_retorno(poblacion[-1]), calcular_riesgo(poblacion[-1])])
plt.show()

