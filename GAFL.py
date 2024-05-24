import numpy as np
import random


class GAFL:
    def __init__(self, n_features, population_size, generations, mutation_rate, crossover_rate, fitness_function):
        self.n_features = n_features
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.fitness_function = fitness_function

    def initialize_population(self):
        return [np.random.randint(0, 2, self.n_features).tolist() for _ in range(self.population_size)]

    def evaluate_population(self, population, X, y):
        fitness_scores = []
        for individual in population:
            selected_features = [index for index, bit in enumerate(individual) if bit == 1]
            if len(selected_features) == 0:
                fitness_scores.append(0)
            else:
                X_selected = X[:, selected_features]
                fitness_scores.append(self.fitness_function(X_selected, y))
        return fitness_scores

    def select_parents(self, population, fitness_scores):
        sorted_population = [x for _, x in sorted(zip(fitness_scores, population), reverse=True)]
        return sorted_population[:self.population_size // 2]

    def crossover(self, parent1, parent2):
        if random.random() < self.crossover_rate:
            point = random.randint(1, self.n_features - 1)
            return parent1[:point] + parent2[point:], parent2[:point] + parent1[point:]
        return parent1, parent2

    def mutate(self, individual):
        for i in range(self.n_features):
            if random.random() < self.mutation_rate:
                individual[i] = 1 - individual[i]
        return individual

    def run(self, X, y):
        population = self.initialize_population()
        for generation in range(self.generations):
            fitness_scores = self.evaluate_population(population, X, y)
            parents = self.select_parents(population, fitness_scores)
            next_population = parents.copy()
            while len(next_population) < self.population_size:
                parent1, parent2 = random.sample(parents, 2)
                offspring1, offspring2 = self.crossover(parent1, parent2)
                next_population.append(self.mutate(offspring1))
                if len(next_population) < self.population_size:
                    next_population.append(self.mutate(offspring2))
            population = next_population
            print(f"Generation {generation}: Best Fitness = {max(fitness_scores)}")
        best_individual = population[np.argmax(fitness_scores)]
        return best_individual