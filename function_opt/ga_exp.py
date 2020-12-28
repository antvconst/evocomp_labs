from deap import tools, base
from multiprocessing import Pool
from ga_scheme import eaMuPlusLambda
from numpy import random as rnd
import numpy as np
from deap import creator
from deap import benchmarks

creator.create("BaseFitness", base.Fitness, weights=(1.0,))
creator.create("Individual", np.ndarray, fitness=creator.BaseFitness)


# minimal value for sigma parameter to avoid
# it being set to 0 or a negative value
min_std = 0.0001


# the mutation is radical, during mutation the
# coordinates are reset to new randomly generated
# values, mu and sigma evolve with the solution
def mutate(ind):
    n = len(ind)-2
    mu, sigma = ind[-2:]
    values = rnd.normal(mu, sigma, size=n)
    mu = rnd.normal(mu, 0.01)
    sigma = max(min_std, rnd.normal(sigma, 0.01))
    ind[:-2] = values
    ind[-2:] = mu, sigma
    return ind,


# the crossover is a milder version of cxBlend
# which ensures that sigma stays positive
def blend(ind1, ind2, alpha):
    n = len(ind1)
    alpha_ = rnd.uniform(0, alpha)
    value_1 = alpha_*ind1 + (1-alpha_)*ind2
    value_2 = alpha_*ind2 + (1-alpha_)*ind1
    ind1[:] = value_1
    ind2[:] = value_2
    return ind1, ind2


class SimpleGAExperiment:
    def factory(self):
        values = rnd.random(self.dimension) * 10 - 5
        params = np.array([
            rnd.normal(0, 1),        # mu
            rnd.uniform(min_std, 1)  # sigma
        ])
        return np.concatenate((values, params))

    def __init__(self, function, dimension, pop_size, iterations):
        self.pop_size = pop_size
        self.iterations = iterations
        self.mut_prob = 0.3
        self.cross_prob = 0.3

        self.function = function
        self.dimension = dimension

        self.engine = base.Toolbox()
        self.engine.register("map", map)
        self.engine.register("individual", tools.initIterate, creator.Individual, self.factory)
        self.engine.register("population", tools.initRepeat, list, self.engine.individual, self.pop_size)
        self.engine.register("mate", blend, alpha=0.3)
        self.engine.register("mutate", mutate)
        self.engine.register("select", tools.selRoulette)
        self.engine.register("evaluate", self.function)

    def run(self):
        pop = self.engine.population()
        hof = tools.HallOfFame(3, np.array_equal)
        stats = tools.Statistics(lambda ind: ind.fitness.values[0])
        stats.register("avg", np.mean)
        stats.register("std", np.std)
        stats.register("min", np.min)
        stats.register("max", np.max)

        pop, log = eaMuPlusLambda(pop, self.engine, mu=self.pop_size, lambda_=int(self.pop_size*0.8), cxpb=self.cross_prob, mutpb=self.mut_prob,
                                  ngen=self.iterations,
                                  stats=stats, halloffame=hof, verbose=True)
        print("Best = {}".format(hof[0]))
        print("Best fit = {}".format(hof[0].fitness.values[0]))
        return log

from functions import rastrigin
if __name__ == "__main__":

    def function(x):
        x = x[:-2]
        res = rastrigin(x)
        return res,

    dimension = 100
    pop_size = 100
    iterations = 1000
    scenario = SimpleGAExperiment(function, dimension, pop_size, iterations)
    log = scenario.run()
    from draw_log import draw_log
    draw_log(log)
