from deap import tools, base, creator
import numpy as np
from function_opt.ga_scheme import eaMuPlusLambda
import numpy.random as rnd
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, InputLayer
import gym
from function_opt.draw_log import draw_log
from copy import deepcopy

creator.create("BaseFitness", base.Fitness, weights=(1.0, ))
creator.create("Individual", list, fitness=creator.BaseFitness)

class RL_ga_experiment:
    def factory(self):
        individual = list()
        for i in range(len(self.params)):
            if i % 2 == 0:
                individual.append(rnd.normal(0.1, 0.3, size=self.params[i].shape))
            else:
                individual.append(np.zeros(shape=self.params[i].shape))
        return creator.Individual(individual)

    def mutation(self, ind):
        for i in range(len(ind)):
            shape = ind[i].shape
            mask = rnd.random(size=shape) < 0.15
            if i % 2 == 0:
                shift = rnd.normal(0, 0.2, size=shape)
                ind[i] += mask*shift
        return ind,

    def crossover(self, p1, p2):
        c1 = list()
        c2 = list()

        for p1_param, p2_param in zip(p1, p2):
            p1_new, p2_new = tools.cxTwoPoint(p1_param, p2_param)
            c1.append(p1_new)
            c2.append(p2_new)
        return creator.Individual(c1), creator.Individual(c2)

    def __init__(self, input_dim, l1, l2, output_dim, pop_size, iterations):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.l1 = l1
        self.l2 = l2
        self.pop_size = pop_size
        self.iterations = iterations

        self.model = self.build_model()
        self.params = self.model.get_weights()
        self.env = gym.make("CartPole-v1")

        self.engine = base.Toolbox()
        self.engine.register('map', map)
        self.engine.register('individual', tools.initIterate, creator.Individual, self.factory)
        self.engine.register('population', tools.initRepeat, list, self.engine.individual, self.pop_size)
        self.engine.register('mutate', self.mutation)
        self.engine.register("mate", self.crossover)
        self.engine.register('select', tools.selTournament, tournsize=3)
        self.engine.register('evaluate', self.fitness)

    def compare(self, ind1, ind2):
        result = True
        for i in range(len(ind1)):
            if i % 2 == 0:
                for j in range(len(ind1[i])):
                    for k in range(len(ind1[i][j])):
                        if ind1[i][j][k] != ind2[i][j][k]:
                            return False
        return result

    def run(self):
        pop = self.engine.population()
        hof = tools.HallOfFame(3, similar=self.compare)
        stats = tools.Statistics(lambda ind: ind.fitness.values[0])
        stats.register('min', np.min)
        stats.register('max', np.max)
        stats.register('avg', np.mean)
        stats.register('std', np.std)

        pop, log = eaMuPlusLambda(pop, self.engine, mu=self.pop_size, lambda_=int(0.8 * self.pop_size), cxpb=0.4, mutpb=0.4,
                                  ngen=self.iterations, verbose=True, halloffame=hof, stats=stats)
        best = hof[0]
        print("Best fitness = {}".format(best.fitness.values[0]))
        return log, best


    def build_model(self):
        model = Sequential()
        model.add(InputLayer(self.input_dim))
        model.add(Dense(self.l1, activation='tanh'))
        model.add(Dense(self.l2, activation='tanh'))
        model.add(Dense(self.output_dim, activation='softmax'))
        model.compile(optimizer='adam', loss='mse')
        return model

    def fitness(self, individual):
        self.model.set_weights(individual)
        scores = []
        for _ in range(3):
            state = self.env.reset()
            score = 0.0
            for t in range(200):
                self.env.render()
                act_prob = self.model.predict(state.reshape(1, self.input_dim)).squeeze()
                action = rnd.choice(np.arange(self.output_dim), 1, p=act_prob)[0]
                next_state, reward, done, _ = self.env.step(action)
                score += reward
                state = next_state
                if done:
                    break
            scores.append(score)
        return np.mean(scores),

if __name__ == '__main__':
    input_dim = 4
    l1 = 20
    l2 = 12
    output_dim = 2
    pop_size = 10
    iterations = 100

    exp = RL_ga_experiment(input_dim, l1, l2, output_dim, pop_size, iterations)
    log, best = exp.run()
    draw_log(log)

    rewards = []
    for _ in range(10):
        rewards.append(exp.fitness(best)[0])
    print(f'Mean reward of the best individual: {np.mean(rewards)}')