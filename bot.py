import random
import csv
import math
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

class Individual:
    dictionary = [['R', 'R', 'R', 'R'], ['R', 'R', 'R', 'P'],
              ['R', 'R', 'R', 'S'], ['R', 'R', 'P', 'R'],
              ['R', 'R', 'P', 'P'], ['R', 'R', 'P', 'S'],
              ['R', 'R', 'S', 'R'], ['R', 'R', 'S', 'P'],
              ['R', 'R', 'S', 'S'], ['R', 'P', 'R', 'R'],
              ['R', 'P', 'R', 'P'], ['R', 'P', 'R', 'S'],
              ['R', 'P', 'P', 'R'], ['R', 'P', 'P', 'P'],
              ['R', 'P', 'P', 'S'], ['R', 'P', 'S', 'R'],
              ['R', 'P', 'S', 'P'], ['R', 'P', 'S', 'S'],
              ['R', 'S', 'R', 'R'], ['R', 'S', 'R', 'P'],
              ['R', 'S', 'R', 'S'], ['R', 'S', 'P', 'R'],
              ['R', 'S', 'P', 'P'], ['R', 'S', 'P', 'S'],
              ['R', 'S', 'S', 'R'], ['R', 'S', 'S', 'P'],
              ['R', 'S', 'S', 'S'], ['P', 'R', 'R', 'R'],
              ['P', 'R', 'R', 'P'], ['P', 'R', 'R', 'S'],
              ['P', 'R', 'P', 'R'], ['P', 'R', 'P', 'P'],
              ['P', 'R', 'P', 'S'], ['P', 'R', 'S', 'R'],
              ['P', 'R', 'S', 'P'], ['P', 'R', 'S', 'S'],
              ['P', 'P', 'R', 'R'], ['P', 'P', 'R', 'P'],
              ['P', 'P', 'R', 'S'], ['P', 'P', 'P', 'R'],
              ['P', 'P', 'P', 'P'], ['P', 'P', 'P', 'S'],
              ['P', 'P', 'S', 'R'], ['P', 'P', 'S', 'P'],
              ['P', 'P', 'S', 'S'], ['P', 'S', 'R', 'R'],
              ['P', 'S', 'R', 'P'], ['P', 'S', 'R', 'S'],
              ['P', 'S', 'P', 'R'], ['P', 'S', 'P', 'P'],
              ['P', 'S', 'P', 'S'], ['P', 'S', 'S', 'R'],
              ['P', 'S', 'S', 'P'], ['P', 'S', 'S', 'S'],
              ['S', 'R', 'R', 'R'], ['S', 'R', 'R', 'P'],
              ['S', 'R', 'R', 'S'], ['S', 'R', 'P', 'R'],
              ['S', 'R', 'P', 'P'], ['S', 'R', 'P', 'S'],
              ['S', 'R', 'S', 'R'], ['S', 'R', 'S', 'P'],
              ['S', 'R', 'S', 'S'], ['S', 'P', 'R', 'R'],
              ['S', 'P', 'R', 'P'], ['S', 'P', 'R', 'S'],
              ['S', 'P', 'P', 'R'], ['S', 'P', 'P', 'P'],
              ['S', 'P', 'P', 'S'], ['S', 'P', 'S', 'R'],
              ['S', 'P', 'S', 'P'], ['S', 'P', 'S', 'S'],
              ['S', 'S', 'R', 'R'], ['S', 'S', 'R', 'P'],
              ['S', 'S', 'R', 'S'], ['S', 'S', 'P', 'R'],
              ['S', 'S', 'P', 'P'], ['S', 'S', 'P', 'S'],
              ['S', 'S', 'S', 'R'], ['S', 'S', 'S', 'P'],
              ['S', 'S', 'S', 'S']]
    
    def __init__(self):
        self.fitness = 0
        self.win = 0
        self.lose = 0
        self.draw = 0
        self.dna = {}
        for item in self.dictionary:
            self.dna[str(item)] = random.choice(['R', 'P', 'S'])

    def get_fitness(self)->int:
        self.fitness = self.win * 5 - self.lose * 2 - self.draw * 1
        return self.fitness

    def crossover(self, other):
        baby = Individual()
        i = 0
        for key in self.dna.keys():
            if(i < 40):
                baby.dna.update({key : self.dna.get(key)}) 
            else:
                baby.dna.update({key : other.dna.get(key)}) 
            i += 1

        return baby

    def mutate(self, rate = 0.01):
        mutateRate =  math.ceil(rate * 81)
        mutateGene = ""
        if mutateRate == 0:
            return 0
        
        for i in range(0, mutateRate):
            for j in range(0, 4):
                mutateGene += random.choice(['R', 'P', 'S'])
        
            if self.dna.get(mutateGene) == "P":
                self.dna[mutateGene] = random.choice(['R', 'S'])
            elif self.dna.get(mutateGene) == "S":
                self.dna[mutateGene] = random.choice(['R', 'P'])
            else:
                self.dna[mutateGene] = random.choice(['P', 'S'])

            mutateGene = ""
        return 0
    def __str__(self) -> str:
        str = ""
        for key, value in self.dna:
            if key == "SSSS":
                str += value
            else:
                str += value + ","

        return str


class Envirnoment:
    def __init__(self, populationSize, envirnoment, mutateRate) -> None:
        self.size = populationSize
        self.rate = mutateRate
        self.pop = []
        self.matingPool = []
        self.avgFitPerGene = []
        self.sumFitPerGen = []
        self.avgPosFitPerGene = []
        self.env = envirnoment
        self.envSize = len(envirnoment)
        self.gen_itr = self.envSize//(81 * self.size)
         
        self.repopulate()

    def repopulate(self):
        for i in range (0 , self.size):
            self.pop.append(Individual())

    def fitness_test(self, ind, gene):
        gene_test = str(gene[0])

        if ind.dna.get(gene_test) == "P":
            if gene[1] == "P":
                ind.draw += 1
            elif gene[1] == "S":
                ind.lose += 1
            else:
                ind.win += 1
        elif ind.dna.get(gene_test) == "S":
            if gene[1] == "P":
                ind.win += 1
            elif gene[1] == "S":
                ind.draw += 1
            else:
                ind.lose += 1
        else:
            if gene[1] == "P":
                ind.lose += 1
            elif gene[1] == "S":
                ind.win += 1
            else:
                ind.draw += 1

            
    def test_individaul(self, ind, test):
        for gene in test:
            self.fitness_test(ind, gene)
    
    def test_env(self):
       
        for i in range(0, self.gen_itr):
            pos_sum_fitness = 0
            total_fitness = 0
            self.matingPool = []

            #test fitness of an individual 
            for ind in self.pop:
                test = []
                for j in range(0, 81):
                    test.append(self.env.pop(0))
                self.test_individaul(ind, test)
                if ind.get_fitness() > 0:
                    pos_sum_fitness += ind.get_fitness()
                total_fitness += ind.get_fitness()

            ##################STATISTICS##################
            totalAvg = total_fitness//len(self.pop)
            self.avgFitPerGene.append(totalAvg)
            self.sumFitPerGen.append(total_fitness)    
            self.avgPosFitPerGene.append(pos_sum_fitness)
            print(f"Generation {i}: Total Fitness - {total_fitness}, Avg Fitness - {totalAvg}, Total Postive - {pos_sum_fitness} ")
            ##################STATISTICS##################

            # selection of individaul to enter mating pool
            for ind in self.pop:
                if ind.get_fitness() > 0:
                    n = ind.get_fitness()
                    for i in range(0, n):
                        self.matingPool.append(ind)
            
            self.pop = []
            for i in range(0, self.size):
                a = random.randint(0, len(self.matingPool) - 1)
                b = random.randint(0, len(self.matingPool) - 1)
                #print(f"Mating Pool Length: {len(self.matingPool)}, parentA index: {a}, parentB index: {b}")
                parentA = self.matingPool[a]
                parentB = self.matingPool[b]

                baby = parentA.crossover(parentB)
                baby.mutate(self.rate)

                self.pop.append(baby)

data = []
game = 'data1.csv'

with open(game, 'r') as file:
    reader = csv.reader(file)
    for row in reader:
        data.append(row)

game = Envirnoment(60,data, 0.01)
game.test_env()

# create some dummy data for three datasets
generation = np.arange(0, game.gen_itr)
fitness1 = np.array(game.avgFitPerGene)
fitness2 = np.array(game.sumFitPerGen)
fitness3 = np.array(game.avgPosFitPerGene)

# create a line plot with three different lines
sns.lineplot(x=generation, y=fitness1, color='blue', label='Dataset 1')
sns.lineplot(x=generation, y=fitness2, color='red', label='Dataset 2')
sns.lineplot(x=generation, y=fitness3, color='green', label='Dataset 3')

# set the title and axis labels
plt.title('Fitness vs Generation')
plt.xlabel('Generation')
plt.ylabel('Fitness')

# show the legend
plt.legend()

# show the plot
plt.show()

            
            
            

