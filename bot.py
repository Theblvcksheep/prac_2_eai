import random
import csv
import math
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

random.seed(20516)
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
        self.untested = 0
        self.curr_gen = 0
        self.dna = {}
        for item in self.dictionary:
            self.dna["".join(item)] = [random.choice(['R', 'P', 'S']), -10]

    def get_fitness(self, Ideal = False)->int:
        if Ideal:
            self.win = 81
            self.fitness = int(((self.win/81)*100))* 5
            return self.fitness
        
        self.win = 0 
        self.draw = 0
        self.lose = 0
        self.untested = 0
        for keys, values in self.dna.items():
            if values[1] == -1:
                self.lose += 1
            elif values[1] == 1:
                self.win += 1
            elif values[1] == 0:
                self.draw += 1
            else:
                self.untested += 1

        self.fitness = int(((self.win/81)*100)) * 5 - int(((self.lose/81)*100)) * 2 - int(((self.draw/81)*100)) 
        return self.fitness

    def crossover(self, other):
        baby = Individual()
        
        for key in baby.dna.keys():
            if self.dna.get(key)[1] > other.dna.get(key)[1]:
                baby.dna[key] = self.dna.get(key)
            else:
                baby.dna[key] = other.dna.get(key)

        return baby

    def mutate(self, rate = 0.01):
        mutateRate =  math.ceil(rate * 81)
        mutateGene = ""
        if mutateRate == 0:
            return 0
        
        for i in range(0, mutateRate):
            for j in range(0, 4):
                mutateGene += random.choice(['R', 'P', 'S'])

            if self.dna.get(mutateGene)[0] == "P":
                self.dna[mutateGene] = [random.choice(['R', 'S']), -10]
            elif self.dna.get(mutateGene)[0] == "S":
                self.dna[mutateGene] = [random.choice(['R', 'P']), -10]
            else:
                self.dna[mutateGene] = [random.choice(['P', 'S']), -10]
            mutateGene = ""

        return 0
    
    def __str__(self) -> str:
        str = ""
        for key, value in self.dna.items():
            if key == 'SSSS':
                str += value[0]
            else:
                str += value[0] + ","

        return str


class Envirnoment:
    def __init__(self, populationSize, envirnoment, mutateRate, el, el_rate, cul) -> None:
        self.size = populationSize
        self.rate = mutateRate
        self.elite_flag = el
        self.elite_rate = el_rate
        self.elites = []
        self.culling = cul
        self.ideal = 0
        self.pop = []
        self.matingPool = []
        self.idealFitPerGene = []
        self.maxIndFitPerGen = []
        self.avgFitPerGen = []
        self.env = envirnoment
        self.envSize = len(envirnoment)
        self.gen_itr = self.envSize//(81 * self.size)
        self.max_individaul = Individual()
        self.repopulate()
        self.sim_ideal()
        self.repopulate()

    def repopulate(self):
        for i in range (0 , self.size):
            self.pop.append(Individual())

    def fitness_test(self, ind, gene):
        gene_test = str(gene[0])
        
        if ind.dna.get(gene_test)[0] == "P":
            if gene[1] == "P":
                ind.dna[gene_test][1] = 0
            elif gene[1] == "S":
                ind.dna[gene_test][1] = -1
            else:
                ind.dna[gene_test][1] = 1

        elif ind.dna.get(gene_test)[0] == "S":
           
            if gene[1] == "P":
                ind.dna[gene_test][1] = 1
            elif gene[1] == "S":
                ind.dna[gene_test][1] = 0
            else:
                ind.dna[gene_test][1] = -1
        else:
            if gene[1] == "P":
                ind.dna[gene_test][1] = -1
            elif gene[1] == "S":
                ind.dna[gene_test][1] = 1
            else:
                ind.dna[gene_test][1] = 0

            
    def test_individaul(self, ind, test):
        for gene in test:
            self.fitness_test(ind, gene)
            ind.get_fitness()

    def sim_ideal(self):
        for ind in self.pop:
            self.ideal += ind.get_fitness(True)
        self.pop.clear()

    def get_max_individaul(self):
        max = Individual()
        for ind in self.pop:
            if max.get_fitness() < ind.get_fitness():
                max = ind
        
        if self.max_individaul.get_fitness() < max.get_fitness():
            self.max_individaul = max

        return max
    
    def get_elites(self):
        self.elites = [Individual()] * int(self.size * self.elite_rate)
        for ind in self.pop:
            if ind.get_fitness() > min([elite.get_fitness() for elite in self.elites]):
                self.elites.remove(min(self.elites, key=lambda elite: elite.get_fitness()))
                self.elites.append(ind)

    def test_env(self):
       
        for i in range(0, self.gen_itr):
            total_fitness = 0
            self.matingPool = []

            #test fitness of an individual 
            for ind in self.pop:
                test = []
                for j in range(0, 81):
                    test.append(self.env.pop(0))
                ind.curr_gen = i
                self.test_individaul(ind, test)
                total_fitness += ind.get_fitness()

            ##################STATISTICS##################
            idealFitPer = round((total_fitness/self.ideal) * 100, 2)
            avgFit = total_fitness//len(self.pop)
            self.idealFitPerGene.append(idealFitPer)
            self.avgFitPerGen.append(avgFit)    
            print(f"Generation {i}: Avg. Fitness - {avgFit}, Ideal Fitness - {self.ideal}, Ideal Precentage - {idealFitPer}%")
            max = self.get_max_individaul()
            self.maxIndFitPerGen.append(max.get_fitness())
            print(f"Individaul with the highest fitness: {max.get_fitness()}, Selection Threshold: {max.get_fitness()//2}")
            print(f"Max individual Win: {max.win}, Losses: {max.lose}, Draws: {max.draw}, Untested: {max.untested}")
            ##################STATISTICS##################

            # selection of individaul to enter mating pool
            for ind in self.pop:
                if ind.get_fitness() > int(max.get_fitness() * self.culling):
                    n = ind.get_fitness()
                    for k in range(0, n):
                        self.matingPool.append(ind)
            
            self.pop = []
            self.get_elites()
            for n in range(0, self.size):
                if self.elites and len(self.elites) != 0:
                    baby = self.elites.pop(0)
                    baby.win = 0
                    baby.lose = 0
                    baby.fitness = 0
                    baby.draw = 0
                    self.pop.append(baby)
                else:    
                    a = random.randint(0, len(self.matingPool) - 1)
                    b = random.randint(0, len(self.matingPool) - 1)
                    parentA = self.matingPool[a]
                    parentB = self.matingPool[b]

                    baby = parentA.crossover(parentB)
                    baby.mutate(self.rate)

                    self.pop.append(baby)

data = []
game_file = 'data2'
size = 100
rate = 0.05
elitism = True 
eliteRate = 0.2
cullingRate = 0.5
with open(game_file + '.csv', 'r') as file:
    reader = csv.reader(file)
    for row in reader:
        data.append(row)

game = Envirnoment(size, data, rate, elitism, eliteRate, cullingRate)
game.test_env()
max = game.max_individaul

with open(game_file +'.txt', "w") as f:
    # write the string to the file
    f.write(str(max))

# close the file
f.close()
print("")
print(f"Individaul with the highest fitness: {max.get_fitness()}")
print(f"Max individual Win: {max.win}, Losses: {max.lose}, Draws: {max.draw}, Untested: {max.untested}")
for key, value in max.dna.items():
    if value[1] == 1:
        print(f"Max individual DNA: History Gene {key}, Move {value[0]} -- WIN!! ")
    elif value[1] == -1:
        print(f"Max individual DNA: History Gene {key}, Move {value[0]} -- LOSS!! ")
    elif value[1] == 0:
        print(f"Max individual DNA: History Gene {key}, Move {value[0]} -- DRAW!! ")
    else:
        print(f"Max individual DNA: History Gene {key}, Move {value[0]} -- UNTESTED!! ")

# create some dummy data for three datasets
generation = np.arange(0, game.gen_itr)
idealFit = np.array(game.idealFitPerGene)
avgFit = np.array(game.avgFitPerGen)
maxFit = np.array(game.maxIndFitPerGen)

#create a line plot with three different lines
sns.lineplot(x=generation, y=avgFit, color='blue', label='Avg. Fitness', marker='x')
sns.lineplot(x=generation, y=maxFit, color='black', label='Most Fit Individaul', marker='o')
# set the title and axis labels
plt.title('Total Fitness vs Generation')
plt.xlabel('Generation')
plt.ylabel('Fitness')

# show the plot
plt.show()

            
            
            

