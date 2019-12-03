from chromosome import link, chromosome, neuron
import _pickle as cPickle
from copy import deepcopy
import mutate, crossover
import random, math

COMPATIBILITY_RANGE = 4.5
C1 = 1
C2 = 2
C3 = 0.1
def compatibilityDistance(representative, newChromosome):
	''' See excessDisjointWeight in https://github.com/basanthjenuhb/Mario-AI/blob/master/neat.py
	'''
	representativeLinks = deepcopy(representative.links)
	newChromosomeLinks = deepcopy(newChromosome.links)

	representativeLinks = sorted(representativeLinks)
	newChromosomeLinks = sorted(newChromosomeLinks)


	excess, disjoint, W, i , j = 0.0, 0.0, 0.0, 0, 0
	divisorForWeightDifference = 1
	while(i < len(representativeLinks) and j < len(newChromosomeLinks)):
		if(representativeLinks[i] == newChromosomeLinks[j]):
			W = W + abs(representativeLinks[i].weight - newChromosomeLinks[j].weight)
			divisorForWeightDifference += 1
			i = i + 1
			j = j + 1
		elif(representativeLinks[i] < newChromosomeLinks[j]):
			i = i + 1
			disjoint = disjoint + 1
		else:
			j = j + 1
			disjoint = disjoint + 1

	excess = len(representativeLinks[i:]) + len(newChromosomeLinks[j:])
	N = float( max( len(representativeLinks), len(newChromosomeLinks) ) )
	#print(disjoint,excess,N,W)
	if N < 20:
		N = 1.0
	distance = float(C1 * excess / N) + float(C2 * disjoint / N) + float(C3 * W / divisorForWeightDifference) 
	return distance

class species:
	def __init__(self, representative):
		self.subpopulation = []
		self.representative = representative
		self.subpopulation.append(representative)
		self.avgFitness=0
		self.staleness=0
		self.prevTopFit=0
	def addChromosome(self, newChromosome):
		self.subpopulation.append(newChromosome)

	def removeHalf(self):
		self.subpopulation=sorted(self.subpopulation)
		if len(self.subpopulation)==1 and self.subpopulation[0].fitnessValue>=population.maxFitness:
			return len(self.subpopulation)
		self.subpopulation=self.subpopulation[:math.floor(len(self.subpopulation)/2)]
		return len(self.subpopulation)

	def removeAllExceptOne(self):
		self.subpopulation=sorted(self.subpopulation)
		self.subpopulation=self.subpopulation[:1]
		self.representative=self.subpopulation[0]

	def calcAvgFitness(self):
		if len(self.subpopulation)<1:
			return 0
		self.avgFitness=0
		for chrom in self.subpopulation:
			self.avgFitness+=chrom.fitnessValue
		self.avgFitness=self.avgFitness/len(self.subpopulation)
		return self.avgFitness

	def getChild(self):
		PROBABILITY_crossover=0.75


		if random.random()<PROBABILITY_crossover:
			parent1=self.subpopulation[random.randrange(0,len(self.subpopulation))]
			parent2=self.subpopulation[random.randrange(0,len(self.subpopulation))]
			child=crossover.crossover(parent1,parent2)
			return child
		else:
			child=deepcopy(self.subpopulation[random.randrange(0,len(self.subpopulation))])
			return child

class population:
	maxFitness=0
	globalInnovationNumber=0
	def __init__(self, N):
		self.generationNumber = 0;
		self.numberOfIndividuals = N
		self.index=0
		self.populationSpecies = []

	def nextGen(self):
		self.save()
		totalAvgFit=0
		remaining=0
		avgPopFit=0
		for spec in self.populationSpecies:
			remaining+=spec.removeHalf()
			tmp=spec.calcAvgFitness()
			totalAvgFit+=tmp
			avgPopFit+=(tmp*len(spec.subpopulation))
		children=[]
		#self.removeStale()
		self.removeWeak()
		

		print()
		print("Generation",self.generationNumber)
		print("Max Fitness:",self.maxFitness)
		print("Innovation Number:",self.globalInnovationNumber)
		print("No. of Species:",len(self.populationSpecies))
		print("Avg pop fitness:",avgPopFit/remaining)
		print("Total Population:",self.index-1)
		

		for spec in self.populationSpecies:
			n=math.floor(spec.avgFitness/totalAvgFit)*self.numberOfIndividuals-1
			for i in range(n):
				ch=spec.getChild()
				if ch:
					self.globalInnovationNumber, ch = mutate.mutate(ch,self.globalInnovationNumber)
					children.append(ch)
			spec.removeAllExceptOne()

		while self.numberOfIndividuals > len(children) + len(self.populationSpecies):
			spec = self.populationSpecies[random.randrange(0, len(self.populationSpecies))]
			ch = spec.getChild()
			if ch:
				self.globalInnovationNumber, ch = mutate.mutate(ch, self.globalInnovationNumber)
				children.append(ch)

		M=self.numberOfIndividuals-len(self.populationSpecies)
		for i in range(0,M):
			self.addChromosome(children[i])
		print("Children produced:",M)
		print()

		self.generationNumber+=1
		self.index=0
		self.maxFitness=0

		#self.save()

	
	def removeWeak(self):
		specList=[]
		for spec in self.populationSpecies:
			if len(spec.subpopulation)>=1:
				#print("removeWeak")
				specList.append(spec)

		self.populationSpecies=specList

	def removeStale(self):
		specList=[]
		for spec in self.populationSpecies:
			if spec.subpopulation[0].fitnessValue>spec.prevTopFit:
				spec.staleness=0
			else:
				spec.staleness+=1
			spec.prevTopFit=spec.subpopulation[0].fitnessValue
			if spec.staleness<10 or spec.prevTopFit>=population.maxFitness:
				specList.append(spec)
		self.populationSpecies=specList



	def addChromosome(self, chromosome):
		#toAdd = True
		for spec in self.populationSpecies:
			#print(compatibilityDistance(chromosome, spec.representative))
			if(compatibilityDistance(chromosome, spec.representative) < COMPATIBILITY_RANGE):
				spec.addChromosome(deepcopy(chromosome))
				#toAdd = False
				return;
		#if(toAdd == True):
		self.populationSpecies.append(deepcopy(species(chromosome)))


	def initializePopulation(self):
		for i in range(self.numberOfIndividuals):
			temp = deepcopy(chromosome())
			#for j in range(100):
			self.globalInnovationNumber, temp=mutate.mutate(temp,self.globalInnovationNumber)
			self.addChromosome(temp)
			#print(self.globalInnovationNumber)
	
	def save(self):
		pickle_out = open("savedPopulations/generation"+str(self.generationNumber)+".gen", "wb+")
		cPickle.dump(self, pickle_out)
	
	def copy(self, other):
		self.generationNumber = deepcopy(other.generationNumber)
		self.numberOfIndividuals = deepcopy(other.numberOfIndividuals)
		self.individuals = deepcopy(other.individuals)
	
	def load(self, generationNumber):
		#this initializes first and then loads, can make it efficient
		pickle_in = open("savedPopulations/generation"+str(generationNumber)+".gen", "rb")
		other = cPickle.load(pickle_in)
		self.generationNumber = deepcopy(other.generationNumber)
		self.numberOfIndividuals = deepcopy(other.numberOfIndividuals)
		self.index = deepcopy(other.index)
		self.populationSpecies = deepcopy(other.populationSpecies)
		self.globalInnovationNumber = deepcopy(other.globalInnovationNumber)
		self.maxFitness = deepcopy(other.maxFitness)

	
	def printPopulation(self):
		print(self.generationNumber, self.numberOfIndividuals)
		for i in range(len(self.populationSpecies)):
			print("Species",i,":",self.populationSpecies[i].numIndividuals)

	def fetchNext(self):
		tmp=self.index
		self.index+=1
		for species in self.populationSpecies:
			if tmp>=len(species.subpopulation):
				tmp = tmp-len(species.subpopulation)
			else:
				return species.subpopulation[tmp]
		return

# p = population(10)
# p.printPopulation()
# p.save()
'''p = population(1)
p.printPopulation()
p.load(0)
p.printPopulation()'''


