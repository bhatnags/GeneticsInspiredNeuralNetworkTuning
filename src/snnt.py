from memory_profiler import profile
import logging

import collections
from collections import OrderedDict

import network
import genetic

@profile
def NNT():
	"""Give parameters"""
	gen = 30
	dataset = 'cifar10'
	numNetworks = 7
	mutationChance = 2 
	
	param = collections.OrderedDict({
		'nbNeurons': {1:32, 2:64, 3:128, 4:256, 5:512, 6:768, 7:1024},
		'nbLayers': {1:1, 2:3, 3:6, 4:9, 5:12, 6:15, 7:20},
		'activation': {1:'sigmoid', 2:'elu', 3:'selu', 4:'relu', 5:'tanh', 6:'hard_sigmoid', 7:'linear'}, 
		'optimizer': {1:'sgd', 2:'rmsprop', 3:'adagrad', 4:'adadelta', 5:'adam', 6:'adamax', 7:'nadam'},
		'dropout': {1:0.1, 2:0.15, 3:0.2, 4:0.25, 5:0.3, 6:0.4, 7:0.5}
	})
	
	filename = 'output.log'
	logger = logging.getLogger()
	handler = logging.FileHandler(filename)
	handler.setLevel(logging.DEBUG)
	formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
	handler.setFormatter(formatter)
	logger.addHandler(handler)
	logger.setLevel(logging.DEBUG)


	# Initialize the population
	data = {}
	for i in range(0, numNetworks):
		net = network.Network(param)
		data[i] = net.initNetwork()
		logger.debug('Initialized network = %d, %s', i, data[i])
	
	bestFitness = -1
	ga = genetic.geneticAlgorithm(param)

	for g in range(gen):
		#for all networks in each generation
		for i in range(0, numNetworks):
			if(bestFitness < 100):
				#Natural Selection & Crossover, part I of Genetic Algorithm
				mum = data[i]
				#print("parent", data[i])
				fitnessMum = ga.getFitness(data[i], dataset)

				pRank = (numNetworks + i - 1)%numNetworks #Take previous network
				dad = data[(pRank)] #Take previous network
				logger.debug('Prev Rank = %d, dad = %s', pRank, dad)
				#Crossover
				child = {}
				child[i] = ga.crossover(mum, dad)
				logger.debug('Crossover Done, generation = %d, child = %d, %s', g, i, child[i])

				#Mutation of child
				ga.mutation(child[i], mutationChance)
				logger.debug('Mutation Done, generation = %d, child = %d, %s', g, i, child[i])
				#print("child", child[i])

				#train both the child and mum
				#select one best out of the parent or child for next generation
				fitnessChild = ga.getFitness(child[i], dataset)

				logger.debug('Training Done, generation = %d, networks = %d, parentFitness = %0.4f, childFitness = %0.4f', g, i, fitnessMum, fitnessChild)

				#print("after fitness calculation................")
				#print("data", data[i])
				#print("child", child[i])
				if (bestFitness<fitnessChild) or (bestFitness<fitnessMum):
					if fitnessChild > fitnessMum:
						data[i] = child[i]
						bestFitness = fitnessChild
					else:
						data[i] = mum
						bestFitness = fitnessMum
				'''
				Profiling memory
				'''
				pRank = None
				child = None
				mum = None
				dad = None
				fitnessMum = None
				fitnessChild = None
			else:
				print "100% fitness achieved"
				print data[i]
		
	del gen
	del dataset
	del numNetworks
	del mutationChance
	del param
	del data
	del net
	del bestFitness
	del ga 
		

if __name__ == '__main__':
	NNT()
		



