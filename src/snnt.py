#from memory_profiler import profile
import logging
import collections
from collections import OrderedDict
from random import randint
import network
import genetic


'''
Get parameters for optimizing Neural Network
'''
def getParameters():
	# Number of generations
	generation = 5
	# Dataset for comparison
	dataset = 'cifar10'
	# Number of networks OR population size in every generations
	numNetworks = 4
	# Rate of mutation
	mutationChance = 30
	# Hyper-parameters to be optimized
	param = collections.OrderedDict({
		'nbNeurons': {1: 32, 2: 30, 3: 15, 4: 10, 5: 5, 6: 3},  # 2:64, 3:128, 4:256, 5:512, 6:1024},
		'nbLayers': {1: 1, 2: 3, 3: 6, 4: 9, 5: 12, 6: 15},
		'activation': {1: 'sigmoid', 2: 'elu', 3: 'selu', 4: 'relu', 5: 'tanh', 6: 'hard_sigmoid'},
		'optimizer': {1: 'sgd', 2: 'nadam', 3: 'adagrad', 4: 'adadelta', 5: 'adam', 6: 'adamax'},
		'dropout': {1: 0.1, 2: 0.2, 3: 0.25, 4: 0.3, 5: 0.4, 6: 0.5}
	})
	return generation, dataset, numNetworks, mutationChance, param



'''
Neural Network Tuning
'''
#@profile
def NNT():

	# Get Parameters
	generation, dataset, numNetworks, mutationChance, param = getParameters()

	# Get the logger
	filename = 'output.log'
	logger = logging.getLogger()
	handler = logging.FileHandler(filename)
	handler.setLevel(logging.DEBUG)
	formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
	handler.setFormatter(formatter)
	logger.addHandler(handler)
	logger.setLevel(logging.DEBUG)

	# Initialize the classes
	net = network.Network(param)
	ga = genetic.geneticAlgorithm(param)
	com = network.compare()

	# Initialize the population & the fitness
	data = {};				# The networks' data array
	fitnessParent = {};		# The fitness array of the parent
	fitnessChild = {}; 		# The fitness array of the child
	networkFitness = {};	# Array of the better out of two fitness - parent and child
	genBestFitness = {}; 	# Array to save the fitness over the generations

	# Initialize the population
	for i in range(numNetworks):
		data[i] = net.initNetwork()
		fitnessParent[i] = -1
		fitnessChild[i] = -1
		networkFitness[i] = -1


	# Start running GA (Genetic Algorithm) generation
	for g in range(generation):

		# For all networks in each generation
		for i in range(numNetworks):
			print('0', data)

			# GET PARENT FITNESS/ACCURACY
			fitnessParent[i] = ga.getFitness(data[i], dataset)
			print('1' ,data)


			# BREED THE CHILD
			child = ga.breeding(i, data, mutationChance, numNetworks)
			print('2',data)


			# GET CHILD'S FITNESS/ACCURACY
			fitnessChild[i] = ga.getFitness(child, dataset)
			print('3',data)


			'''
			If the network fitness has improved over previous generation, 
				then pass on the features/hyperparameters
			Pass on the better of the two (parent or child) from this generation to the next generation
			'''
			networkFitness, data = com.networkData(i, networkFitness, fitnessParent, fitnessChild, data, child)
			print('4', data)
			logger.debug('generation=%d, Rank=%d, parent=%s, child=%s, '
						 'parentFitness=%0.4f, childFitness=%0.4f, networkFitness=%0.4f',
						 g, i, data[i], child,
						 fitnessParent[i], fitnessChild[i], networkFitness[i])


		'''
		Compare the fitness of the best networks of all the families
		Get the best fitness the generation 
		Kill the poorest performing of the population 
		Randomly initialize the poorest fitness population to keep the population constant
		'''
		genBestFitness[g], data = network.genFitness(networkFitness, data, param)
		print('5', data)
		print(genBestFitness[g], data)

#	del gen
#	del dataset
#	del numNetworks
#	del mutationChance
#	del param
#	del data
#	del net
#	del bestFitness
#	del ga
		

if __name__ == '__main__':
	NNT()
		

