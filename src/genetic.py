import random
import network



'''
Contains functions of Genetic ALgorithm:
	Natural Selection
	Crossover
	Mutation
	Breeding = Natural Selection + Crossover + Mutation
	Accuracy Measurement
'''

class geneticAlgorithm(network.Network):

	'''
	Initialize the class inheriting the properties of the "network" class
	'''
	def __init__(self, param=None):

		network.Network.__init__(self, param=param)




	'''
	Natural Selection
	get parents details
		Mum is the current network
		Dad is the previous Network
		Returns parents
	'''
	def getParents(self, i, data, numNetworks):
		mum = data[i]
		pRank = (numNetworks + i - 1) % numNetworks  # Take previous network
		dad = data[(pRank)]  # Take previous network
		return mum, dad




	'''
	Crossover:
		Multiple point crossover
		Randomly choose the value of every position
		And substitute the value with the value of any of the parent
		Returns the network of the child after crossover
	'''
	def crossover(self, mum, dad):
		self.network = {}
		for key in self.param:
			i = random.randint(0, 1)
			if i == 0:
				self.network[key] = mum[key]
			else:
				self.network[key] = dad[key]
			i = None
		return self.network




	'''
	Mutation:
		Keeping mutation chance to be 30%
		Mutate one of the params
		Returns the network of the child after mutation
	'''
	def mutation(self, child_data, mutationChance):

		# probability of mutation
		rand = random.randint(0, 100)

		if rand<mutationChance:

			#choose a random key
			# the value of which is to be mutated
			#optimizer, e.g.
			mutationPoint = random.choice(list(self.param.keys()))

			for key in self.param:
				if key==mutationPoint:
					# Substitute the key with random value
					child_data[key] = random.choice(list(self.param[mutationPoint].values()))
		rand = None
		mutationPoint = None
		return child_data




	'''
	Breeding
	Natural Selection, Crossover & Mutation
	Returns the network of the child
	'''
	def breeding(self, i, data, mutationChance, numNetworks):

		# NATURAL SELECTION
		mum, dad = self.getParents(i, data, numNetworks)

		# CROSSOVER
		child = self.crossover(mum, dad)

		# MUTATION
		child = self.mutation(child, mutationChance)

		return child




	'''
	Accuracy of the network
		Network is trained
		Accuracy is retuned after training
	'''
	def getFitness(self, dt, dataset):

		self.network = {}
		for key in self.param:
			self.network[key] = dt[key]
		self.fitness = (self.train(dataset))*100
		return self.fitness

