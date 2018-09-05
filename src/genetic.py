import random
import network

class geneticAlgorithm(network.Network):

	def __init__(self, param=None):

		network.Network.__init__(self, param=param)

	def crossover(self, mum, dad):
		# logging.warning('doing crossover~~~~~~~~~~~`')
		for key in self.param: 
			i = random.randint(0, 1)
			if i == 0:
				self.network[key] = mum[key] 
			else:
				self.network[key] = dad[key]
			i = None
		return self.network

	def mutation(self, child, mutationChance):

		# probability of mutation
		rand = random.randint(0, 100)
		
		if rand<mutationChance:
			#choose a random key  #optimizer, e.g.
			mutationPoint = random.choice(list(self.param.keys()))
			for key in self.param:
				if key==mutationPoint: 
					#substitute the key with random value
					# Mutate one of the params.
					child[key] = (random.choice(self.param[mutationPoint].values()))
		rand = None
		mutationPoint = None

	def getFitness(self, data, dataset):
		for key in self.param: 
			self.network[key] = data[key] 
		self.fitness = (self.train(dataset))*100
		return self.fitness
