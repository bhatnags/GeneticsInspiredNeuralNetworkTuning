import random

from keras.datasets import cifar10, cifar100
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.utils.np_utils import to_categorical
from keras.callbacks import EarlyStopping


earlyStopper = EarlyStopping( monitor='val_loss', min_delta=0.1, patience=5, verbose=0, mode='auto' )


class Network():

	def __init__(self, param=None):
		self.fitness = 0
		self.param = param
		self.network = {}

	def initNetwork(self):
		for key in self.param:
			rand = random.sample(list(self.param[key]), 1)
			self.network[key] = self.param[key][rand[0]] 
			rand = None
		return self.network
	
	def getnbClasses(self, dataset):
		if dataset == 'cifar10':
			nbClasses = 10
		elif dataset == 'cifar100':
			nbClasses = 100
		return nbClasses

	def getData(self, dataset, nbClasses):
		if dataset == 'cifar10':
			(x_train, y_train), (x_test, y_test) = cifar10.load_data()		
		elif dataset == 'cifar100':
			(x_train, y_train), (x_test, y_test) = cifar100.load_data()		
		
		x_train = x_train.reshape(50000, 3072)
		x_test = x_test.reshape(10000, 3072)
		x_train = x_train.astype('float32')
		x_test = x_test.astype('float32')
		x_train /= 255
		x_test /= 255
		y_train = to_categorical(y_train, nbClasses)
		y_test = to_categorical(y_test, nbClasses)
	
		return x_train, y_train, x_test, y_test
	

	def train(self, dataset):
		batchSize = 64
		input_shape = (3072, )
		
		nbClasses = self.getnbClasses(dataset)
		
		#Fetch data
		x_train, y_train, x_test, y_test = self.getData(dataset, nbClasses)

		nbLayers = self.network['nbLayers']
		nbNeurons = self.network['nbNeurons']
		activation = self.network['activation']
		optimizer = self.network['optimizer']
		dropout = self.network['dropout']

		model = Sequential()

		for i in range(nbLayers):
			if i ==0:
				model.add(Dense(nbNeurons, activation=activation, input_shape = input_shape))
			else:
				model.add(Dense(nbNeurons, activation=activation))
			model.add(Dropout(dropout))
		model.add(Dense(nbClasses, activation='softmax'))

		model.compile(loss = 'categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

		model.fit(x_train, y_train, batch_size=batchSize, epochs=10000, verbose=0, validation_data=(x_test, y_test), callbacks=[earlyStopper])

		fitness = model.evaluate(x_test, y_test, verbose=0)
		

		batchSize = None
		input_shape = None
		nbClasses = None
		x_train = None
		y_train = None
		x_test = None
		y_test = None
		nbLayers = None
		nbNeurons = None
		activation = None
		optimizer = None
		dropout = None
		model = None
		return fitness[1]

