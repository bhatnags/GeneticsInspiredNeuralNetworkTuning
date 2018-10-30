import random

from keras.datasets import cifar10, cifar100
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.utils.np_utils import to_categorical
from keras.callbacks import EarlyStopping

earlyStopper = EarlyStopping(monitor='val_loss', min_delta=0.1, patience=5, verbose=0, mode='auto')

'''
Contains functions of Network:
	Initializes the network class without any hyper-parameter set
	Initializes network with randomly selected hyper-parameter
	Get number of classification classes in the dataset
	Get the details of the training and test dataset
	Train the dataset
'''


class Network():
    '''
	Initializes the network class
		and it's features - fitness, param and network
		without any hyper-parameter set (param is set to none)
	'''

    def __init__(self, param=None):
        self.fitness = 0
        self.param = param
        self.network = {}

    '''
	Initializes network with randomly selected hyper-parameter
		Returns the network
	'''

    def initNetwork(self):
        for key in self.param:
            rand = random.sample(list(self.param[key]), 1)
            self.network[key] = self.param[key][rand[0]]
            rand = None
        return self.network

    '''
	Get number of classification classes in the dataset
	'''

    def getnbClasses(self, dataset):
        if dataset == 'cifar10':
            nbClasses = 10
        elif dataset == 'cifar100':
            nbClasses = 100
        return nbClasses

    '''
	Get the details of the training and test dataset	
	'''

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

    '''
	Train the dataset
	Returns the accuracy
	'''

    def train(self, dataset):
        batchSize = 64
        input_shape = (3072,)

        # Get number of classification classes
        nbClasses = self.getnbClasses(dataset)

        # Fetch details of the dataset
        x_train, y_train, x_test, y_test = self.getData(dataset, nbClasses)

        # Get details of the neural network to be designed
        nbLayers = self.network['nbLayers']
        nbNeurons = self.network['nbNeurons']
        activation = self.network['activation']
        optimizer = self.network['optimizer']
        dropout = self.network['dropout']

        # Initializes the model type to be trained
        model = Sequential()

        # Create the neural network
        # Add the layers in the neural network
        for i in range(nbLayers):
            if i == 0:
                model.add(Dense(nbNeurons, activation=activation, input_shape=input_shape))
            else:
                model.add(Dense(nbNeurons, activation=activation))
            model.add(Dropout(dropout))
        model.add(Dense(nbClasses, activation='softmax'))

        # Compile the model
        model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

        # Fit the model 
        model.fit(x_train, y_train, batch_size=batchSize, epochs=10, verbose=0, validation_data=(x_test, y_test),
                  callbacks=[earlyStopper])

        # Get the fitness of the model
        # Accuracy and Error
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


'''
Class to make comparisons of the values
	Compare the network fitness
	Get the array indexes of the best and worst fitness data
'''


class compare():
    '''
	Compare the network fitness
	Compares the network's fitness in the previous generation with
		parent fitness, and
		child fitness
	If there has been any improvisation:
		Checks the best of two fitness (parent & child)
		And accordingly substitutes the network fitness and the data
	Returns the network fitness and the data

	'''

    def networkData(self, i, networkFitness, fitnessMum, fitnessChild, data, child):
        if (networkFitness[i] < fitnessMum[i]) and (networkFitness[i] < fitnessChild[i]):
            if fitnessMum[i] > fitnessChild[i]:
                networkFitness[i] = fitnessMum[i]
            else:
                networkFitness[i] = fitnessChild[i]
                data[i] = child
        return networkFitness, data

    '''
	Get the index of the best fitness the generation
	Get the index of the worst fitness the generation
	'''

    def BestAndPoorest(networkFitness):
        maxIndex = max(networkFitness)
        minIndex = min(networkFitness)
        return maxIndex, minIndex


'''
To get the best fitness of the generation	
Returns the best fitness of the generation and the data
'''


def genFitness(networkFitness, data, param):
    # Get the generation best fitness, if it's better then proceed
    maxIndex, minIndex = compare.BestAndPoorest(networkFitness)

    # The network with the worst fitness is deleted and reinitialized looking for better possibilities
    data[minIndex] = Network(param).initNetwork()

    return networkFitness[maxIndex], data

