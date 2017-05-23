# NeuralNetwork
# Neural network implementation.
# Frederik Roenn Stensaeth.
# 05.17.17

import numpy as np, random, cPickle, sys
from copy import deepcopy

class NeuralNetwork:
	def __init__(self, layers, empty = False):
		"""
        Constructor for the neural network.
        Sets up the weights and biases.

        @params layers - list of the size of the layers in the network.
        	Example: [5, 10] - input and output layers have 5 and 10
        	neurons respectively.
        @return n/a.
        """
		self.layers = layers

		# Sets up the matrices used for storing the weights and biases.
		# Example: weights[0] is the matrix storing weights connecting the
		# 	first and second layers of neurons.
		# Example: weights[0][j][k] is the weight connecting the kth neuron
		# 	in the first layer to the jth neuron in the second layer.
		# The initial weights and biases are random in the range: [-0.5, 0.5].
		self.weights = []
		self.bias = []
		if not empty:
			# The first layer has no connections entering it, so we skip it.
			for i in range(1, len(self.layers)):
				self.weights.append(np.zeros((self.layers[i], self.layers[i - 1])))
				self.bias.append(np.zeros((self.layers[i], 1)))
				for j in range(self.layers[i]):
					self.bias[-1][j][0] = random.uniform(-1, 1)
					for k in range(self.layers[i - 1]):
						self.weights[-1][j][k] = random.uniform(-1, 1)

	# def mutate(self, rate):
	# 	"""
	# 	Mutates the weights and biases of the network according to a given
	# 	rate.
	# 	"""
	# 	for i in range(1, len(self.layers)):
	# 		for j in range(self.layers[i]):
	# 			if random.random() < rate:
	# 				# Mutate bias.
	# 				self.bias[i - 1][j][0] = np.random.normal(0, 0.5)
	# 			for k in range(self.layers[i - 1]):
	# 				if random.random() < rate:
	# 					# Mutate weight.
	# 					self.weights[i - 1][j][k] = np.random.normal(0, 0.5)

	# def crossover(self, individual):
	# 	"""
	# 	Crossover the network with the given individual.
	# 	"""
	# 	# Store weights/bias temporarily.
	# 	old_weights = self.weights[0]
	# 	old_bias = self.bias[0]

	# 	# Get weights/bias from individual.
	# 	ind_weights, ind_bias = individual.getWeightsAndBias()
		
	# 	# Swap weights/bias.
	# 	self.weights[0] = ind_weights[0]
	# 	self.bias[0] = ind_bias[0]
	# 	ind_weights[0] = old_weights
	# 	ind_bias[0] = old_bias
	# 	individual.setWeightsAndBias(ind_weights, ind_bias)

	def feedForward(self, result):
		"""
		Feeds forward an example through the network.
		Returns the various thresholds and activations found along the way.
		"""
		thresholds = []
		acts = [result]
		for layer in range(len(self.weights)):
			weights = self.weights[layer]
			bias = self.bias[layer]
			threshold = np.dot(weights, result) + bias
			thresholds.append(threshold)
			result = self.sigmoid(threshold)
			acts.append(result)
		return thresholds, acts

	def sigmoid(self, x):
		"""
		Calculates the sigmoid value for a given x.
		"""
		return 1.0 / (1 + np.exp(-x))

	def saveWeightsAndBias(self, filename):
		"""
		Pickles (saves) the weights and biases.
		"""
		f = open(filename, "w")
		cPickle.dump({
			'weights' : self.weights,
			'bias' : self.bias
		}, f)
		f.close()

	def loadWeightsAndBias(self, filename):
		"""
		Loads saved weights and biases.
		"""
		f = open(filename, "r")
		weights_bias = cPickle.load(f)
		f.close()
		weights = weights_bias['weights']
		bias = weights_bias['bias']

		self.weights = weights
		self.bias = bias

	def getWeightsAndBias(self):
		"""
		Returns the weights and biases.
		"""
		return [deepcopy(self.weights), deepcopy(self.bias)]

	def setWeightsAndBias(self, weights, bias):
		"""
		Sets the weights/bias.
		"""
		self.weights = weights
		self.bias = bias

	def getStructure(self):
		"""
		Returns the structure of the network.
		"""
		return self.layers

	def isEmpty(self):
		"""
		Checks if network is empty.
		"""
		return self.weights == [] and self.bias == []