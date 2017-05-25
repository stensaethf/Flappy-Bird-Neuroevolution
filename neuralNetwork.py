# NeuralNetwork
# Neural network implementation.
# Frederik Roenn Stensaeth.
# 05.17.17

import numpy as np, random, cPickle, sys, data
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

	def backpropagate(self, example, label):
		"""
		Performs backpropagation on an example and label.
		Updates the weights and biases in the neural network by propagating
		backwards the change calculated for the output layer.
		"""
		# Find changes that needs to be made to the weights and biases.
		weights_change, bias_change = self.getWeightsChange(example, label)

		# Update every weight and bias in network with the found changes.
		for l in range(len(self.layers) - 1):
			self.weights[l] = (
				self.weights[l] + 
				self.alpha * weights_change[l]
			)
			self.bias[l] = (
				self.bias[l] + 
				self.alpha * bias_change[l]
			)

	def getWeightsChange(self, x, y):
		"""
		Calcualates the changes needed to be made to each weight and bias
		given an x (example) and a y (label).
		"""
		# Feed forward the x value to find thresholds and activations.
		thresholds, acts = self.feedForward(x)

		# Propagate deltas backward from output layer to input layer.
		# Start by calculating the deltas of the output layer.
		delta = (
			self.sigmoidPrime(thresholds[-1]) *
			self.error(y, acts[-1])
		)

		# Each bias and weight has a change associated with it, so
		# let's create matrices of the same structure as we already have.
		weights_change = []
		for weights in self.weights:
			weights_change.append(np.zeros(weights.shape))

		bias_change = []
		for bias in self.bias:
			bias_change.append(np.zeros(bias.shape))

		# The change made to the weights are the dot product of the
		# delta for that layer and the activations of the previous
		# layer.
		# Activations for biases are always 1, so the change needed
		# is just the delta.
		weights_change[-1] = np.dot(delta, acts[-2].transpose())
		bias_change[-1] = delta

		# Now we want to find the changes needed to be made to the
		# rest of the weights and biases and apply them.
		for l in range(2, len(self.layers)):
			delta = (
				self.sigmoidPrime(thresholds[-l]) *
				np.dot(self.weights[-l + 1].transpose(), delta)
			)

			bias_change[-l] = delta
			weights_change[-l] = np.dot(delta, acts[-l - 1].transpose())

		return weights_change, bias_change

	def train(self, examples, labels, alpha, iterations):
		"""
		Trains the weights and biases of the neural network using examples
		and labels.
		"""
		iterations = iterations
		for t in range(iterations):
			# Calculate decaying alpha.
			self.alpha = alpha - (alpha * t / iterations)

			# Shuffle the data so that we see the data in different
			# orders while training.
			examples, labels = self.doubleShuffle(examples, labels)

			for e in range(len(examples)):
				# Backpropogate the example and label.
				self.backpropagate(examples[e], labels[e])

			# Check how many examples we classify correctly.
			self.numberCorrect(examples, labels)

	def error(self, label, activation):
		"""
		Calculates the difference between a label and an activation.
		"""
		return (label - activation)

	def numberCorrect(self, inputs_list, labels):
		"""
		Classifies a given set of inputs and labels using the trained
		weights and biases. Prints the number of correct classifications.
		"""
		count = 0
		for i in range(len(inputs_list)):
			inputs = inputs_list[i]
			label = labels[i]

			# Feed forward the inputs.
			thresholds, acts = self.feedForward(inputs)

			# Check whether the classification was correct.
			if label[0][0] == 1.0:
				if acts[-1][0][0] > 0.5:
					# we found that we should flap --> correct.
					count += 1
			else:
				if acts[-1][0][0] <= 0.5:
					# we found that we should not flap --> correct.
					count += 1

		print str(count) + "/" + str(len(inputs_list))

	def doubleShuffle(self, list1, list2):
		"""
		Shuffles two corresponding lists of equal length.
		"""
		list1_new = []
		list2_new = []
		index_shuf = range(len(list1))
		# Randomly shuffle the input
		random.shuffle(index_shuf)
		for i in index_shuf:
			list1_new.append(list1[i])
			list2_new.append(list2[i])

		return list1_new, list2_new

	def sigmoid(self, x):
		"""
		Calculates the sigmoid value for a given x.
		"""
		return 1.0 / (1 + np.exp(-x))

	def sigmoidPrime(self, x):
		"""
		Calculates the sigmoid prime value for a given x.
		"""
		return self.sigmoid(x) * (1.0 - self.sigmoid(x))

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

def main():
	alpha = 0.3
	layers = [3, 7, 1]
	iterations = 10

	data.genData(10000)
	train_inputs, train_outputs = data.loadData(layers)

	network = NeuralNetwork(layers)
	network.train(train_inputs, train_outputs, alpha, iterations)

if __name__ == '__main__':
	main()