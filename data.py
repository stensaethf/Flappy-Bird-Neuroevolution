import sys, random, cPickle, numpy as np

def genData(num):
	"""
	Function for generating a dataset.
	"""
	data = []
	for i in range(num):
		distance = random.uniform(0, 1)
		bird_height = random.uniform(0, 1)
		hole_height = random.uniform(0, 1)
		if bird_height > hole_height:
			data.append([bird_height, hole_height, distance, 1.0])
		else:
			data.append([bird_height, hole_height, distance, 0.0])

	cPickle.dump(data, open('data/flappyBirdData.pickle', 'wb'))

def loadData(structure):
	"""
	Function for loading the dataset.
	"""
	# structure = [x, y, ..., z]
	# flappyBirdData.pkl --> 
	# 	[[bird height, hole height, distance, flap/no flap], ...]
	data = cPickle.load(open('data/flappyBirdData.pickle', 'rb'))
	return reshape(data, structure)

def reshape(data, structure):
	"""
	Function for reshaping the dataset. We want the inputs to be numpy arrays
	of 3x1 or 2x1 and the outputs to be numpy arrays of 1x1.
	"""
	# structure = [x, y, ..., z]
	reshaped_inputs = []
	reshaped_outputs = []
	for i in data:
		# check for how many inputs the network has.
		num_inputs = structure[0]
		if num_inputs == 3:
			inputs = i[:-1]
		elif num_inputs == 2:
			inputs = i[:-2]
		else:
			print 'Invalid number of inputs.'
			sys.exit()
		outputs = i[-1]

		# reshape inputs and outputs.
		reshaped_input = np.reshape(inputs, (num_inputs, 1))
		reshaped_output = np.reshape(outputs, (1, 1))

		reshaped_inputs.append(reshaped_input)
		reshaped_outputs.append(reshaped_output)

	return (reshaped_inputs, reshaped_outputs)

def main():
	# python genData.py <num data points> <structure>

	genData(int(sys.argv[1]))
	if len(sys.argv) != 3:
		generated_data = loadData([3, 7, 1])
	else:
		structure = [int(i) for i in sys.argv[2].split('/')]
		generated_data = loadData(structure)
	for i in generated_data:
		print i

if __name__ == '__main__':
	main()