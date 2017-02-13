from __future__ import print_function 
import os, sys, time
import tensorflow as tf
import numpy as np
from .optimize import *
from .metrics import *
from .utils import *


__all__ = [
	"NeuralNet",
	"NeuralTrainer",
	"MonitorPerformance"
]



#------------------------------------------------------------------------------------------
# Neural Network model class
#------------------------------------------------------------------------------------------



class NeuralNet:
	"""Class to build a neural network and perform basic functions."""
	
	def __init__(self, network, input_vars):
		self.network = network
		self.input_vars = input_vars
		
		self.saver = tf.train.Saver()


	def inspect_layers(self):
		"""print(each layer type and parameters"""

		print('----------------------------------------------------------------------------')
		print('Network architecture:')
		print('----------------------------------------------------------------------------')
		counter = 1
		for layer in self.network:
			output_shape = self.network[layer].get_output_shape()

			print('layer'+str(counter) + ': ' + layer )
			print(output_shape)
			counter += 1
		print('----------------------------------------------------------------------------')

	
	def get_parameters(self, sess, layer=[]):
		"""return all the parameters of the network"""
		if layer:
			return sess.run(get_layer_parameters(self.network, layer=layer))
		else:
			return sess.run(get_all_parameters(self.network))


	def get_activations(self, sess, layer, X, batch_size=500):
		"""get the feature maps of a given convolutional layer"""
		
		batch_generator = BatchGenerator(X, self.placeholders, batch_size, shuffle=False)
			
		fmaps = []
		for i in range(batch_generator.get_num_batches()):  
			feed_dict = batch_generator.next_minibatch(X) 
			fmaps.append(self.sess.run(self.network[layer].get_output(), feed_dict=feed_dict))
		fmaps = np.vstack(fmaps)
		return fmaps



	def save_model_parameters(self, sess, filepath='model.ckpt'):
		"""save model parameters to a file"""
		print("saving model to: ", filepath)
		self.saver.save(sess, save_path=filepath)
		

	def load_model_parameters(self, sess, filepath='model.ckpt'):
		"""initialize network with all_param_values"""

		print("loading model from: ", filepath)
		self.saver.restore(sess, filepath)




#----------------------------------------------------------------------------------------------------
# Train neural networks class
#----------------------------------------------------------------------------------------------------


class NeuralTrainer():

	def __init__(self, nnmodel, placeholders, optimization, save='best', filepath='.'):
		self.nnmodel = nnmodel
		self.input_vars = nnmodel.input_vars
		self.targets = placeholders['targets']
		self.placeholders = placeholders
		
		self.optimization = optimization    
		self.objective = optimization['objective']
		self.save = save

		self.filepath = filepath
		
		# get predictions
		self.predictions = get_predictions(nnmodel.network, optimization['objective'])
		self.loss = build_loss(nnmodel.network, self.predictions, self.targets, optimization)

		# setup optimizer
		self.updates = build_updates(optimization)

		# get list of trainable parameters (default is trainable)
		trainable_params = get_trainable(nnmodel.network)
		self.train_step = self.updates.minimize(self.loss, var_list=trainable_params)


		self.train_monitor = MonitorPerformance(name="train", objective=self.objective, verbose=1)
		self.test_monitor = MonitorPerformance(name="test", objective=self.objective, verbose=1)
		self.valid_monitor = MonitorPerformance(name="cross-validation", objective=self.objective, verbose=1)

		
	def train_epoch(self, sess, feed_X, batch_size=128, verbose=1, shuffle=True):        
		"""Train a mini-batch --> single epoch"""

		# set timer for epoch run
		performance = MonitorPerformance('train', self.objective, verbose)
		performance.set_start_time(start_time = time.time())

		batch_generator = BatchGenerator(feed_X['inputs'], self.placeholders, batch_size, shuffle)
		num_batches = batch_generator.get_num_batches()

		value = 0
		for i in range(num_batches):
			feed_dict = batch_generator.next_minibatch(feed_X)     
			results = sess.run([self.train_step, self.loss, self.predictions], feed_dict=feed_dict)           
			value += self.train_metric(results[2], feed_dict[self.targets])
			performance.add_loss(results[1])
			performance.progress_bar(i+1., num_batches, value/(i+1))
		if verbose:
			print("")
		return performance.get_mean_loss()


	def train_metric(self, predictions, y):
		"""metric to monitor performance during training"""

		if self.objective == 'categorical':
			return np.mean(np.argmax(predictions, axis=1) == np.argmax(y, axis=1))
		
		elif self.objective == 'binary':
			return np.mean(np.round(predictions) == y)
		
		elif self.objective == 'squared_error':
			
			num_dims = y.shape[1]
			
			C = 0
			for i in range(num_dims):
				C += np.corrcoef(predictions[:,i],y[:,i])[0][1]
			return C/num_dims

		
		
	def test_model(self, sess, feed_X, batch_size=128, name='test', verbose=1):
		"""perform a complete forward pass, store and print(results)"""

		performance = MonitorPerformance('test',self.objective, verbose)
		batch_generator = BatchGenerator(feed_X['inputs'], self.placeholders, batch_size, shuffle=False)

		label = []
		prediction = []
		for i in range(batch_generator.get_num_batches()):
			feed_dict = batch_generator.next_minibatch(feed_X)         
			results = sess.run([self.loss, self.predictions], feed_dict=feed_dict)          
			performance.add_loss(results[0])
			prediction.append(results[1])
			label.append(feed_dict[self.targets])
		prediction = np.vstack(prediction)
		label = np.vstack(label)
		test_loss = performance.get_mean_loss()

		if verbose:
			if name == "train":
				self.train_monitor.update(test_loss, prediction, label)
				self.train_monitor.print_results(name)
			elif name == "valid":
				self.valid_monitor.update(test_loss, prediction, label)
				self.valid_monitor.print_results(name)
			elif name == "test":
				self.test_monitor.update(test_loss, prediction, label)
				self.test_monitor.print_results(name)
		return test_loss


	def add_loss(self, loss, name):
		"""add loss score to monitor class"""

		if name == "train":
			self.train_monitor.add_loss(loss)
		elif name == "valid":
			self.valid_monitor.add_loss(loss)
		elif name == "test":
			self.test_monitor.add_loss(loss)


	def save_model(self, sess, epoch=None):
		"""save model parameters to file, according to filepath"""

		if self.save == 'best':
			min_loss, min_epoch = self.valid_monitor.get_min_loss()
			if self.valid_monitor.loss[-1] <= min_loss:
				print('lower cross-validation found')
				filepath = self.filepath + '_best.ckpt'
				self.nnmodel.save_model_parameters(sess, filepath)
		elif self.save == 'all':
			epoch = len(self.valid_monitor.loss)
			filepath = self.filepath + '_' + str(epoch) + '.ckpt'
			self.nnmodel.save_model_parameters(sess, filepath)


	def save_all_metrics(self, filepath):
		"""save all performance metrics"""

		self.train_monitor.save_metrics(filepath)
		self.test_monitor.save_metrics(filepath)
		self.valid_monitor.save_metrics(filepath)


	def early_stopping(self, current_loss, current_epoch, patience):
		"""check if validation loss is not improving and stop after patience
		runs out"""

		min_loss, min_epoch = self.valid_monitor.get_min_loss()
		status = True

		if min_loss < current_loss:
			if patience - (current_epoch - min_epoch) < 0:
				status = False
				print("Patience ran out... Early stopping.")
		return status


	def set_best_parameters(self, sess, filepath=[]):
		""" set the best parameters from file"""
		
		if not filepath:
			filepath = self.filepath + '_best.ckpt'

		self.nnmodel.load_model_parameters(sess, filepath)



#----------------------------------------------------------------------------------------------------
# Monitor performance metrics class
#----------------------------------------------------------------------------------------------------

class MonitorPerformance():
	"""helper class to monitor and store performance metrics during 
	   training. This class uses the metrics for early stopping. """

	def __init__(self, name='', objective='binary', verbose=1):
		self.name = name
		self.objective = objective
		self.verbose = verbose
		self.loss = []
		self.metric = []
		self.metric_std = []


	def set_verbose(self, verbose):
		self.verbose = verbose


	def add_loss(self, loss):
		self.loss = np.append(self.loss, loss)


	def add_metrics(self, scores):
		self.metric.append(scores[0])
		self.metric_std.append(scores[1])


	def get_length(self):
		return len(self.loss)


	def update(self, loss, prediction, label):
		scores = calculate_metrics(label, prediction, self.objective)
		self.add_loss(loss)
		self.add_metrics(scores)


	def get_mean_loss(self):
		return np.mean(self.loss)


	def get_metric_values(self):
		return self.metric[-1], self.metric_std[-1]


	def get_min_loss(self):
		min_loss = min(self.loss)
		min_index = np.argmin(self.loss)
		return min_loss, min_index


	def set_start_time(self, start_time):
		self.start_time = start_time


	def print_results(self, name):
		if self.verbose == 1:
			if name == 'test':
				name += ' '

			print("  " + name + " loss:\t\t{:.5f}".format(self.loss[-1]))
			mean_vals, error_vals = self.get_metric_values()

			if (self.objective == "binary") | (self.objective == "categorical"):
				print("  " + name + " accuracy:\t{:.5f}+/-{:.5f}".format(mean_vals[0], error_vals[0]))
				print("  " + name + " auc-roc:\t{:.5f}+/-{:.5f}".format(mean_vals[1], error_vals[1]))
				print("  " + name + " auc-pr:\t\t{:.5f}+/-{:.5f}".format(mean_vals[2], error_vals[2]))
			elif (self.objective == 'squared_error'):
				print("  " + name + " Pearson's R:\t{:.5f}+/-{:.5f}".format(mean_vals[0], error_vals[0]))
				print("  " + name + " rsquare:\t{:.5f}+/-{:.5f}".format(mean_vals[1], error_vals[1]))
				print("  " + name + " slope:\t\t{:.5f}+/-{:.5f}".format(mean_vals[2], error_vals[2]))


	def progress_bar(self, epoch, num_batches, value, bar_length=30):
		if self.verbose == 1:
			remaining_time = (time.time()-self.start_time)*(num_batches-epoch)/epoch
			percent = epoch/num_batches
			progress = '='*int(round(percent*bar_length))
			spaces = ' '*int(bar_length-round(percent*bar_length))
			if (self.objective == "binary") | (self.objective == "categorical"):
				sys.stdout.write("\r[%s] %.1f%% -- time=%ds -- loss=%.5f -- accuracy=%.2f%%  " \
				%(progress+spaces, percent*100, remaining_time, self.get_mean_loss(), value*100))
			elif (self.objective == 'squared_error'):
				sys.stdout.write("\r[%s] %.1f%% -- time=%ds -- loss=%.5f -- correlation=%.5f  " \
				%(progress+spaces, percent*100, remaining_time, self.get_mean_loss(), value))
			sys.stdout.flush()


	def save_metrics(self, filepath):
		savepath = filepath + "_" + self.name +"_performance.pickle"
		print("saving metrics to " + savepath)

		f = open(savepath, 'wb')
		cPickle.dump(self.name, f, protocol=cPickle.HIGHEST_PROTOCOL)
		cPickle.dump(self.loss, f, protocol=cPickle.HIGHEST_PROTOCOL)
		cPickle.dump(self.metric, f, protocol=cPickle.HIGHEST_PROTOCOL)
		cPickle.dump(self.metric_std, f, protocol=cPickle.HIGHEST_PROTOCOL)
		f.close()


#--------------------------------------------------------------------------------------------------
# Batch Generator Class
#--------------------------------------------------------------------------------------------------


class BatchGenerator():
	def __init__(self, X, placeholders, batch_size=128, shuffle=False):

		if isinstance(X, list):
			self.num_data = X[0].shape[0]
		else:
			self.num_data = X.shape[0]

		self.placeholders = placeholders
		self.current_batch = 0

		self.generate_minibatches(batch_size, shuffle)

	def generate_minibatches(self, batch_size=128, shuffle=False):

		if shuffle:
			index = np.random.permutation(self.num_data)
		else:
			index = range(self.num_data)

		self.batch_size = batch_size
		self.num_batches = self.num_data // self.batch_size

		self.indices = []
		for i in range(self.num_batches):
			self.indices.append(index[i*self.batch_size:i*self.batch_size+self.batch_size])

		# get remainder
		index = range(self.num_batches*self.batch_size, self.num_data)    
		if index:
			self.indices.append(index)
			self.num_batches += 1

		self.current_batch = 0

	def next_minibatch(self, feed_X):
		feed_dict = {}
		for key in self.placeholders.keys():
			if isinstance(self.placeholders[key], list):
				for i in range(len(self.placeholders[key])):
					if X[key][i].shape[0] > 1:
						feed_dict[self.placeholders[key][i]] = feed_X[key][i][self.indices[self.current_batch]]
					else:
						feed_dict[self.placeholders[key][i]] = feed_X[key][i]
			else:
				if key in feed_X.keys():
					if hasattr(feed_X[key], "__len__"):
						feed_dict[self.placeholders[key]] = feed_X[key][self.indices[self.current_batch]]
					else:
						feed_dict[self.placeholders[key]] = feed_X[key]

		self.current_batch += 1
		if self.current_batch == self.num_batches:
			self.current_batch = 0

		return feed_dict

	def get_batch_index(self):
		return self.current_batch

	def get_num_batches(self):
		return self.num_batches


#--------------------------------------------------------------------------------------------------
# helper functions
#--------------------------------------------------------------------------------------------------

def get_predictions(network, objective):
	if objective == 'vae':
		predictions = []
	elif (objective == 'binary') | (objective == 'categorical') | (objective == 'squared_error'):
		predictions = network['output'].get_output()
		
	return predictions



def get_all_parameters(net):    
	params = []
	for layer in net:
		if hasattr(net[layer], 'get_variable'):
			variables = net[layer].get_variable()
			if isinstance(variables, list):
				for var in variables:
					params.append(var.get_variable())
			else:
				params.append(variables.get_variable())
	return params



def get_trainable(net):    
	params = []
	for layer in net:
		if hasattr(net[layer], 'is_trainable'):
			if net[layer].is_trainable():
				variables = net[layer].get_variable()
				if isinstance(variables, list):
					for var in variables:
						params.append(var.get_variable())
				else:
					params.append(variables.get_variable())
	return params



def get_layer_parameters(net, layer):
	params = []
	if hasattr(net[layer], 'get_variable'):
		variables = net[layer].get_variable()
		if isinstance(variables, list):
			for var in variables:
				params.append(var.get_variable())
		else:
			params.append(variables.get_variable())
	else:
		print(layer + " has no parameters")
	return params


	