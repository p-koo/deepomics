from __future__ import print_function
import tensorflow as tf 
from tfomics import layers
from tfomics import init, utils

from collections import OrderedDict

__all__ = [
	"NeuralBuild"
]

class NeuralBuild():
	def __init__(self, model_layers, supervised=True):
		self.model_layers = model_layers
		self.placeholders = {}
		self.placeholders['inputs'] = []
		self.lastlayer = ''
		self.num_dropout = 0

		self.hidden_feed_dict = {}
		self.is_training = tf.placeholder(tf.bool, name='is_training')
		self.hidden_feed_dict[self.is_training] = True

		self.network = OrderedDict()	
		self.build_layers()

		if supervised:
			targets = utils.placeholder(shape=(None, model_layers[-1]['num_units']), name='output')
			self.placeholders['targets'] = targets
			self.network['output'] = self.network[self.lastlayer]
		else:
			self.placeholders['targets'] = self.placeholders['inputs']
			
	def get_network_build(self):
		return self.network, self.placeholders, self.hidden_feed_dict

	def build_layers(self):

		# loop to build each layer of network
		for model_layer in self.model_layers:
			layer = model_layer['layer']
			name = model_layer['name']	

			if layer == "input":

				# add input layer
				self.single_layer(model_layer)

			else:
				if layer == 'residual-conv1d':
					conv1d_residual_block(model_layer)

				elif layer == 'residual-conv2d':
					conv2d_residual_block(model_layer)

				elif layer == 'residual-dense':
					dense_residual_block(model_layer)

				else:
					# add core layer
					self.single_layer(model_layer)
					
					# add bias layer
					if 'b' in model_layer:
						newlayer = name+'_bias'
						self.network[newlayer] = layers.BiasLayer(self.network[self.lastlayer], b=model_layer['b'])
						self.lastlayer = newlayer    


			# add Batch normalization layer
			if 'norm' in model_layer:
				if 'batch' in model_layer['norm']:
					newlayer = name + '_batch' #str(counter) + '_' + name + '_batch'
					self.network[newlayer] = layers.BatchNormLayer(self.network[self.lastlayer], self.is_training)
					self.lastlayer = newlayer

			# add activation layer
			if 'activation' in model_layer:
				newlayer = name+'_active'
				self.network[newlayer] = layers.ActivationLayer(self.network[self.lastlayer], function=model_layer['activation']) 
				self.lastlayer = newlayer

			# add max-pooling layer
			if 'pool_size' in model_layer:  
				newlayer = name+'_pool'  # str(counter) + '_' + name+'_pool' 
				if isinstance(model_layer['pool_size'], (tuple, list)):
					self.network[newlayer] = layers.MaxPool2DLayer(self.network[self.lastlayer], pool_size=model_layer['pool_size'])
				else:
					self.network[newlayer] = layers.MaxPool2DLayer(self.network[self.lastlayer], pool_size=(model_layer['pool_size'], 1))
				self.lastlayer = newlayer       

			# add dropout layer
			if 'dropout' in model_layer:
				newlayer = name+'_dropout' # str(counter) + '_' + name+'_dropout'

				if model_layer['dropout']:
					dropout = model_layer['dropout']
					placeholder_name = 'keep_prob'+str(self.num_dropout)
					exec(placeholder_name+" = tf.placeholder(tf.float32, name='"+placeholder_name+"')")
					#exec("self.placeholders["+placeholder_name+"] = " + placeholder_name)				
					exec("self.hidden_feed_dict[" + placeholder_name+"] = "+str(dropout))
					self.num_dropout += 1

				self.network[newlayer] = layers.DropoutLayer(self.network[self.lastlayer], keep_prob=model_layer['dropout'])
				self.lastlayer = newlayer


	def single_layer(self, model_layer):
		""" build a single layer"""

		name = model_layer['name']

		# input layer
		if model_layer['layer'] == 'input':

			input_shape = str(model_layer['input_shape'])
			exec(name+"=utils.placeholder(shape="+input_shape+", name='"+name+"')")	
			exec("self.network['"+model_layer['name']+"'] = layers.InputLayer("+name+")")
			exec("self.placeholders['inputs'].append(" + name + ")")


		# dense layer
		elif model_layer['layer'] == 'dense':
			if 'W' not in model_layer.keys():
				model_layer['W'] = init.HeNormal()
			if 'b' not in model_layer.keys():
				model_layer['b'] = init.Constant(0.05)
			self.network[name] = layers.DenseLayer(self.network[self.lastlayer], num_units=model_layer['num_units'],
												 W=model_layer['W'],
												 b=model_layer['b'])

		# convolution layer
		elif (model_layer['layer'] == 'conv2d'):

			if 'W' not in model_layer.keys():
				W = init.HeUniform()
			else:
				W = model_layer['W']
			if 'padding' not in model_layer.keys():
				padding = 'VALID'
			else:
				padding = model_layer['padding']
			if 'strides' not in model_layer.keys():
				strides = (1, 1)
			else:
				strides = model_layer['strides']

			self.network[name] = layers.Conv2DLayer(self.network[self.lastlayer], num_filters=model_layer['num_filters'],
												  filter_size=model_layer['filter_size'],
												  W=W,
												  padding=padding,
												  strides=strides)
			
		elif model_layer['layer'] == 'conv1d':
			if 'W' not in model_layer.keys():
				W = init.HeNormal()
			else:
				W = model_layer['W']
			if 'padding' not in model_layer.keys():
				padding = 'VALID'
			else:
				padding = model_layer['padding']
			if 'strides' not in model_layer.keys():
				strides = 1
			else:
				strides = model_layer['strides']


			self.network[name] = layers.Conv1DLayer(self.network[self.lastlayer], num_filters=model_layer['num_filters'],
												  filter_size=model_layer['filter_size'],
												  W=W,
												  padding=padding,
												  strides=strides)
		self.lastlayer = model_layer['name']


	def conv1d_residual_block(self, model_layer):

		lastlayer = self.lastlayer

		name = model_layer['name']
		filter_size = model_layer['filter_size']
		is_training = model_layer['batch_norm']
		if 'function' in model_layer:
			activation = model_layer['function']
		else:
			activation = 'relu'

		# original residual unit
		shape = self.network[lastlayer].get_output_shape()
		num_filters = shape[-1].value

		if not isinstance(filter_size, (list, tuple)):
			filter_size = (filter_size, 1)

		self.network[name+'_1resid'] = layers.Conv2DLayer(self.network[lastlayer], num_filters=num_filters, filter_size=filter_size, padding='SAME')
		self.network[name+'_1resid_norm'] = layers.BatchNormLayer(self.network[name+'_1resid'], is_training)
		self.network[name+'_1resid_active'] = layers.ActivationLayer(self.network[name+'_1resid_norm'], function=activation)

		if 'dropout_block' in model_layer:
			dropout = model_layer['dropout_block']
			placeholder_name = 'keep_prob'+str(len(self.num_dropout))
			exec(placeholder_name+" = tf.placeholder(tf.float32, name='"+placeholder_name+"')")
			#exec("self.placeholders["+placeholder_name+"] = " + placeholder_name)			
			exec("self.network["+name+"+'_dropout1'] = layers.DropoutLayer(self.network["+name+"+'_1resid_active'], keep_prob="+placeholder_name+")")				
			exec("self.hidden_feed_dict["+placeholder_name+"] ="+str(dropout))
			self.num_dropout += 1
			lastname = name+'_dropout1'
		else:
			lastname = name+'_1resid_active'

		self.network[name+'_2resid'] = layers.Conv2DLayer(self.network[lastname], num_filters=num_filters, filter_size=filter_size, padding='SAME')
		self.network[name+'_2resid_norm'] = layers.BatchNormLayer(self.network[name+'_2resid'], is_training)
		self.network[name+'_resid_sum'] = layers.ElementwiseSumLayer([self.network[lastlayer], self.network[name+'_2resid_norm']])
		self.network[name+'_resid'] = layers.ActivationLayer(self.network[name+'_resid_sum'], function=activation)

		self.lastlayer = name+'_resid'

		return network


	def conv2d_residual_block(self, model_layer):

		lastlayer = self.lastlayer
		name = model_layer['name']
		filter_size = model_layer['filter_size']
		is_training = model_layer['batch_norm']
		if 'function' in model_layer:
			activation = model_layer['function']
		else:
			activation = 'relu'

		# original residual unit
		shape = self.network[lastlayer].get_output_shape()
		num_filters = shape[-1].value

		if not isinstance(filter_size, (list, tuple)):
			filter_size = (filter_size, 1)

		self.network[name+'_1resid'] = layers.Conv2DLayer(self.network[lastlayer], num_filters=num_filters, filter_size=filter_size, padding='SAME')
		self.network[name+'_1resid_norm'] = layers.BatchNormLayer(nself.etwork[name+'_1resid'], is_training)
		self.network[name+'_1resid_active'] = layers.ActivationLayer(self.network[name+'_1resid_norm'], function=activation)


		if 'dropout_block' in model_layer:
			dropout = model_layer['dropout_block']
			placeholder_name = 'keep_prob'+str(len(self.num_dropout))
			exec(placeholder_name+" = tf.placeholder(tf.float32, name='"+placeholder_name+"')")
			#exec("self.placeholders["+placeholder_name+"] = " + placeholder_name)			
			exec("self.network["+name+"+'_dropout1'] = layers.DropoutLayer(self.network["+name+"+'_1resid_active'], keep_prob="+placeholder_name+")")				
			exec("self.hidden_feed_dict["+placeholder_name+"] ="+str(dropout))
			lastname = name+'_dropout1'
			self.num_dropout += 1
		else:
			lastname = name+'_1resid_active'

		self.network[name+'_2resid'] = layers.Conv2DLayer(self.network[lastname], num_filters=num_filters, filter_size=filter_size, padding='SAME')
		self.network[name+'_2resid_norm'] = layers.BatchNormLayer(self.network[name+'_2resid'], is_training)
		self.network[name+'_resid_sum'] = layers.ElementwiseSumLayer([self.network[lastlayer], self.network[name+'_2resid_norm']])
		self.network[name+'_resid'] = layers.ActivationLayer(self.network[name+'_resid_sum'], function=activation)
		return network



	def dense_residual_block(self, model_layer):

		lastlayer = self.lastlayer

		name = model_layer['name']
		is_training = model_layer['batch_norm']
		if 'function' in model_layer:
			activation = model_layer['function']
		else:
			activation = 'relu'

		# original residual unit
		shape = self.network[lastlayer].get_output_shape()
		num_units = shape[-1].value

		self.network[name+'_1resid'] = layers.DenseLayer(self.network[lastlayer], num_units=num_units, b=None)
		self.network[name+'_1resid_norm'] = layers.BatchNormLayer(self.network[name+'_1resid'], is_training)
		self.network[name+'_1resid_active'] = layers.ActivationLayer(self.network[name+'_1resid_norm'], function=activation)

		

		if 'dropout_block' in model_layer:
			dropout = model_layer['dropout_block']
			placeholder_name = 'keep_prob'+str(len(self.num_dropout))
			exec(placeholder_name+" = tf.placeholder(tf.float32, name='"+placeholder_name+"')")
			#exec("self.placeholders["+placeholder_name+"] = " + placeholder_name)			
			exec("self.network["+name+"+'_dropout1'] = layers.DropoutLayer(self.network["+name+"+'_1resid_active'], keep_prob="+placeholder_name+")")				
			exec("self.hidden_feed_dict["+placeholder_name+"] ="+str(dropout))
			lastname = name+'_dropout1'
			self.num_dropout += 1
		else:
			lastname = name+'_1resid_active'

		self.network[name+'_2resid'] = layers.DenseLayer(self.network[lastname], num_units=num_units, b=None)
		self.network[name+'_2resid_norm'] = layers.BatchNormLayer(self.network[name+'_2resid'], is_training)
		self.network[name+'_resid_sum'] = layers.ElementwiseSumLayer([self.network[lastlayer], self.network[name+'_2resid_norm']])
		self.network[name+'_resid'] = layers.ActivationLayer(self.network[name+'_resid_sum'], function=activation)
		return network

