import sys
sys.path.append('..')
from tfomics import utils, init
from tfomics.build_network import *
import tensorflow as tf


def model(input_shape, num_labels=None):
	# design a neural network model
	
	# placeholders
	inputs = utils.placeholder(shape=input_shape, name='input')
	is_training = tf.placeholder(tf.bool, name='is_training')
	keep_prob = tf.placeholder(tf.float32, name='keep_prob')
	targets = utils.placeholder(shape=(None,num_labels), name='output')
	
	# placeholder dictionary
	placeholders = {'inputs': inputs, 
					'targets': targets, 
					'keep_prob': keep_prob, 
					'is_training': is_training}

	# create model
	layer1 = {'layer': 'input',
						'inputs': inputs,
						'name': 'input'
						}
	layer2 = {'layer': 'conv1d', 
						'num_filters': 25,
						'filter_size': 19,
						'batch_norm': is_training,
						'padding': 'SAME',
						'activation': 'relu',
						'pool_size': 10,
						'name': 'conv1'
						}
	layer3 = {'layer': 'residual-conv1d',
						'filter_size': 5,
						'is_training': is_training,
						'dropout': keep_prob,
						'name': 'resid1'
					 }
	layer4 = {'layer': 'conv1d', 
						'num_filters': 50,
						'filter_size': 6,
						'batch_norm': is_training,
						'padding': 'VALID',
						'activation': 'relu',
						'dropout': keep_prob,
						'pool_size': 5,
						'name': 'conv2'
						}
	layer5 = {'layer': 'dense', 
				'W': init.HeNormal(),
				'b': init.Constant(0.05),
				'num_units': num_labels,
				'activation': 'sigmoid',
				'name': 'dense1'
				}

	#from tfomics import build_network
	model_layers = [layer1, layer2, layer3, layer4, layer5]
	net = build_network(model_layers)

	# optimization parameters
	optimization = {"objective": "binary",
					"optimizer": "adam",
					"learning_rate": 0.001,      
					"l2": 1e-6,
					# "l1": 0, 
					}

	return net, placeholders, optimization

	