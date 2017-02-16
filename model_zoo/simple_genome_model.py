import sys
sys.path.append('..')
from tfomics import utils, init
from tfomics.build_network import *
import tensorflow as tf


def model(input_shape, num_labels=None):
	# design a neural network model
	

	# create model
	layer1 = {'layer': 'input',
						'inputs': inputs,
						'name': 'input'
						}
	layer2 = {'layer': 'conv2d', 
						'num_filters': 25,
						'filter_size': (19,1),
						'W': init.GlorotUniform(),
						'b': init.Constant(0.1),
						#'batch_norm': is_training,
						'padding': 'SAME',
						'activation': 'relu',
						'pool_size': (40,1),
						'name': 'conv1'
						}
	layer3 = {'layer': 'residual-conv2d',
						'filter_size': (5,1),
						'batch_norm': is_training,
						'dropout': keep_prob,
						'pool_size': (40,1),
						'name': 'resid1'
					 }
	layer4 = {'layer': 'dense', 
				'num_units': 128,
				'activation': 'relu',
				'W': init.GlorotUniform(),
				'b': init.Constant(0.1),
				'dropout': keep_prob,
				'name': 'dense1'
				}
	layer5 = {'layer': 'dense', 
				'num_units': num_labels,
				'W': init.GlorotUniform(),
				'b': init.Constant(0.1),
				'activation': 'sigmoid',
				'name': 'dense2'
				}

	#from tfomics import build_network
	model_layers = [layer1, layer2, layer4, layer5]
	net = build_network(model_layers)

	# optimization parameters
	optimization = {"objective": "binary",
					"optimizer": "adam",
					"learning_rate": 0.001,      
					"l2": 1e-6,
					# "l1": 0, 
					}

	return net, placeholders, optimization

	