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
  layer2 = {'layer': 'conv2d', 
            'num_filters': 32,
            'filter_size': (2, 5),
            'batch_norm': is_training,
            'activation': 'leaky_relu',
            'dropout': keep_prob,
            'name': 'conv1'
            }
  layer3 = {'layer': 'residual-conv2d',
            'filter_size': (2,5),
            'batch_norm': is_training,
            'name': 'resid1'
           }
  layer4 = {'layer': 'conv2d', 
            'num_filters': 64,
            'filter_size': (2, 5),
            'batch_norm': is_training,
            'activation': 'leaky_relu',
            'dropout': keep_prob,
            'name': 'conv2'
            }
  layer5 = {'layer': 'residual-conv2d',
            'filter_size': (1,5),
            'batch_norm': is_training,
            'pool_size': (1,10),
            'name': 'resid2'
           }
  layer6 = {'layer': 'conv2d', 
            'num_filters': 128,
            'filter_size': (1,1),
            'batch_norm': is_training,
            'activation': 'leaky_relu',
            'dropout': keep_prob,
            'name': 'conv3'
            }
  layer7 = {'layer': 'dense', 
            'num_units': 256,
            'batch_norm': is_training,
            'activation': 'leaky_relu',
            'dropout': keep_prob,
            'name': 'dense1'
            }  
  layer8 = {'layer': 'residual-dense',
            'batch_norm': is_training,
            'dropout': keep_prob,
            'name': 'resid3'
           }
  layer9 = {'layer': 'dense', 
            'num_units': num_labels,
            'activation': 'softmax',
            'name': 'dense2'
            }

  #from tfomics import build_network
  model_layers = [layer1, layer2, layer3, layer4, layer5, layer6, layer7, layer8, layer9]
  net = build_network(model_layers)

  # optimization parameters
  optimization = {"objective": "categorical",
                  "optimizer": "adam",
                  "learning_rate": 0.001,
                  "l2": 1e-6,
                  # "l1": 0, 
                  }

  return net, placeholders, optimization

