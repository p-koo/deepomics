import sys
sys.path.append('..')
from tfomics import utils, init
from tfomics.build_network import *
import tensorflow as tf


def model(input_shape, num_labels=None):

  # create model
  layer1 = {'layer': 'input',
            'input_shape': input_shape,
            'name': 'input'
            }
  layer2 = {'layer': 'conv2d', 
            'num_filters': 32,
            'filter_size': (2, 5),
            'activation': 'prelu',
            'dropout': 0.2,
            'name': 'conv1'
            }
  layer3 = {'layer': 'residual-conv2d',
            'function': 'prelu',
            'filter_size': (2,5),
            'dropout_block': 0.1,
            'dropout': 0.2,
            'norm': 'batch',
            'name': 'resid1'
           }
  layer4 = {'layer': 'conv2d', 
            'num_filters': 64,
            'filter_size': (2, 5),
            'norm': 'batch',
            'activation': 'prelu',
            'dropout': 0.2,
            'name': 'conv2'
            }
  layer5 = {'layer': 'residual-conv2d',
            'function': 'prelu',
            'filter_size': (1,5),
            'dropout_block': 0.1,
            'dropout': 0.2,
            'pool_size': (1,10),
            'name': 'resid2'
           }
  layer6 = {'layer': 'conv2d', 
            'num_filters': 128,
            'filter_size': (1,1),
            'norm': 'batch',
            'activation': 'prelu',
            'dropout': 0.2,
            'name': 'conv3'
            }
  layer7 = {'layer': 'dense', 
            'num_units': 256,
            'activation': 'prelu',
            'dropout': 0.5,
            'name': 'dense1'
            }  
  layer8 = {'layer': 'residual-dense',
            'function': 'prelu',
            'dropout_block': 0.1,
            'dropout': 0.5,
            'name': 'resid3'
           }
  layer9 = {'layer': 'dense', 
            'num_units': num_labels,
            'activation': 'softmax',
            'name': 'dense2'
            }

  #from tfomics import build_network
  model_layers = [layer1, layer2, layer3, layer4, layer5, layer6, layer7, layer8, layer9]

  # optimization parameters
  optimization = {"objective": "categorical",
                  "optimizer": "adam",
                  "learning_rate": 0.001,      
                  "l2": 1e-6,
                  # "l1": 0, 
                  }

  return model_layers, optimization

