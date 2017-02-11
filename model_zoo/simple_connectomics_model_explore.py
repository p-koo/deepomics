import sys
sys.path.append('..')
from tfomics import utils, init
import tensorflow as tf



def model(input_shape, num_labels):

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
                'num_filters': {'start': 20, 'bounds': [1, 200], 'scale': 20},
                'filter_size': {'start': 5, 'bounds': [3, 19], 'scale': 6, 'multiples': 2, 'offset': 0},
                'batch_norm': is_training,
                'padding': 'SAME',
                'activation': 'relu',
                'pool_size': {'start': 4, 'bounds': [1, 10], 'scale': 6, 'multiples': 4},
                'name': 'conv1'
                }
    layer3 = {'layer': 'conv1d', 
                'num_filters': {'start': 20, 'bounds': [1, 200], 'scale': 20},
                'filter_size': {'start': 5, 'bounds': [3, 19], 'scale': 6, 'multiples': 2, 'offset': 0},
                'batch_norm': is_training,
                'padding': 'SAME',
                'activation': 'relu',
                'pool_size': {'start': 4, 'bounds': [1, 10], 'scale': 6, 'multiples': 4},
                'dropout': keep_prob,
                'name': 'conv2'
                }
    layer4 = {'layer': 'dense', 
                'num_units': {'start': 120, 'bounds': [20, 1000], 'scale': 100, 'multiples': 10},
                'activation': 'sigmoid',
                'dropout': keep_prob,
                'name': 'dense1'
                }
    layer5 = {'layer': 'dense', 
                'num_units': num_labels,
                'activation': 'sigmoid',
                'name': 'dense2'
                }

    #from tfomics import build_network
    model_layers = [layer1, layer2, layer3, layer4, layer5]

    # optimization parameters
    optimization = {"objective": "binary",
                    "optimizer": "adam",
                    "learning_rate": {'start': -3, 'bounds': [-4, -1], 'scale': 1.5, 'transform': 'log'},      
                    "l2": {'start': -6, 'bounds': [-8, -2], 'scale': 3, 'transform': 'log'},
                    # "l1": 0, 
                    }
    return model_layers, placeholders, optimization