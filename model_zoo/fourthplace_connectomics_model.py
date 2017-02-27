
def model(input_shape, num_labels=None):
  # design a neural network model
  
  # create model
  layer1 = {'layer': 'input',
            'input_shape': input_shape,
            'name': 'input'
            }
  layer2 = {'layer': 'conv2d', 
            'num_filters': 18,
            'filter_size': (2, 5),
            'activation': 'tanh'
            }
  layer3 = {'layer': 'conv2d', 
            'num_filters': 40,
            'filter_size': (2, 5),
            'activation': 'tanh',
            'pool_size': (1,10)
            }
  layer4 = {'layer': 'conv2d', 
            'num_filters': 15,
            'filter_size': (1,1),
            'activation': 'tanh'
            }
  layer5 = {'layer': 'dense', 
            'num_units': 100,
            'activation': 'relu'
            }
  layer6 = {'layer': 'dense', 
            'num_units': num_labels,
            'activation': 'softmax'
            }

  #from tfomics import build_network
  model_layers = [layer1, layer2, layer3, layer4, layer5, layer6]

  # optimization parameters
  optimization = {"objective": "categorical",
                  "optimizer": "adam",
                  "learning_rate": 0.001,      
                  "l2": 1e-6,
                  # "l1": 0, 
                  }

  return model_layers, optimization

