
def model(input_shape, num_labels=None):
  # design a neural network model

  # create model
  layer1 = {'layer': 'input',
            'input_shape': input_shape
            }
  layer2 = {'layer': 'conv1d', 
            'num_filters': 32,
            'filter_size': 7,
            'norm': 'batch',
            'activation': 'leaky_relu',
            'dropout': 0.2
            }
  layer3 = {'layer': 'conv1d_residual',
            'filter_size': 5,
            'function': 'relu',
            'pool_size': 10,
            'dropout': 0.2
           }
  layer4 = {'layer': 'conv1d', 
            'num_filters': 64,
            'filter_size': 7,
            'norm': 'batch',
            'activation': 'leaky_relu',
            'dropout': 0.2
            }
  layer5 = {'layer': 'conv1d_residual',
            'filter_size': 5,
            'function': 'relu',
            'pool_size': 10,
            'dropout': 0.2
           }
  layer7 = {'layer': 'dense', 
            'num_units': 256,
            #'norm': 'batch',
            'activation': 'leaky_relu',
            'dropout': 0.5
            }  
  layer8 = {'layer': 'dense', 
            'num_units': num_labels,
            'activation': 'softmax',
            }

  #from tfomics import build_network
  model_layers = [layer1, layer2, layer3, layer4, layer5, layer7, layer8]

  # optimization parameters
  optimization = {"objective": "categorical",
                  "optimizer": "adam",
                  "learning_rate": 0.001,      
                  "l2": 1e-6,
                  # "l1": 0, 
                  }

  return model_layers, optimization

