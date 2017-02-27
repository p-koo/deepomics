
def model(input_shape, num_labels):


    # create model
    layer1 = {  'layer': 'input',
                'shape': input_shape,
                }
    layer2 = {  'layer': 'conv1d', 
                'num_filters': {'start': 32, 'bounds': [1, 200], 'scale': 25, 'multiples': 4},
                'filter_size': {'start': 7, 'bounds': [5, 32], 'scale': 5, 'odd': True, 'multiples': 2},
                'norm': 'batch',
                'padding': 'same',
                'activation': 'relu',
                'dropout': {'start': 0.2, 'bounds': [0., 0.6], 'scale': 0.1},
                'name': 'conv1'
                }
    layer3 = {  'layer': 'conv1d_residual', 
                'filter_size': {'start': 5, 'bounds': [3, 9], 'scale': 4, 'odd': True},
                'activation': 'relu',
                'pool_size': {'start': 10, 'bounds': [1, 200], 'scale': 30, 'multiples': 2},
                'dropout': {'start': .2, 'bounds': [0., .6], 'scale': .1},
                'name': 'conv1_resid'
              }
    layer4 = {  'layer': 'conv1d', 
                'num_filters': {'start': 64, 'bounds': [1, 200], 'scale': 25, 'multiples': 4},
                'filter_size': {'start': 7, 'bounds': [5, 32], 'scale': 5, 'odd': True, 'multiples': 2},
                'norm': 'batch',
                'padding': 'same',
                'activation': 'relu',
                'dropout': {'start': 0.2, 'bounds': [0., 0.6], 'scale': 0.1},
                'name': 'conv2'
                }
    layer5 = {  'layer': 'conv1d_residual', 
                'filter_size': {'start': 5, 'bounds': [3, 9], 'scale': 4, 'odd': True},
                'activation': 'relu',
                'pool_size': {'start': 10, 'bounds': [1, 200], 'scale': 30, 'multiples': 2},
                'dropout': {'start': .2, 'bounds': [0., .6], 'scale': .1},
                'name': 'conv2_resid'
              }              
    layer6 = {  'layer': 'dense', 
                'num_units': {'start': 256, 'bounds': [16, 1000], 'scale': 50, 'multiples': 4},
                #'norm': 'batch',
                'activation': 'relu',
                'dropout': {'start': 0.4, 'bounds': [0., 0.8], 'scale': 0.2},
                'name': 'dense1'
                }            
    layer7 = {  'layer': 'dense', 
                'num_units': num_labels,
                'activation': 'softmax',
                }

    #from tfomics import build_network
    model_layers = [layer1, layer2, layer3, layer4, layer5, layer6, layer7]

    # optimization parameters
    optimization = {"objective": "categorical",
                    "optimizer": "adam",
                    "learning_rate": 0.001,
                    "l2": 1e-6
                    #"learning_rate": {'start': -3, 'bounds': [-4, -1], 'scale': 1.5, 'transform': 'log'},      
                    #"l2": {'start': -6, 'bounds': [-8, -2], 'scale': 3, 'transform': 'log'},
                    # "l1": 0, 
                    }
    return model_layers, optimization