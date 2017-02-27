
def model(input_shape, num_labels=None):

	# create model
	layer1 = { 'layer': 'input',
				'input_shape': input_shape
			 }
	layer2 = {  'layer': 'conv2d', 
				'num_filters': 25,
				'filter_size': (19,1),
				#'norm': 'batch',
				'padding': 'SAME',
				'activation': 'relu',
				'dropout': 0.1
				}
	layer3 = { 'layer': 'residual-conv2d',
				'filter_size': (5,1),
				'dropout': 0.1,
				'pool_size': (40,1)
			 }
	layer4 = {'layer': 'dense', 
				'num_units': 128,
				#'norm': 'batch',
				'activation': 'relu',
				'dropout': 0.5
				}
	layer5 = {'layer': 'dense', 
				'num_units': num_labels,
				'activation': 'sigmoid'
				}

	#from tfomics import build_network
	model_layers = [layer1, layer2, layer3, layer4, layer5]
	
	# optimization parameters
	optimization = {"objective": "binary",
					"optimizer": "adam",
					"learning_rate": 0.001,      
					"l2": 1e-6
					# "l1": 0, 
					}

	return model_layers, optimization

	