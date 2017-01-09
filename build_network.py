#/bin/python
from lasagne import layers, nonlinearities, init

def build_network(model_layers):
	""" build all layers in the model """
 
	# loop to build each layer of network
	lastlayer = ''
	for model_layer in model_layers:
		name = model_layer['name']

		if name == "input":
			# add input layer
			network[name] = single_layer(model_layer, network)
			lastlayer = name
		else:

			if name == 'residual':
				network = residual_block(network, lastlayer, model_layer['name'], 
										 model_layer['filter_size'], nonlinearity=nonlinearities.rectify)

			# add core layer
			newlayer = name #'# str(counter) + '_' + name + '_batch'
			network[newlayer] = single_layer(model_layer, network[lastlayer])
			lastlayer = newlayer

			# add bias layer
			if 'b' in model_layer:
				newlayer = name+'_bias'
				network[newlayer] = layers.BiasLayer(network[lastlayer], b=model_layer['b'])
				lastlayer = newlayer    
				
			
		# add Batch normalization layer
		if 'norm' in model_layer:
			if 'batch' in model_layer['norm']:
				newlayer = name + '_batch' #str(counter) + '_' + name + '_batch'
				network[newlayer] = layers.BatchNormLayer(network[lastlayer])
				lastlayer = newlayer
			
		# add activation layer
		if 'activation' in model_layer:
			newlayer = name+'_active'
			network[newlayer] = layers.ActivationLayer(network[lastlayer], function=model_layer['activation']) 
			lastlayer = newlayer
		
		# add Batch normalization layer
		#if 'norm' in model_layer:
		#    if 'local' in model_layer['norm']:
		#        newlayer = name + '_local' # str(counter) + '_' + name + '_local'
		#        network[newlayer] = layers.LocalResponseNormLayer(network[lastlayer], 
		#                                            alpha=.001/9.0, k=1., beta=0.75, n=5)
		#        lastlayer = newlayer
				
		
		# add dropout layer
		if 'dropout' in model_layer:
			newlayer = name+'_dropout' # str(counter) + '_' + name+'_dropout'
			network[newlayer] = layers.DropoutLayer(network[lastlayer], p=model_layer['dropout'])
			lastlayer = newlayer

		# add max-pooling layer
		if 'pool_size' in model_layer:  
			newlayer = name+'_pool'  # str(counter) + '_' + name+'_pool' 
			network[newlayer] = layers.MaxPool1DLayer(network[lastlayer], pool_size=model_layer['pool_size'])
			lastlayer = newlayer       

	network['output'] = network[lastlayer]

	return network


def single_layer(model_layer, network_last):
	""" build a single layer"""

	# input layer
	if model_layer['layer'] == 'input':
		network = layers.InputLayer(model_layer['shape'], input_var=model_layer['input_var'])

	# dense layer
	elif model_layer['layer'] == 'dense':
		network = layers.DenseLayer(network_last, num_units=model_layer['num_units'],
											 W=model_layer['W'],
											 b=None, 
											 nonlinearity=None)

	# convolution layer
	elif model_layer['layer'] == 'convolution':
		network = layers.Conv1DLayer(network_last, num_filters=model_layer['num_filters'],
											  filter_size=model_layer['filter_size'],
											  W=model_layer['W'],
											  b=None, 
											  pad=model_layer['pad'],
											  nonlinearity=None)

	#elif model_layer['layer'] == 'lstm':
	#    l_forward = layers.LSTMLayer(network_last, num_units=model_layer['num_units'], 
	#                                        grad_clipping=model_layer['grad_clipping'])
	#    l_backward = layers.LSTMLayer(network_last, num_units=model_layer['num_units'], 
	#                                        grad_clipping=model_layer['grad_clipping'], 
	#                                        backwards=True)
	#    network = layers.ConcatLayer([l_forward, l_backward])


	#elif model_layer['layer'] == 'highway':
	#    network = layers.DenseLayer(network_last, num_units=model_layer['num_units'],
	#                                         W=model_layer['W'],
	#                                         b=None, 
	#                                         nonlinearity=None)
	#    for k in range(model_layer['num_layers']):
	#        network = highway_dense(network)

	return network


"""
#--------------------------------------------------------------------------------------------------------------------
# highway MLP layer

class MultiplicativeGatingLayer(layers.MergeLayer):
	def __init__(self, gate, input1, input2, **kwargs):
		incomings = [gate, input1, input2]
		super(MultiplicativeGatingLayer, self).__init__(incomings, **kwargs)
	
	def get_output_shape_for(self, input_shapes):
		return input_shapes[0]
	
	def get_output_for(self, inputs, **kwargs):
		return inputs[0] * inputs[1] + (1 - inputs[0]) * inputs[2]



def highway_dense(incoming, W_dense=init.Orthogonal(), b_dense=init.Constant(0.0),
				  W_gate=init.Orthogonal(), b_gate=init.Constant(-4.0),
				  nonlinearity=nonlinearities.rectify, **kwargs):

	num_inputs = int(np.prod(incoming.output_shape[1:]))

	# regular layer
	l_dense = layers.DenseLayer(incoming, num_units=num_inputs, W=W_dense, b=b_dense, nonlinearity=nonlinearity)

	# gate layer
	l_gate = layers.DenseLayer(incoming, num_units=num_inputs, W=W_gate, b=b_gate, nonlinearity=nonlinearities.sigmoid)
	
	return MultiplicativeGatingLayer(gate=l_gate, input1=l_dense, input2=incoming)


#--------------------------------------------------------------------------------------------------------------------
# residual learning layer

def residual_block(net, last_layer, name, filter_size, nonlinearity=nonlinearities.rectify):

	
	# original residual unit
	shape = layers.get_output_shape(net[last_layer])
	num_filters = shape[1]

	net[name+'_1resid'] = layers.Conv2DLayer(net[last_layer], num_filters=num_filters, filter_size=filter_size, stride=(1, 1),    # 1000
					 W=init.HeNormal(), b=None, nonlinearity=None, pad='same')
	net[name+'_1resid_norm'] = layers.BatchNormLayer(net[name+'_1resid'])
	net[name+'_1resid_active'] = layers.NonlinearityLayer(net[name+'_1resid_norm'], nonlinearity=nonlinearity)

	# bottleneck residual layer
	net[name+'_2resid'] = layers.Conv2DLayer(net[name+'_1resid_active'], num_filters=num_filters, filter_size=filter_size, stride=(1, 1),    # 1000
					 W=init.HeNormal(), b=None, nonlinearity=None, pad='same')
	net[name+'_2resid_norm'] = layers.BatchNormLayer(net[name+'_2resid'])

	# combine input with residuals
	net[name+'_residual'] = layers.ElemwiseSumLayer([net[last_layer], net[name+'_2resid_norm']])
	net[name+'_resid'] = layers.NonlinearityLayer(net[name+'_residual'], nonlinearity=nonlinearity)
	
	# new residual unit
	shape = layers.get_output_shape(net[last_layer])
	num_filters = shape[1]

	net[name+'_1resid_norm'] = layers.BatchNormLayer(net[last_layer])
	net[name+'_1resid_active'] = layers.NonlinearityLayer(net[name+'_1resid_norm'], nonlinearity=nonlinearity)
	net[name+'_1resid'] = layers.Conv2DLayer(net[name+'_1resid_active'], num_filters=num_filters, filter_size=filter_size, stride=(1, 1),    # 1000
					 W=init.HeNormal(), b=None, nonlinearity=None, pad='same')

	# bottleneck residual layer
	net[name+'_2resid_norm'] = layers.BatchNormLayer(net[name+'_1resid'])
	net[name+'_2resid_active'] = layers.NonlinearityLayer(net[name+'_2resid_norm'], nonlinearity=nonlinearity)
	net[name+'_2resid'] = layers.Conv2DLayer(net[name+'_2resid_active'], num_filters=num_filters, filter_size=filter_size, stride=(1, 1),    # 1000
					 W=init.HeNormal(), b=None, nonlinearity=None, pad='same')

	# combine input with residuals
	net[name+'_resid'] = layers.ElemwiseSumLayer([net[last_layer], net[name+'_2resid']])
	return net


def residual_bottleneck(net, last_layer, name, num_filters, filter_size, nonlinearity=nonlinearities.rectify):

	# initial residual unit
	shape = layers.get_output_shape(net[last_layer])
	num_filters = shape[1]
	reduced_filters = np.round(num_filters/4)

	# 1st residual layer
	net[name] = layers.Conv2DLayer(net[last_layer], num_filters=reduced_filters, filter_size=(1,1), stride=(1, 1),  
					 W=init.HeNormal(), b=None, nonlinearity=None, pad='same')
	net[name+'_norm'] = layers.BatchNormLayer(net[name])
	net[name+'_active'] = layers.NonlinearityLayer(net[name+'_norm'], nonlinearity=nonlinearities.rectify)

	net[name+'_resid'] = layers.Conv2DLayer(net[name+'_active'], num_filters=reduced_filters, filter_size=filter_size, stride=(1, 1),   
					 W=init.HeNormal(), b=None, nonlinearity=None, pad='same')
	net[name+'_resid_norm'] = layers.BatchNormLayer(net[name+'_resid'])
	net[name+'_resid_active'] = layers.NonlinearityLayer(net[name+'_resid_norm'], nonlinearity=nonlinearity)

	# bottleneck residual layer
	net[name+'_bottle'] = layers.Conv2DLayer(net[name+'_resid_active'], num_filters=num_filters, filter_size=(1,1), stride=(1, 1),    
					 W=init.HeNormal(), b=None, nonlinearity=None, pad='same')
	net[name+'_bottle_norm'] = layers.BatchNormLayer(net[name+'_bottle'])

	# combine input with residuals
	net[name+'_residual'] = layers.ElemwiseSumLayer([net[last_layer], net[name+'_bottle_norm']])
	net[name+'_residual_active'] = layers.NonlinearityLayer(net[name+'_residual'], nonlinearity=nonlinearity)

	return net

#--------------------------------------------------------------------------------------------------------------------
# Denoising layer for ladder network

class DenoiseLayer(layers.MergeLayer):
	
	def __init__(self, u_net, z_net,
				 nonlinearity=nonlinearities.sigmoid, **kwargs):
		super(DenoiseLayer, self).__init__([u_net, z_net], **kwargs)

		u_shp, z_shp = self.input_shapes


		if not u_shp[-1] == z_shp[-1]:
			raise ValueError("last dimension of u and z  must be equal"
							 " u was %s, z was %s" % (str(u_shp), str(z_shp)))
		self.num_inputs = z_shp[-1]
		self.nonlinearity = nonlinearity
		constant = init.Constant
		self.a1 = self.add_param(constant(0.), (self.num_inputs,), name="a1")
		self.a2 = self.add_param(constant(1.), (self.num_inputs,), name="a2")
		self.a3 = self.add_param(constant(0.), (self.num_inputs,), name="a3")
		self.a4 = self.add_param(constant(0.), (self.num_inputs,), name="a4")

		self.c1 = self.add_param(constant(0.), (self.num_inputs,), name="c1")
		self.c2 = self.add_param(constant(1.), (self.num_inputs,), name="c2")
		self.c3 = self.add_param(constant(0.), (self.num_inputs,), name="c3")

		self.c4 = self.add_param(constant(0.), (self.num_inputs,), name="c4")

		self.b1 = self.add_param(constant(0.), (self.num_inputs,),
								 name="b1", regularizable=False)

	def get_output_shape_for(self, input_shapes):
		output_shape = list(input_shapes[0])  # make a mutable copy
		return tuple(output_shape)

	def get_output_for(self, inputs, **kwargs):
		u, z_lat = inputs
		sigval = self.c1 + self.c2*z_lat
		sigval += self.c3*u + self.c4*z_lat*u
		sigval = self.nonlinearity(sigval)
		z_est = self.a1 + self.a2 * z_lat + self.b1*sigval
		z_est += self.a3*u + self.a4*z_lat*u
		return z_est

def variable_type(var_type):
	if var_type == 'constant':
		# kwargs include: value=0.05, dtype=tf.float32
		var = init.Constant(**kwargs)

	elif var_type == 'uniform':
		# kwargs include: minval=-0.1, maxval=0.1, dtype=tf.float32
		var = init.Uniform(**kwargs)

	elif var_type == 'normal':
		# kwargs include: mean=0.0, stddev=0.1, dtype=tf.float32
		var = init.Normal(**kwargs)

	elif var_type == 'truncated_normal':
		# kwargs include: mean=0.0, stddev=0.1, dtype=tf.float32
		var = init.TruncatedNormal(**kwargs)

	elif var_type == 'glorot-uniform':
		# kwargs include: dtype=tf.float32
		var = init.GlorotUniform(**kwargs)

	elif var_type == 'glorot-normal':
		# kwargs include: mean=0.0, dtype=tf.float32
		var = init.GlorotNormal(**kwargs)

	elif var_type == 'he-uniform':
		# kwargs include: dtype=tf.float32
		var = init.HeUniform(**kwargs)

	elif var_type == 'he-normal':
		# kwargs include: mean=0.0, dtype=tf.float32
		var = init.HeNormal(**kwargs)

	elif var_type == 'orthogonal':
		# kwargs include: gain=1.1, dtype=tf.float32
		var = init.Orthogonal(**kwargs)

	return var
"""