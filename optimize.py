import tensorflow as tf


__all__ = [
	"build_updates",
	"build_loss",
	"cost_function"
]


def build_updates(optimizer, **kwargs):
	if optimizer == 'sgd':
		learning_rate = 0.005
		if 'learning_rate' in kwargs.keys():
			leanring_rate = kwargs['learning_rate']
		use_locking = False
		if 'use_locking' in kwargs.keys():
			use_locking = kwargs['use_locking']
		name = 'adadelta'
		if 'name' in kwargs.keys():
			name = kwargs['name']
		return tf.train.GradientDescentOptimizer(learning_rate=learning_rate, 
												 use_locking=use_locking, 
												 name=name)

	elif optimizer == 'momentum':
		learning_rate = 0.005
		if 'learning_rate' in kwargs.keys():
			leanring_rate = kwargs['learning_rate']
		momentum = 0.9
		if 'momentum' in kwargs.keys():
			momentum = kwargs['momentum']
		use_nesterov = True
		if 'use_nesterov' in kwargs.keys():
			use_nesterov = kwargs['use_nesterov']
		use_locking = False
		if 'use_locking' in kwargs.keys():
			use_locking = kwargs['use_locking']
		name = 'momenum'
		if 'name' in kwargs.keys():
			name = kwargs['name']
		return tf.train.MomentumOptimizer(learning_rate=learning_rate, 
										  momentum=momentum, 
										  use_nesterov=use_nesterov, 
										  use_locking=use_locking, 
										  name=name)
	
	elif optimizer == 'adam':
		learning_rate = 0.001
		if 'learning_rate' in kwargs.keys():
			leanring_rate = kwargs['learning_rate']
		beta1 = 0.95
		if 'beta1' in kwargs.keys():
			beta1 = kwargs['beta1']
		beta2 = 0.999
		if 'beta2' in kwargs.keys():
			beta2 = kwargs['beta2']
		epsilon = 1e-08
		if 'epsilon' in kwargs.keys():
			epsilon = kwargs['epsilon']
		use_locking = False
		if 'use_locking' in kwargs.keys():
			use_locking = kwargs['use_locking']
		name = 'adam'
		if 'name' in kwargs.keys():
			name = kwargs['name']
		return tf.train.AdamOptimizer(learning_rate=learning_rate, 
									  beta1=beta1, 
									  beta2=beta2, 
									  epsilon=epsilon, 
									  use_locking=use_locking, 
									  name=name)

	elif optimizer == 'rmsprop':
		learning_rate = 0.001
		if 'learning_rate' in kwargs.keys():
			leanring_rate = kwargs['learning_rate']
		rho = 0.95
		if 'rho' in kwargs.keys():
			rho = kwargs['rho']
		epsilon = 1e-08
		if 'epsilon' in kwargs.keys():
			epsilon = kwargs['epsilon']
		use_locking = False
		if 'use_locking' in kwargs.keys():
			use_locking = kwargs['use_locking']
		name = 'rmsprop'
		if 'name' in kwargs.keys():
			name = kwargs['name']
		return tf.train.RMSPropOptimizer(learning_rate=learning_rate, 
										 rho=rho, 
										 epsilon=epsilon, 
										 use_locking=use_locking, 
										 name=name)
	
	elif optimizer == 'adadelta':
		learning_rate = 0.001
		if 'learning_rate' in kwargs.keys():
			leanring_rate = kwargs['learning_rate']
		rho = 0.95
		if 'rho' in kwargs.keys():
			rho = kwargs['rho']
		epsilon = 1e-08
		if 'epsilon' in kwargs.keys():
			epsilon = kwargs['epsilon']
		use_locking = False
		if 'use_locking' in kwargs.keys():
			use_locking = kwargs['use_locking']
		name = 'adadelta'
		if 'name' in kwargs.keys():
			name = kwargs['name']
		return tf.train.AdadeltaOptimizer(learning_rate=learning_rate, 
										  rho=rho, 
										  epsilon=epsilon, 
										  use_locking=use_locking, 
										  name=name)

	elif optimizer == 'adagrad':
		learning_rate = 0.001
		if 'learning_rate' in kwargs.keys():
			leanring_rate = kwargs['learning_rate']
		initial_accumulator_value = 0.95
		if 'initial_accumulator_value' in kwargs.keys():
			initial_accumulator_value = kwargs['initial_accumulator_value']
		use_locking = False
		if 'use_locking' in kwargs.keys():
			use_locking = kwargs['use_locking']
		name = 'adagrad'
		if 'name' in kwargs.keys():
			name = kwargs['name']
		return tf.train.AdagradOptimizer(learning_rate=learning_rate, 
										 initial_accumulator_value=initial_accumulator_value, 
										 use_locking=use_locking, 
										 name=name)



def build_loss(network, predictions, targets, optimization):
	
	# cost function
	if (optimization['objective'] == 'binary') | (optimization['objective'] == 'categorical'):
		clip_value = True
	else:
		if 'clip_value' in optimization.keys():
			clip_value = optimization['clip_value']
		else:
			clip_value = False

	# build loss function
	loss = cost_function(predictions=predictions, targets=targets, 
						 objective=optimization['objective'], 
						 clip_value=clip_value)

	if 'l1' in optimization.keys():
		l1 = get_l1_parameters(network)
		loss += tf.reduce_sum(tf.abs(l1)) * optimization['l1']

	if 'l2' in optimization.keys():
		l2 = get_l1_parameters(network)
		loss += tf.reduce_sum(tf.square(l2)) * optimization['l2']

	return loss



def cost_function(predictions, targets, objective, **kwargs):

	if 'clip_value' in kwargs.keys():
		if kwargs['clip_value']:
			predictions = tf.clip_by_value(predictions,1e-7,1-1e-7)
		
	if objective == 'binary':   
		loss = -tf.reduce_mean(targets*tf.log(predictions) + (1-targets)*tf.log(1-predictions))
		# loss = tf.nn.sigmoid_cross_entropy_with_logits(predictions, targets, name='binary_loss')

	elif objective == 'categorical':
		predictions = tf.clip_by_value(predictions,1e-7,1-1e-7)
		#loss = -tf.reduce_sum(targets*tf.log(predictions))
		loss = tf.nn.softmax_cross_entropy_with_logits(predictions, targets, name='softmax_loss')
	
	elif objective == 'squared_error':    
		loss = tf.reduce_mean(tf.square(predictions - targets))
		#loss = tf.nn.l2_loss(predictions-targets, name='squared_error')

	elif objective == 'vae':
		loss = []
		
	return loss


def get_l1_parameters(net):    
	params = []
	for layer in net:
		if hasattr(net[layer], 'is_l1_regularize'):
			if net[layer].is_l1_regularize():
				variables = net[layer].get_variable()
				if isinstance(variables, list):
					for var in variables:
						params.append(var.get_variable())
				else:
					params.append(variables.get_variable())
	return merge_parameters(params)


def get_l2_parameters(net):    
	params = []
	for layer in net:
		if hasattr(net[layer], 'is_l2_regularize'):
			if net[layer].is_l2_regularize():
				variables = net[layer].get_variable()
				if isinstance(variables, list):
					for var in variables:
						params.append(var.get_variable())
				else:
					params.append(variables.get_variable())
	return merge_parameters(params)



def merge_parameters(params):
	all_params = []
	for param in params:
		all_params = tf.concat(0, [all_params, tf.reshape(param, [-1,])])
	return all_params
	
