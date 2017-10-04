import tensorflow as tf
import numpy as np

__all__ = [
	"build_updates",
	"build_loss",
	"cost_function"
]


def build_updates(optimization):
	"""Build updates"""

	if 'optimizer' in optimization.keys():
		optimizer = optimization['optimizer']
	else:
		optimizer = 'adam'
		optimization['learning_rate'] = 0.001

	if optimizer == 'sgd':
		learning_rate = 0.005
		if 'learning_rate' in optimization.keys():
			leanring_rate = optimization['learning_rate']
		use_locking = False
		if 'use_locking' in optimization.keys():
			use_locking = optimization['use_locking']
		name = 'adadelta'
		if 'name' in optimization.keys():
			name = optimization['name']
		return tf.train.GradientDescentOptimizer(learning_rate=learning_rate,
												 use_locking=use_locking,
												 name=name)

	elif optimizer == 'momentum':
		learning_rate = 0.005
		if 'learning_rate' in optimization.keys():
			leanring_rate = optimization['learning_rate']
		momentum = 0.9
		if 'momentum' in optimization.keys():
			momentum = optimization['momentum']
		use_nesterov = True
		if 'use_nesterov' in optimization.keys():
			use_nesterov = optimization['use_nesterov']
		use_locking = False
		if 'use_locking' in optimization.keys():
			use_locking = optimization['use_locking']
		name = 'momenum'
		if 'name' in optimization.keys():
			name = optimization['name']
		return tf.train.MomentumOptimizer(learning_rate=learning_rate,
										  momentum=momentum,
										  use_nesterov=use_nesterov,
										  use_locking=use_locking,
										  name=name)

	elif optimizer == 'adam':
		learning_rate = 0.001
		if 'learning_rate' in optimization.keys():
			leanring_rate = optimization['learning_rate']
		beta1 = 0.95
		if 'beta1' in optimization.keys():
			beta1 = optimization['beta1']
		beta2 = 0.999
		if 'beta2' in optimization.keys():
			beta2 = optimization['beta2']
		epsilon = 1e-08
		if 'epsilon' in optimization.keys():
			epsilon = optimization['epsilon']
		use_locking = False
		if 'use_locking' in optimization.keys():
			use_locking = optimization['use_locking']
		name = 'adam'
		if 'name' in optimization.keys():
			name = optimization['name']
		return tf.train.AdamOptimizer(learning_rate=learning_rate,
									  beta1=beta1,
									  beta2=beta2,
									  epsilon=epsilon,
									  use_locking=use_locking,
									  name=name)

	elif optimizer == 'rmsprop':
		learning_rate = 0.001
		if 'learning_rate' in optimization.keys():
			leanring_rate = optimization['learning_rate']
		rho = 0.95
		if 'rho' in optimization.keys():
			rho = optimization['rho']
		epsilon = 1e-08
		if 'epsilon' in optimization.keys():
			epsilon = optimization['epsilon']
		use_locking = False
		if 'use_locking' in optimization.keys():
			use_locking = optimization['use_locking']
		name = 'rmsprop'
		if 'name' in optimization.keys():
			name = optimization['name']
		return tf.train.RMSPropOptimizer(learning_rate=learning_rate,
										 rho=rho,
										 epsilon=epsilon,
										 use_locking=use_locking,
										 name=name)

	elif optimizer == 'adadelta':
		learning_rate = 0.001
		if 'learning_rate' in optimization.keys():
			leanring_rate = optimization['learning_rate']
		rho = 0.95
		if 'rho' in optimization.keys():
			rho = optimization['rho']
		epsilon = 1e-08
		if 'epsilon' in optimization.keys():
			epsilon = optimization['epsilon']
		use_locking = False
		if 'use_locking' in optimization.keys():
			use_locking = optimization['use_locking']
		name = 'adadelta'
		if 'name' in optimization.keys():
			name = optimization['name']
		return tf.train.AdadeltaOptimizer(learning_rate=learning_rate,
										  rho=rho,
										  epsilon=epsilon,
										  use_locking=use_locking,
										  name=name)

	elif optimizer == 'adagrad':
		learning_rate = 0.001
		if 'learning_rate' in optimization.keys():
			leanring_rate = optimization['learning_rate']
		initial_accumulator_value = 0.95
		if 'initial_accumulator_value' in optimization.keys():
			initial_accumulator_value = optimization['initial_accumulator_value']
		use_locking = False
		if 'use_locking' in optimization.keys():
			use_locking = optimization['use_locking']
		name = 'adagrad'
		if 'name' in optimization.keys():
			name = optimization['name']
		return tf.train.AdagradOptimizer(learning_rate=learning_rate,
										 initial_accumulator_value=initial_accumulator_value,
										 use_locking=use_locking,
										 name=name)



def build_loss(network, predictions, targets, optimization):

	# build loss function
	if 'label_smoothing' not in optimization.keys():
		optimization['label_smoothing'] = 0
	loss = cost_function(network, targets, optimization)

	if 'l1' in optimization.keys():
		l1 = get_l1_parameters(network)
		loss += tf.reduce_sum(tf.abs(l1)) * optimization['l1']

	if 'l2' in optimization.keys():
		l2 = get_l2_parameters(network)
		loss += tf.reduce_sum(tf.square(l2)) * optimization['l2']

	return loss


def cost_function(network, targets, optimization):

	objective = optimization['objective']
	label_smoothing = optimization['label_smoothing']

	if objective == 'binary':
		predictions = network['output'].get_output()
		if label_smoothing > 0:
			  targets = (targets*(1-label_smoothing) + 0.5*label_smoothing)
		predictions = tf.clip_by_value(predictions,1e-7,1-1e-7)
		loss = -tf.reduce_mean(targets*tf.log(predictions) + (1-targets)*tf.log(1-predictions))

	elif objective == 'categorical':
		predictions = network['output'].get_output()
		if label_smoothing > 0:
			num_classes = targets.get_shape()[-1].value
			smooth_positives = 1.0 - label_smoothing
			smooth_negatives = label_smoothing/num_classes
			targets = targets*smooth_positives + smooth_negatives
		predictions = tf.clip_by_value(predictions,1e-7,1-1e-7)
		#loss = -tf.reduce_mean(tf.reduce_sum(targets*tf.log(predictions), axis=1))
		loss = -tf.reduce_mean(targets*tf.log(predictions))

	elif objective == 'squared_error':
		predictions = network['output'].get_output()
		loss = tf.reduce_mean(tf.square(targets - predictions))

	elif objective == 'cdf':
		predictions = network['output'].get_output()
		loss = tf.reduce_mean(tf.square(targets - predictions))

	elif objective == 'kl_divergence':
		predictions = network['output'].get_output()
		loss = tf.reduce_mean(tf.multiply(predictions, tf.log(tf.divide(predictions, targets+1e-7))))

	elif objective == 'elbo_gaussian':
		X_mu = network['X'].get_output()
		if 'X_sigma' in network.keys():
			X_sigma = network['X_sigma'].get_output()
			X_log_var = tf.log(tf.square(X_sigma))
		else:
			X_logvar = tf.log(X_mu*(1-X_mu))

		Z_mu = network['Z_mu'].get_output()
		Z_logsigma = network['Z_logsigma'].get_output()
		Z_sigma = tf.exp(Z_logsigma)+1e-10

		half = tf.constant(0.5, dtype=tf.float32)
		const = tf.constant(-0.5*np.log(2*np.float32(np.pi)), dtype=tf.float32)

		# calculate kl divergence
		kl_divergence = 0.5*tf.reduce_sum(1 + 2*tf.log(Z_sigma) - tf.square(Z_mu) - tf.exp(2*tf.log(Z_sigma)), axis=1)

		# calculate reconstructed likelihood
		log_likelihood = tf.reduce_sum(tf.square(targets-X_mu), axis=1) #tf.reduce_sum(const - 0.5*X_logvar - 0.5*tf.divide(tf.square(targets-X_mu),tf.exp(X_logvar)), axis=1)
	
		# variational lower bound (evidence lower bound)
		if 'KL_weight' in optimization:
			KL_weight = optimization['KL_weight']
		else:
			KL_weight = 1.0 
		loss = tf.reduce_mean(log_likelihood - KL_weight*kl_divergence)


	elif objective == 'elbo_binary':
		X_mu = network['X'].get_output()
		Z_mu = network['Z_mu'].get_output()
		Z_logsigma = network['Z_logsigma'].get_output()
		Z_sigma = tf.exp(Z_logsigma)+1e-10
		half = tf.constant(0.5, dtype=tf.float32)
		const = tf.constant(-0.5*np.log(2*np.float32(np.pi)), dtype=tf.float32)

		# calculate kl divergence
		kl_divergence = 0.5*tf.reduce_sum(1 + 2*tf.log(Z_sigma) - tf.square(Z_mu) - tf.exp(2*tf.log(Z_sigma)), axis=1)

		# calculate reconstructed likelihood
		X_mu = tf.clip_by_value(X_mu, 1e-10, 1-1e-10)
		log_likelihood = tf.reduce_sum(targets*tf.log(X_mu) + (1.0-targets)*tf.log(1.0-X_mu), axis=1)

		# variational lower bound (evidence lower bound)
		if 'KL_weight' in optimization:
			KL_weight = optimization['KL_weight']
		else:
			KL_weight = 1.0 
		loss = tf.reduce_mean(-log_likelihood -  KL_weight*kl_divergence)


	elif objective == 'elbo_softmax_normal':
		X = network['X'].get_output()
		Z = network['Z'].get_output()
		log_Z = tf.log(Z+1e-20)
		shape = optimization['shape']
		num_categories = shape[0]
		num_classes = shape[1]

		if 'KL_weight' in optimization:
			KL_weight = optimization['KL_weight']
		else:
			KL_weight = 1.0 

		kl_tmp = tf.reshape(Z*(log_Z-tf.log(1.0/num_classes)),[-1,num_categories,num_classes])
		kl_divergence = tf.reduce_sum(kl_tmp, [1,2])

		# calculate reconstructed likelihood
		log_likelihood = tf.reduce_sum(tf.square(targets-X), axis=1) #tf.reduce_sum(const - 0.5*X_logvar - 0.5*tf.divide(tf.square(targets-X_mu),tf.exp(X_logvar)), axis=1)
		loss = tf.reduce_mean(log_likelihood -  KL_weight*kl_divergence)



	elif objective == 'elbo_softmax_binary':
		X = network['X'].get_output()
		Z = network['Z'].get_output()
		log_Z = tf.log(Z+1e-20)
		shape = optimization['shape']
		num_categories = shape[0]
		num_classes = shape[1]

		if 'KL_weight' in optimization:
			KL_weight = optimization['KL_weight']
		else:
			KL_weight = 1.0 

		kl_tmp = tf.reshape(Z*(log_Z-tf.log(1.0/num_classes)),[-1,num_categories,num_classes])
		kl_divergence = tf.reduce_sum(kl_tmp, [1,2])

		# calculate reconstructed likelihood
		X = tf.clip_by_value(X, 1e-10, 1-1e-10)
		log_likelihood = tf.reduce_sum(targets*tf.log(X) + (1.0-targets)*tf.log(1-X), axis=1)
		loss = tf.reduce_mean(-log_likelihood -  KL_weight*kl_divergence)

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
		all_params = tf.concat([all_params, tf.reshape(param, [-1,])], axis=0)
	return all_params
