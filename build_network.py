
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
