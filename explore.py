
class NeuralOptimizer:
	"""Class to build a neural network and perform basic functions"""

	def __init__(self, model_layers, input_var, target_var, optimization, filepath):
		self.model_layers = model_layers
		self.input_var = input_var
		self.target_var = target_var
		self.optimization = optimization
		self.filepath = filepath
		self.optimal_loss = 1e20
		self.hyperparameters = []
		self.loss = []


	def sample_network(self):
		"""generate a network, sampling from the ranges provided by
			hyperparameter search"""
		model_layers = []
		for layer in self.model_layers:

			layers = {}
			for key in layer.keys():
				if isinstance(layer[key], str):
					layers[key] = layer[key]
				else:
					if len(keys) == 2:
						bounds = layer[key]
						MIN = bounds[0]
						MAX = bounds[1]
						val = np.random.randint(MIN, MAX)
						layers[key] = val
					if len(keys) == 3:
						bounds = layer[key]
						MEAN = bounds[0]
						MIN = bounds[1]
						MAX = bounds[2]
						STD = (MAX-MIN)/6
						val = np.round(np.random.normal(MEAN, STD))
						val = np.min([val, MAX])
						val = np.max([val, MIN])
						layers[key] = val
					else:
						layers[key] = layer[key]
			model_layers.append(layers)
		self.architecture.append(model_layers)

		return model_layers


	def sample_optimization(self):
		""" generate an optimization dictionary from the ranges in 
		hyperparameter search"""

		optimization = {}
		for key in self.optimization.keys():
			if not isinstance(self.optimization[key], str):
				if len(keys) == 2:
					bounds = self.optimization[key]
					MIN = bounds[0]
					MAX = bounds[1]
					val = np.random.randint(MIN, MAX)
					optimization[key] = val
				if len(keys) == 3:
					bounds = self.optimization[key]
					MEAN = bounds[0]
					MIN = bounds[1]
					MAX = bounds[2]
					STD = (MAX-MIN)/6
					val = np.round(np.random.normal(MEAN, STD))
					val = np.min([val, MAX])
					val = np.max([val, MIN])
					optimization[key] = val
				else:
					optimization[key] = layer[key]
		return optimization


	def update_network(self, model_layers):
		"""update the means of the network hyperparameters"""

		for i in range(len(self.model_layers)):
			for key in self.model_layers[i].keys():
				if not isinstance(self.model_layers[i][key], str):
					if len(keys) == 3:
						bounds = self.model_layers[i][key]
						bounds[0] = model_layers[i][key][0]
						self.model_layers[i][key] = bounds


	def update_optimization(self, optimization):
		"""update the means of the optimization hyperparameters"""

		current_optimization = self.optimization
		for key in self.optimization.keys():
			if not isinstance(self.optimization[key], str):
				if len(keys) == 3:
					bounds = self.optimization[key]
					bounds[0] = optimization[key][0]
					self.optimization[key] = bounds


	def check_gradient_flow(self, X, batch_size=500):
		"""get the feature maps of a given convolutional layer"""

		forward_pass = []
		for layer in layers:
			fmap = nnmodel.get_feature_maps(layer, X)
			forward_pass.append(np.reshape(fmap,[-1,]))

		#backward_pass = []
		#for layer in layers:

		return forward_pass #, backward_pass


	def explore(self, batch_size, num_epochs):

		# generate new network
		model_layers = self.sample_network()
		net = build_network(model_layers)

		# generate new optimization
		optimization = self.sample_optimization()

		# build network
		nnmodel = nn.NeuralNet(net, self.input_var, self.target_var)

		# build trainer
		nntrainer = nn.NeuralTrainer(nnmodel, optimization, save='best', filepath=self.filepath)

		# train network
		fit.train_minibatch(nntrainer, data={'train': train}, batch_size=batch_size, 
							num_epochs=num_epochs, patience=[], verbose=0, shuffle=True)
		
		loss = nntrainer.train_monitor.get_loss()

		return model_layers, optimization, loss


	def optimize(self, num_trials, batch_size, num_epochs):

		# sample different hyperparameter settings
		for i in range(num_trials):
			model_layers, optimization, loss = self.explore(batch_size, num_epochs)
			if loss[-1] < self.optimal_loss:
				print "lower loss found. Updating parameters"
				self.optimal_loss = loss[-1]
				self.update_network(model_layers)
				self.update_optimization(optimization)
			self.hyperparameters.append({'model': model_layers, 
										 'optimization': optimization,
										 'loss': loss})



