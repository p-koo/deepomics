import os, sys



__all__ = [
	"train_minibatch",
	"train_decay_learning_rate",
	"train_anneal_batch_size",
	"train_anneal_learning_rate"
]



def train_minibatch(sess, nntrainer, data, batch_size=128, num_epochs=500, 
					patience=10, verbose=1, shuffle=True, save_all=False):
	"""Train a model with cross-validation data and test data"""

	for epoch in range(num_epochs):
		if verbose >= 1:
			sys.stdout.write("\rEpoch %d out of %d \n"%(epoch+1, num_epochs))
		else:
			if epoch % 10 == 0:
				sys.stdout.write("\rEpoch %d out of %d \n"%(epoch+1, num_epochs))

		# training set
		train_loss = nntrainer.train_epoch(sess, data['train'], 
											batch_size=batch_size, 
											verbose=verbose, 
											shuffle=shuffle)

		# test current model with cross-validation data and store results
		if save_all:
			loss, mean_vals, error_vals = nntrainer.test_model(sess, data['train'], 
																	name="train", 
																	batch_size=batch_size,
																	verbose=verbose)
		
			if 'X_test' in data.keys():
				loss, mean_vals, error_vals = nntrainer.test_model(sess, data['test'], 
																		name="test", 
																		batch_size=batch_size,
																		verbose=verbose)

		# test current model with cross-validation data and store results
		if 'valid' in data.keys():
			loss, mean_vals, error_vals = nntrainer.test_model(sess, data['valid'], 
																	name="valid", 
																	batch_size=batch_size,
																	verbose=verbose)
			# save model
			nntrainer.save_model(sess)

			# check for early stopping
			if patience:
				status = nntrainer.early_stopping(loss, patience)
				if not status:
					break

	nntrainer.save_model(sess, 'last')



def train_decay_learning_rate(sess, nntrainer, data, learning_rate=0.01, decay_rate=0.9, batch_size=128, 
					num_epochs=500, patience=10, verbose=1, shuffle=True, save_all=False):
	"""Train a model with cross-validation data and test data"""

	nntrainer.nnmodel.feed_dict['learning_rate'] = learning_rate
	for epoch in range(num_epochs):
		if verbose >= 1:
			sys.stdout.write("\rEpoch %d out of %d \n"%(epoch+1, num_epochs))
		else:
			if epoch % 10 == 0:
				sys.stdout.write("\rEpoch %d out of %d \n"%(epoch+1, num_epochs))

		nntrainer.nnmodel.feed_dict['learning_rate'] *= decay_rate
		nntrainer.update_feed_dict(nntrainer.nnmodel.placeholders, nntrainer.nnmodel.feed_dict)

		# training set
		train_loss = nntrainer.train_epoch(sess, data['train'], 
											batch_size=batch_size, 
											verbose=verbose, 
											shuffle=shuffle)

		# test current model with cross-validation data and store results
		if save_all:
			loss, mean_vals, error_vals = nntrainer.test_model(sess, data['train'], 
																	name="train", 
																	batch_size=batch_size,
																	verbose=verbose)
		
			if 'X_test' in data.keys():
				loss, mean_vals, error_vals = nntrainer.test_model(sess, data['test'], 
																		name="test", 
																		batch_size=batch_size,
																		verbose=verbose)

		# test current model with cross-validation data and store results
		if 'valid' in data.keys():
			loss, mean_vals, error_vals = nntrainer.test_model(sess, data['valid'], 
																	name="valid", 
																	batch_size=batch_size,
																	verbose=verbose)

			# save model
			nntrainer.save_model(sess)

			# check for early stopping
			if patience:
				status = nntrainer.early_stopping(loss, patience)
				if not status:
					break

	nntrainer.save_model(sess, 'last')

	return results




def train_anneal_batch_size(sess, nntrainer, train, valid, batch_schedule, 
							num_epochs=500, patience=10, verbose=1, shuffle=True, save_all=False):
	"""Train a model with cross-validation data and test data
			batch_schedule = {  0: 50, 
								20: 100,
								40: 500,
								55: 1000,
								50: 1500,
								65: 2000
								}	
	"""

	# train model
	for epoch in range(num_epochs):
		if verbose == 1:
			sys.stdout.write("\rEpoch %d out of %d \n"%(epoch+1, num_epochs))

		# change learning rate if on schedule
		if epoch in batch_schedule.keys():
			batch_size = batch_schedule[epoch]

		# training set
		train_loss = nntrainer.train_epoch(sess, data['train'], 
											batch_size=batch_size, 
											verbose=verbose, 
											shuffle=shuffle)

		# test current model with cross-validation data and store results
		if save_all:
			loss, mean_vals, error_vals = nntrainer.test_model(sess, data['valid'], 
																	name="valid", 
																	batch_size=batch_size,
																	verbose=verbose)
		
			if 'X_test' in data.keys():
				loss, mean_vals, error_vals = nntrainer.test_model(sess, data['test'], 
																		name="test", 
																		batch_size=batch_size,
																		verbose=verbose)

		# test current model with cross-validation data and store results
		if 'valid' in data.keys():
			loss, mean_vals, error_vals = nntrainer.test_model(sess, data['valid'], 
																	name="valid", 
																	batch_size=batch_size,
																	verbose=verbose)

			# save model
			nntrainer.save_model(sess)

			# check for early stopping
			if patience:
				status = nntrainer.early_stopping(loss, patience)
				if not status:
					break

	nntrainer.save_model(sess, 'last')


def train_anneal_learning_rate(nntrainer, train, valid, learning_rate_schedule, 
						batch_size=128, num_epochs=500, patience=10, verbose=1, shuffle=True, save_all=False):
	"""Train a model with cross-validation data and test data
			learning_rate_schedule = {  0: 0.001
										2: 0.01,
										5: 0.001,
										15: 0.0001
										}
	"""

	# train model
	for epoch in range(num_epochs):
		if verbose == 1:
			sys.stdout.write("\rEpoch %d out of %d \n"%(epoch+1, num_epochs))

		# change learning rate if on schedule
		if epoch in learning_rate_schedule:
			nntrainer.placeholders['learning_rate'] = np.float32(learning_rate_schedule[epoch])

		# training set
		train_loss = nntrainer.train_epoch(sess, data['train'], 
											batch_size=batch_size, 
											verbose=verbose, 
											shuffle=shuffle)

		if save_all:
			loss, mean_vals, error_vals = nntrainer.test_model(sess, data['valid'], 
																	name="valid", 
																	batch_size=batch_size,
																	verbose=verbose)
		
			if 'X_test' in data.keys():
				loss, mean_vals, error_vals = nntrainer.test_model(sess, data['test'], 
																		name="test", 
																		batch_size=batch_size,
																		verbose=verbose)

		# test current model with cross-validation data and store results
		if 'valid' in data.keys():
			loss, mean_vals, error_vals = nntrainer.test_model(sess, data['valid'], 
																	name="valid", 
																	batch_size=batch_size,
																	verbose=verbose)

			# save model
			nntrainer.save_model(sess)

			# check for early stopping
			if patience:
				status = nntrainer.early_stopping(loss, patience)
				if not status:
					break

	nntrainer.save_model(sess, 'last')

		