import os, sys



__all__ = [
	"train_minibatch",
	"train_anneal_batch_size",
	"train_anneal_learning_rate"
]



def train_minibatch(sess, nntrainer, data, batch_size=128, num_epochs=500, 
					patience=10, verbose=1, shuffle=True):
	"""Train a model with cross-validation data and test data"""

	# train model
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
		if 'valid' in data.keys():
			valid_loss = nntrainer.test_model(sess, data['valid'], 
												name="valid", 
												batch_size=batch_size,
												verbose=verbose)

			# save model
			nntrainer.save_model(sess, epoch)

			# check for early stopping
			if patience:
				status = nntrainer.early_stopping(valid_loss, epoch, patience)
				if not status:
					break

	return nntrainer



def train_anneal_batch_size(sess, nntrainer, train, valid, batch_schedule, 
							num_epochs=500, patience=10, verbose=1, shuffle=True):
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
		if 'valid' in data.keys():
			valid_loss = nntrainer.test_model(sess, data['valid'], 
												name="valid", 
												batch_size=batch_size,
												verbose=verbose)

			# save model
			nntrainer.save_model(sess)

			# check for early stopping
			if patience:
				status = nntrainer.early_stopping(valid_loss, epoch, patience)
				if not status:
					break

	return nntrainer




def train_anneal_learning_rate(sess, nntrainer, train, valid, learning_rate_schedule, 
						batch_size=128, num_epochs=500, patience=10, verbose=1, shuffle=True):
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
		train_loss = nntrainer.train_epoch(sess, 
											data['train'], 
											batch_size=batch_size, 
											verbose=verbose, 
											shuffle=shuffle)

		# test current model with cross-validation data and store results
		if 'valid' in data.keys():
			valid_loss = nntrainer.test_model(sess, 
												data['valid'], 
												name="valid", 
												batch_size=batch_size,
												verbose=verbose)

			# save model
			nntrainer.save_model(sess)

			# check for early stopping
			if patience:
				status = nntrainer.early_stopping(valid_loss, epoch, patience)
				if not status:
					break

	return nntrainer
	
		