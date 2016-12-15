import os, sys



__all__ = [
    "train_minibatch"
]



def train_minibatch(sess, nntrainer, data, batch_size=128, num_epochs=500, 
                    patience=10, verbose=1, shuffle=True):
    """Train a model with cross-validation data and test data"""

    # train model
    for epoch in range(num_epochs):
        if verbose == 1:
            sys.stdout.write("\rEpoch %d out of %d \n"%(epoch+1, num_epochs))

        # training set
        train_loss = nntrainer.train_epoch(sess, 
                                            data['train'], 
                                            batch_size=batch_size, 
                                            verbose=verbose, 
                                            shuffle=shuffle)
        nntrainer.add_loss(train_loss, 'train') 

        # test current model with cross-validation data and store results
        if 'valid' in data.keys():
            valid_loss = nntrainer.test_model(sess, 
                                                data['valid'], 
                                                name="valid", 
                                                batch_size=batch_size)

        # save model
        nntrainer.save_model()

        # check for early stopping
        if patience:
            status = nntrainer.early_stopping(valid_loss, epoch, patience)
            if not status:
                break

    return nntrainer