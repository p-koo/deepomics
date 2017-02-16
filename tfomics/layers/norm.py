import tensorflow as tf
from tensorflow.python.training import moving_averages
from .base import BaseLayer
from ..utils import Variable
from .. import init

__all__ = [
	"BatchNormLayer"
]



class BatchNormLayer(BaseLayer):
	"""1D convolutional layer"""

	def __init__(self, incoming, is_training, **kwargs):

		# input data shape
		self.incoming = incoming
		incoming_shape = self.incoming.get_output_shape()

		bn_shape = incoming_shape[-1]

		self.bn_axes = [0]
		if len(incoming_shape) > 2:
			self.bn_axes = [0, 1, 2]
		else:
			self.bn_axes = [0,1]

		self.gamma = Variable(var=init.Constant(value=1.), shape=[bn_shape], regularize=False)
		self.beta = Variable(var=init.Constant(value=0.), shape=[bn_shape], regularize=False)

		self.epsilon = 1e-08
		if 'epsilon' in kwargs.keys():
			self.epsilon = kwargs['epsilon']
		self.decay = 0.95
		if 'decay' in kwargs.keys():
			self.decay = kwargs['decay']

		self.is_training = is_training

		self.ema = tf.train.ExponentialMovingAverage(decay=self.decay)
		self.pop_mean = tf.Variable(tf.zeros(bn_shape), trainable=False)
		self.pop_var = tf.Variable(tf.ones(bn_shape), trainable=False)

	
	def get_output(self):
		batch_mean, batch_var = tf.nn.moments(self.incoming.get_output(), self.bn_axes)

		def update_mean_var():
			ema_apply_op = self.ema.apply([batch_mean, batch_var])
			with tf.control_dependencies([ema_apply_op]):
				return tf.identity(batch_mean), tf.identity(batch_var)

		mean, var = tf.cond(self.is_training, update_mean_var, 
							lambda: (self.ema.average(batch_mean), self.ema.average(batch_var)))
		return tf.nn.batch_normalization(self.incoming.get_output(), mean, var, 
		                                 self.beta.get_variable(), self.gamma.get_variable(), self.epsilon)

	def get_output_shape(self):
		return self.incoming.get_output_shape()


	def get_variable(self):
		return [self.gamma, self.beta]

	def set_trainable(self, status):
		self.gamma.set_trainable(status)
		self.beta.set_trainable(status)
		
	def is_trainable(self):
		return self.gamma.is_trainable()
		
	def is_l1_regularize(self):
		return self.gamma.is_l1_regularize()  
		
	def is_l2_regularize(self):
		return self.gamma.is_l2_regularize() 



"""
def get_output(self):
		return tf.select(self.is_training, self._batch_output(), self._pop_output())
	def _batch_output(self):
		mean, variance = tf.nn.moments(self.incoming.get_output(), self.bn_axes)
		update_moving_mean = moving_averages.assign_moving_average(
					moving_mean, mean, decay)
		update_moving_variance = moving_averages.assign_moving_average(
					moving_variance, variance, decay)

		with tf.control_dependencies([update_moving_mean, update_moving_variance]):
			return tf.identity(mean), tf.identity(variance)
	
	def _pop_output(self):
		return tf.nn.batch_normalization(self.incoming.get_output(), self.pop_mean, self.pop_var, 
												 self.beta.get_variable(), self.gamma.get_variable(), self.epsilon)
"""




