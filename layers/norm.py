import tensorflow as tf
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
			self.bn_axes = list(range(len(incoming_shape) - 1))

		self.gamma = Variable(var=init.Constant(value=1.), shape=[bn_shape], regularize=False)
		self.beta = Variable(var=init.Constant(value=0.), shape=[bn_shape], regularize=False)

		self.pop_mean = tf.Variable(tf.zeros(bn_shape), trainable=False)
		self.pop_var = tf.Variable(tf.ones(bn_shape), trainable=False)

		self.epsilon = 1e-08
		if 'epsilon' in kwargs.keys():
			self.epsilon = kwargs['epsilon']
		self.decay = 0.95
		if 'decay' in kwargs.keys():
			self.epsilon = kwargs['decay']

		self.is_training = is_training

	
	def get_output(self):
		return tf.select(self.is_training, self._batch_output(), self._pop_output())

	def _batch_output(self):
		batch_mean, batch_var = tf.nn.moments(self.incoming.get_output(), self.bn_axes)
		train_mean = tf.assign(self.pop_mean, self.pop_mean*self.decay + batch_mean*(1 - self.decay))
		train_var = tf.assign(self.pop_var, self.pop_var*self.decay + batch_var*(1 - self.decay))
		with tf.control_dependencies([train_mean, train_var]):
			return tf.nn.batch_normalization(self.incoming.get_output(), batch_mean, batch_var, 
										 self.beta.get_variable(), self.gamma.get_variable(), self.epsilon)
	
	def _pop_output(self):
		return tf.nn.batch_normalization(self.incoming.get_output(), self.pop_mean, self.pop_var, 
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







