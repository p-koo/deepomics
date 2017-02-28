from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from .base import BaseLayer

		
__all__ = [
	"DropoutLayer",
	"GaussianNoiseLayer",
	"VariationalSampleLayer"
]


class DropoutLayer(BaseLayer):
	def __init__(self, incoming, keep_prob=0.5, **kwargs):
				
		self.incoming_shape = incoming.get_output_shape()
		self.incoming = incoming
		self.keep_prob = keep_prob
		self.output_shape = self.output.get_shape()
		
	def get_input_shape(self):
		return self.incoming_shape
	
	def get_output(self):
		return tf.nn.dropout(self.incoming.get_output(), keep_prob=self.keep_prob)
	
	def get_output_shape(self):
		return self.output_shape


class GaussianNoiseLayer(BaseLayer):
	def __init__(self, incoming, mu=0.0, sigma=0.1, **kwargs):
				
		self.incoming_shape = incoming.get_output_shape()
		self.incoming = incoming
		self.output_shape = self.incoming_shape	
		self.mu = mu
		self.sigma = sigma

		
	def get_input_shape(self):
		return self.incoming_shape
	
	def get_output(self):
		noise = tf.random_normal(shape=self.incoming_shape, mean=self.mu, stddev=self.sigma, dtype=tf.float32) 
	    return self.incoming.get_output() + noise
	
	def get_output_shape(self):
		return self.output_shape
		


class VariationalSampleLayer(BaseLayer):
	def __init__(self, incoming_mu, incoming_sigma, **kwargs):
				
		self.incoming_mu = incoming_mu
		self.incoming_sigma = incoming_sigma
		self.incoming_shape = incoming_mu.get_output_shape()
		self.output_shape = self.incoming_shape		
		
	def get_input_shape(self):
		return self.incoming_shape
	
	def get_output(self):
		z = tf.random_normal(shape=self.incoming_shape) 
		std_encoder = tf.exp(0.5 * incoming_sigma.get_output())
		return self.incoming_mu.get_output() + tf.mul(std_encoder, epsilon)

	def get_output_shape(self):
		return self.output_shape
		


