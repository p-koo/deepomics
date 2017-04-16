from __future__ import print_function 
import os, sys
import numpy as np
import tensorflow as tf
from .init import *



__all__ = [
	"placeholder",
	"Variable",
	"make_directory",
	"normalize_pwm",
	"meme_generate"
]


def initialize_session(placeholders=None):
	# run session
	sess = tf.Session()

	# initialize variables
	if placeholders is None:
		sess.run(tf.global_variables_initializer()) 

	else:
		if 'is_training' in placeholders:
			sess.run(tf.global_variables_initializer(), feed_dict={placeholders['is_training']: True}) 
		else:
			sess.run(tf.global_variables_initializer()) 
	return sess


def placeholder(shape, dtype=tf.float32, name=None):
	return tf.placeholder(dtype=dtype, shape=shape, name=name)
	
			
class Variable():
	def __init__(self, var, shape, reverse=False, **kwargs):

		self.l1_regularize = True
		if 'l1' in kwargs.keys():
			self.l1_regularize = kwargs['l1']
		
		self.l2_regularize = True
		if 'l2' in kwargs.keys():
			self.l2_regularize = kwargs['l2']
		
		if 'regularize' in kwargs.keys():
			if not kwargs['regularize']:
				self.l1_regularize = False
				self.l2_regularize = False

		self.trainable = True
		if 'trainable' in kwargs.keys():
			self.l1_regularize = kwargs['trainable']

		self.name = None
		if 'name' in kwargs.keys():
			self.name = kwargs['name']
			
		self.shape = shape

		variable = var(shape)
		if reverse:
			var2 = tf.reverse(variable, axis=[0, 2])
			variable = tf.concat([variable, var2], axis=3)

		if self.name:
			self.variable = tf.Variable(variable, name=self.name)
		else:
			self.variable = tf.Variable(variable)
		
	def get_variable(self):
		return self.variable
	
	def get_shape(self):
		return self.shape
	
	def set_l1_regularize(self, status):
		self.l1_regularize = status
		
	def set_l2_regularize(self, status):
		self.l2_regularize = status
		
	def set_trainable(self, status):
		self.trainable = status
			
	def is_l1_regularize(self):
		return self.l1_regularize
		
	def is_l2_regularize(self):
		return self.l2_regularize
		
	def is_trainable(self):
		return self.trainable


def make_directory(path, foldername, verbose=1):
	"""make a directory"""

	if not os.path.isdir(path):
		os.mkdir(path)
		print("making directory: " + path)

	outdir = os.path.join(path, foldername)
	if not os.path.isdir(outdir):
		os.mkdir(outdir)
		print("making directory: " + outdir)
	return outdir



def normalize_pwm(pwm, MAX=None, factor=4):

	if len(pwm.shape) > 2:
		pwm_norm = []
		for p in range(pwm.shape[0]):
			if MAX is None:
					MAX = np.max(np.abs(pwm[p]))
			pwm[p] = pwm[p]/MAX*factor
			pwm[p] = np.exp(pwm[p])
			norm = np.outer(np.ones(pwm[p].shape[0]), np.sum(np.abs(pwm[p]), axis=0))
			pwm_norm.append([pwm[p]/norm]) 
		return np.vstack(pwm_norm)
	else:
		if MAX is None:
				MAX = np.max(np.abs(pwm))
		pwm = pwm/MAX*factor
		pwm= np.exp(pwm)
		norm = np.outer(np.ones(pwm.shape[0]), np.sum(np.abs(pwm), axis=0))
		return pwm/norm


def meme_generate(W, output_file='meme.txt', prefix='filter'):

	# background frequency        
	nt_freqs = [1./4 for i in range(4)]

	# open file for writing
	f = open(output_file, 'w')

	# print intro material
	f.write('MEME version 4')
	f.write('')
	f.write('ALPHABET= ACGT')
	f.write('')
	f.write('Background letter frequencies:')
	f.write('A %.4f C %.4f G %.4f T %.4f' % tuple(nt_freqs))
	f.write('')

	for j in range(len(W)):
		pwm = np.array(W[j])

		f.write('MOTIF %s%d' % (prefix, j))
		f.write('letter-probability matrix: alength= 4 w= %d nsites= %d' % (pwm.shape[1], 1000))
		for i in range(pwm.shape[1]):
			f.write('%.4f %.4f %.4f %.4f' % tuple(pwm[:,i]))
		f.write('')

	f.close()
	

