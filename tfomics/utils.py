from __future__ import print_function 
import os, sys
import tensorflow as tf
from .init import *


__all__ = [
	"placeholder",
	"Variable",
	"make_directory",
	"normalize_pwm",
	"meme_generate"
]


def placeholder(shape, dtype=tf.float32, name=None):
	return tf.placeholder(dtype=dtype, shape=shape, name=name)
	
			
class Variable():
	def __init__(self, var, shape, **kwargs):

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

		if self.name:
			self.variable = tf.Variable(var(shape), name=self.name)
		else:
			self.variable = tf.Variable(var(shape))
		
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




def normalize_pwm(pwm, method=2):
	if method == 1:
		pwm = pwm/np.max(np.abs(pwm))
		pwm += .25
		pwm[pwm<0] = 0
	elif method == 2:
		MAX = np.max(np.abs(pwm))
		pwm = pwm/MAX*4
		pwm = np.exp(pwm)

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
	

