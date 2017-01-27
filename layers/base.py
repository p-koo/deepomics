
__all__ = [
	"BaseLayer",
	"InputLayer",
]


class BaseLayer(object):
	"""Base class for neural network layers."""
	def __init__(self, name=None):
		self.name = name

	def output(self):
		raise NotImplementedError()
		



class InputLayer(BaseLayer):
	"""Input layer to feed in data"""
	def __init__(self, incoming, **kwargs):
		
		self.incoming = incoming        
		
	def output(self):
		return self.incoming
	
	def output_shape(self):
		return self.incoming.get_shape()
