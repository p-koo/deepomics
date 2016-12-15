

class InputLayer(BaseLayer):
	"""Input layer to feed in data"""
	def __init__(self, incoming, **kwargs):
		
		self.incoming_shape = incoming.get_shape()
		self.output = incoming        
		self.output_shape = incoming.get_shape()
		
	def get_input_shape(self):
		return self.incoming_shape
	
	def get_output(self):
		return self.output
	
	def get_output_shape(self):
		return self.output_shape    
