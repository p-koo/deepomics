import os, sys
import numpy as np
import matplotlib.image as img
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib as mpl
from matplotlib import rcParams
rcParams.update({'figure.autolayout': True})
from scipy.misc import imresize
import pandas as pd
import utils 



def plot_roc_all(final_roc):
	"""Plot ROC curve for each class"""

	fig = plt.figure()
	for i in range(len(final_roc)):
		plt.plot(final_roc[i][0],final_roc[i][1])
	plt.xlabel('False positive rate', fontsize=22)
	plt.ylabel('True positive rate', fontsize=22)
	plt.plot([0, 1],[0, 1],'k--')
	ax = plt.gca()
	ax.xaxis.label.set_fontsize(17)
	ax.yaxis.label.set_fontsize(17)
	map(lambda xl: xl.set_fontsize(13), ax.get_xticklabels())
	map(lambda yl: yl.set_fontsize(13), ax.get_yticklabels())
	plt.tight_layout()
	#plt.legend(loc='best', frameon=False, fontsize=14)
	return fig, plt


def plot_pr_all(final_pr):
	"""Plot PR curve for each class"""

	fig = plt.figure()
	for i in range(len(final_pr)):
		plt.plot(final_pr[i][0],final_pr[i][1])
	plt.xlabel('Recall', fontsize=22)
	plt.ylabel('Precision', fontsize=22)
	ax = plt.gca()
	ax.xaxis.label.set_fontsize(17)
	ax.yaxis.label.set_fontsize(17)
	map(lambda xl: xl.set_fontsize(13), ax.get_xticklabels())
	map(lambda yl: yl.set_fontsize(13), ax.get_yticklabels())
	plt.tight_layout()
	#plt.legend(loc='best', frameon=False, fontsize=14)
	return fig, plt


def plot_filter_logos(W, figsize=(2,10), height=25, nt_width=10, norm=0, alphabet='dna'):
	W =  np.squeeze(W)
	num_filters = W.shape[0]
	num_rows = int(np.ceil(np.sqrt(num_filters)))    
	grid = mpl.gridspec.GridSpec(num_rows, num_rows)
	grid.update(wspace=0.2, hspace=0.2, left=0.1, right=0.2, bottom=0.1, top=0.2) 
	fig = plt.figure(figsize=figsize);
	for i in range(num_filters):
		logo = seq_logo(W[i], height=height, nt_width=nt_width, norm=norm, alphabet=alphabet)
		plt.subplot(grid[i]);
		plot_seq_logo(logo, nt_width=nt_width, step_multiple=None)
		if np.mod(i, num_rows) != 0:
			plt.yticks([])
	return fig, plt


def plot_neg_logo(W, height=50, nt_width=20, alphabet='dna', figsize=(50,20)):

	num_rows = 2
	grid = mpl.gridspec.GridSpec(num_rows, 1)
	grid.update(wspace=0.2, hspace=0.0, left=0.1, right=0.2, bottom=0.1, top=0.2) 

	fig = plt.figure(figsize=figsize);

	plt.subplot(grid[0])
	pwm = utils.normalize_pwm(W, method=2)
	pos_logo = seq_logo(pwm, height=height, nt_width=nt_width, norm=0, alphabet=alphabet)
	plt.imshow(pos_logo, interpolation='none')
	plt.xticks([])
	plt.yticks([0, 100], ['2.0','0.0'])
	ax = plt.gca()
	ax.spines['right'].set_visible(False)
	ax.spines['top'].set_visible(False)
	ax.yaxis.set_ticks_position('none')
	ax.xaxis.set_ticks_position('none')


	plt.subplot(grid[1]);
	pwm = utils.normalize_pwm(-W, method=2)
	neg_logo = seq_logo(pwm, height=height, nt_width=nt_width, norm=0, alphabet=alphabet)
	plt.imshow(neg_logo[::-1,:,:], interpolation='none')
	plt.xticks([])
	plt.yticks([0, 100], ['0.0','2.0'])
	ax = plt.gca()
	ax.spines['right'].set_visible(False)
	ax.spines['bottom'].set_visible(False)
	ax.yaxis.set_ticks_position('none')
	ax.xaxis.set_ticks_position('none')
	return fig, plt


def plot_seq_logo(logo, nt_width=None, step_multiple=None):
	plt.imshow(logo, interpolation='none')
	if nt_width:
		num_nt = logo.shape[1]/nt_width
		if step_multiple:
			step_size = num_nt/(step_multiple+1)
			nt_range = range(step_size, step_size*step_multiple)              
			plt.xticks([step_size*nt_width, step_size*2*nt_width, step_size*3*nt_width, step_size*4*nt_width], 
						[str(step_size), str(step_size*2), str(step_size*3), str(step_size*4)])
		else:
			plt.xticks([])
		plt.yticks([0, 100], ['2.0','0.0'])
		ax = plt.gca()
		ax.spines['right'].set_visible(False)
		ax.spines['top'].set_visible(False)
		ax.yaxis.set_ticks_position('none')
		ax.xaxis.set_ticks_position('none')
	else:
		plt.imshow(logo, interpolation='none')
		plt.axis('off');
	return plt


def plot_neg_saliency(X, W, height=50, nt_width=20, alphabet='dna', figsize=(100,8)):

	num_rows = 3
	grid = mpl.gridspec.GridSpec(num_rows, 1)
	grid.update(wspace=0.2, hspace=0.2, left=0.1, right=0.2, bottom=0.1, top=0.2) 

	fig = plt.figure(figsize=figsize);

	plt.subplot(grid[0])
	pwm = utils.normalize_pwm(W, method=2)
	pos_logo = seq_logo(pwm, height=height, nt_width=nt_width, norm=0, alphabet=alphabet)
	plt.imshow(pos_logo, interpolation='none')
	plt.xticks([])
	plt.yticks([0, 100], ['2.0','0.0'])
	ax = plt.gca()
	ax.spines['right'].set_visible(False)
	ax.spines['top'].set_visible(False)
	ax.yaxis.set_ticks_position('none')
	ax.xaxis.set_ticks_position('none')

	plt.subplot(grid[1])
	logo = seq_logo(np.squeeze(X), height=height, nt_width=nt_width, norm=0, alphabet=alphabet)
	plt.imshow(logo, interpolation='none');
	plt.axis('off');

	plt.subplot(grid[2]);
	pwm = utils.normalize_pwm(-W, method=2)
	neg_logo = seq_logo(pwm, height=height, nt_width=nt_width, norm=0, alphabet=alphabet)
	plt.imshow(neg_logo[::-1,:,:], interpolation='none')
	plt.xticks([])
	plt.yticks([0, 100], ['0.0','2.0'])
	ax = plt.gca()
	ax.spines['right'].set_visible(False)
	ax.spines['bottom'].set_visible(False)
	ax.yaxis.set_ticks_position('none')
	ax.xaxis.set_ticks_position('none')
	return fig, plt


def get_filter_logo_scan(X, nnmodel, sess, layer='conv1', window=10, flip_filters=0):
	""" get the filter logo from the highest activations"""
	fmaps = nnmodel.get_activations(sess, layer, X)
	fmaps = np.squeeze(fmaps)
	X = np.squeeze(X)

	W_scan = []
	for filter_index in range(fmaps.shape[1]):
		
		# get filter scan
		scan = fmaps[:,filter_index,:]

		# get threshold
		threshold = np.max(scan)/2

		# find regions above threshold
		x, y = np.where(scan > threshold)

		# sort score 
		index = np.argsort(scan[x,y])[-1:0:-1]
		data_index = x[index].astype(int)
		pos_index = y[index].astype(int)

		if len(pos_index) > 100:
			seq = []
			for i in range(len(pos_index)):
				if (pos_index[i]-window >= 0) & (pos_index[i]+window <= scan.shape[1]):
					seq.append(X[data_index[i],:,pos_index[i]-window:pos_index[i]+window])
			if seq:
				seq = np.array(seq)
				seq = np.mean(seq,axis=0)
				if flip_filters:
					seq = seq[:,::-1]
				W_scan.append(seq)
			else:
				seq = np.ones((4,window*2+1))*.25
	return np.array(W_scan)




#------------------------------------------------------------------------------------------------
# helper functions

def fig_options(plt, options):
	if 'figsize' in options:
		fig = plt.gcf()
		fig.set_size_inches(options['figsize'][0], options['figsize'][1], forward=True)
	if 'ylim' in options:
		plt.ylim(options['ylim'][0],options['ylim'][1])
	if 'yticks' in options:
		plt.yticks(options['yticks'])
	if 'xticks' in options:
		plt.xticks(options['xticks'])
	if 'labelsize' in options:        
		ax = plt.gca()
		ax.tick_params(axis='x', labelsize=options['labelsize'])
		ax.tick_params(axis='y', labelsize=options['labelsize'])
	if 'axis' in options:
		plt.axis(options['axis'])
	if 'xlabel' in options:
		plt.xlabel(options['xlabel'], fontsize=options['fontsize'])
	if 'ylabel' in options:
		plt.ylabel(options['ylabel'], fontsize=options['fontsize'])
	if 'linewidth' in options:
		plt.rc('axes', linewidth=options['linewidth'])


def subplot_grid(nrows, ncols):
	grid= mpl.gridspec.GridSpec(nrows, ncols)
	grid.update(wspace=0.2, hspace=0.2, left=0.1, right=0.2, bottom=0.1, top=0.2) 
	return grid


def load_alphabet(filepath, alphabet):
	if (alphabet < 2) | (alphabet == 'dna') | (alphabet == 'rna'): # dna or rna
		"""load images of nucleotide alphabet """
		df = pd.read_table(os.path.join(filepath, 'A.txt'), header=None);
		A_img = df.as_matrix()
		A_img = np.reshape(A_img, [72, 65, 3], order="F").astype(np.uint8)

		df = pd.read_table(os.path.join(filepath, 'C.txt'), header=None);
		C_img = df.as_matrix()
		C_img = np.reshape(C_img, [76, 64, 3], order="F").astype(np.uint8)

		df = pd.read_table(os.path.join(filepath, 'G.txt'), header=None);
		G_img = df.as_matrix()
		G_img = np.reshape(G_img, [76, 67, 3], order="F").astype(np.uint8)

		if (alphabet == 1) | (alphabet == 'rna'): # RNA
			df = pd.read_table(os.path.join(filepath, 'U.txt'), header=None);
			T_img = df.as_matrix()
			T_img = np.reshape(T_img, [74, 57, 3], order="F").astype(np.uint8)
		else: # DNA
			df = pd.read_table(os.path.join(filepath, 'T.txt'), header=None);
			T_img = df.as_matrix()
			T_img = np.reshape(T_img, [72, 59, 3], order="F").astype(np.uint8)
		chars = [A_img, C_img, G_img, T_img]

	elif (alphabet == 2) | (alphabet == 'structure'): # structural profile
		df = pd.read_table(os.path.join(filepath, 'P.txt'), header=None);
		P_img = df.as_matrix()
		P_img =np. reshape(P_img, [64, 41, 3], order="F").astype(np.uint8)
		df = pd.read_table(os.path.join(filepath, 'E.txt'), header=None);
		E_img = df.as_matrix()      
		E_img = np.reshape(E_img, [64, 36, 3], order="F").astype(np.uint8)
		df = pd.read_table(os.path.join(filepath, 'H.txt'), header=None);
		H_img = df.as_matrix()        
		H_img = np.reshape(H_img, [64, 40, 3], order="F").astype(np.uint8)
		df = pd.read_table(os.path.join(filepath, 'I.txt'), header=None);
		I_img = df.as_matrix()       
		I_img = np.reshape(I_img, [64, 34, 3], order="F").astype(np.uint8)
		df = pd.read_table(os.path.join(filepath, 'M.txt'), header=None);
		M_img = df.as_matrix()
		M_img = np.reshape(M_img, [64, 42, 3], order="F").astype(np.uint8)
		chars = [P_img, H_img, I_img, M_img, E_img]

	elif (alphabet == 3) | (alphabet == 'pu'): # structural profile
		df = pd.read_table(os.path.join(filepath, 'P.txt'), header=None);
		P_img = df.as_matrix()
		P_img =np. reshape(P_img, [64, 41, 3], order="F").astype(np.uint8)
		df = pd.read_table(os.path.join(filepath, 'U_2.txt'), header=None);
		U_img = df.as_matrix()
		U_img = np.reshape(U_img, [64, 40, 3], order="F").astype(np.uint8)
		chars = [P_img, U_img]
	return chars


def seq_logo(pwm, height=30, nt_width=10, norm=0, alphabet='dna'):
	
	def get_nt_height(pwm, height, norm):
		
		def entropy(p):
			s = 0
			for i in range(len(p)):
				if p[i] > 0:
					s -= p[i]*np.log2(p[i])
			return s

		num_nt, num_seq = pwm.shape
		heights = np.zeros((num_nt,num_seq));
		for i in range(num_seq):
			if norm == 1:
				total_height = height
			else:
				total_height = (np.log2(num_nt) - entropy(pwm[:, i]))*height;
			heights[:,i] = np.floor(pwm[:,i]*np.minimum(total_height, height*2));
		return heights.astype(int)


	# get the alphabet images of each nucleotide
	package_directory = os.path.dirname(os.path.abspath(__file__))
	filepath = os.path.join(package_directory,'nt')
	chars = load_alphabet(filepath, alphabet)

	# get the heights of each nucleotide
	heights = get_nt_height(pwm, height, norm)

	# resize nucleotide images for each base of sequence and stack
	num_nt, num_seq = pwm.shape
	width = np.ceil(nt_width*num_seq).astype(int)

	max_height = height*2
	#total_height = np.sum(heights,axis=0) # np.minimum(np.sum(heights,axis=0), max_height)
	logo = np.ones((max_height, width, 3)).astype(int)*255;
	for i in range(num_seq):
		nt_height = np.sort(heights[:,i]);
		index = np.argsort(heights[:,i])
		remaining_height = np.sum(heights[:,i]);
		offset = max_height-remaining_height

		for j in range(num_nt):
			if nt_height[j] > 0:
				# resized dimensions of image
				nt_img = imresize(chars[index[j]], (nt_height[j], nt_width))

				# determine location of image
				height_range = range(remaining_height-nt_height[j], remaining_height)
				width_range = range(i*nt_width, i*nt_width+nt_width)

				# 'annoying' way to broadcast resized nucleotide image
				if height_range:
					for k in range(3):
						for m in range(len(width_range)):
							logo[height_range+offset, width_range[m],k] = nt_img[:,m,k];

				remaining_height -= nt_height[j]

	return logo.astype(np.uint8)
