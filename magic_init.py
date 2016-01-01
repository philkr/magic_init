
INPUT_LAYERS = ['Data', 'ImageData']
PARAMETER_LAYERS = ['Convolution', 'InnerProduct']
SUPPORTED_LAYERS = ['ReLU', 'Sigmoid', 'LRN', 'Pooling']
STRIP_LAYER = ['Softmax', 'SoftmaxWithLoss', 'SigmoidCrossEntropyLoss']
# Use 'Dropout' at your own risk
# Unless Jon merges #2865 , 'Split' cannot be supported
UNSUPPORTED_LAYERS = ['Split', 'BatchNorm', 'Reshape']

def forward(net, i, NIT, data, output_names):
	n = net._layer_names[i]
	# Create the top data if needed
	output = {t: [None]*NIT for t in output_names}
	for it in range(NIT):
		for b in data:
			net.blobs[b].data[...] = data[b][it]
		net._forward(i, i)
		for t in output_names:
			output[t][it] = 1*net.blobs[t].data
	return output

def flattenData(data):
	import numpy as np
	return np.concatenate([d.swapaxes(0, 1).reshape((d.shape[1],-1)) for d in data], axis=1).T

def gatherInputData(net, layer_id, bottom_data, top_name):
	# This functions gathers all input data.
	# In order to not replicate all the internal functionality of convolutions (eg. padding ...)
	# we gather the data in the output space and use random gaussian weights. The output of this
	# function is W and D, there the input data I = D * W^-1  [with some abuse of tensor notation]
	# If we not compute an initialization A for D, we then simply multiply A by W to obtain the
	# proper initialization in the input space
	import numpy as np
	l = net.layers[layer_id]
	NIT = len(list(bottom_data.values())[0])
	# How many times do we need to over-sample to get a full basis (out of random projections)
	OS = int(np.ceil( np.prod(l.blobs[0].data.shape[1:]) / l.blobs[0].data.shape[0] ))
	# Note this could cause some memory issues in the FC layers
	W, D = [], []
	for i in range(OS):
		d = l.blobs[0].data
		d[...] = np.random.normal(0, 1, d.shape)
		W.append(1*d)
		D.append(np.concatenate(forward(net, layer_id, NIT, bottom_data, [top_name])[top_name], axis=0))
	return np.concatenate(W, axis=0), np.concatenate(D, axis=1)

def initializeWeight(D, type, N_OUT):
	# Here we first whiten the data (PCA or ZCA) and then optionally run k-means
	# on this whitened data.
	import numpy as np
	if D.shape[0] < N_OUT:
		print( "  Not enough data for '%s' estimation, using elwise"%type )
		return np.random.normal(0, 1, (N_OUT,D.shape[1]))
	D = D - np.mean(D, axis=0, keepdims=True)
	# PCA, ZCA, K-Means
	assert type in ['pca', 'zca', 'kmeans', 'rand'], "Unknown initialization type '%s'"%type
	C = np.cov(D.T)
	s, V = np.linalg.eigh(C)
	# order the eigenvalues
	ids = np.argsort(s)[-N_OUT:]
	s = s[ids]
	V = V[:,ids]
	s[s<1e-6] = 0
	s[s>=1e-6] = 1. / np.sqrt(s[s>=1e-6]+1e-3)
	S = np.diag(s)
	if type == 'pca':
		return S.dot(V.T)
	elif type == 'zca':
		return V.dot(S.dot(V.T))
	# Whiten the data
	wD = D.dot(V.dot(S))
	wD /= np.linalg.norm(wD, axis=1)[:,None]

	if type == 'kmeans':
		# Run k-means
		from sklearn.cluster import MiniBatchKMeans
		km = MiniBatchKMeans(n_clusters = wD.shape[1], batch_size=10*wD.shape[1]).fit(wD).cluster_centers_
	elif type == 'rand':
		km = wD[np.random.choice(wD.shape[0], wD.shape[1], False)]
	C = km.dot(S.dot(V.T))
	C /= np.std(D.dot(C.T), axis=0, keepdims=True).T
	return C
		

def initializeLayer(net, layer_id, bottom_data, top_name, bias=0, type='elwise'):
	import numpy as np
	l = net.layers[layer_id]
	NIT = len(list(bottom_data.values())[0])
	
	for p in l.blobs: p.data[...] = 0
	# Initialize the weights [k-means, ...]
	if type == 'elwise':
		d = l.blobs[0].data
		d[...] = np.random.normal(0, 1, d.shape)
	else: # Use the input data
		# Gather the input data
		T, D = gatherInputData(net, layer_id, bottom_data, top_name)
		
		# Figure out the output dimensionality of d
		d = l.blobs[0].data
		
		# Prepare the data
		D = D.swapaxes(0, 1).reshape((D.shape[1],-1)).T
		
		# Compute the weights
		W = initializeWeight(D, type, N_OUT=d.shape[0])
		
		# Multiply the weights by the random basis
		# NOTE: This matrix multiplication is a bit large, if it's too slow,
		#       reduce the oversampling in gatherInputData
		d[...] = np.dot(W, T.reshape((T.shape[0],-1))).reshape(d.shape)
	
	# Scale the mean and initialize the bias
	top_data = forward(net, layer_id, NIT, bottom_data, [top_name])[top_name]
	flat_data = flattenData(top_data)
	mu = flat_data.mean(axis=0)
	std = flat_data.std(axis=0)
	l.blobs[0].data[...] /= std.reshape((-1,)+(1,)*(len(l.blobs[0].data.shape)-1))
	for b in l.blobs[1:]:
		b.data[...] = -mu / std + bias



def magicInitialize(net, bias=0, NIT=10, type='elwise', bottom_names={}, top_names={}):
	import numpy as np
	# When was a blob last used
	last_used = {}
	# Make sure all layers are supported, and compute the last time each blob is used
	for i, (n, l) in enumerate(zip(net._layer_names, net.layers)):
		if l.type in UNSUPPORTED_LAYERS:
			print( "WARNING: Layer type '%s' not supported! Things might go very wrong..."%l.type )
		elif l.type not in SUPPORTED_LAYERS+PARAMETER_LAYERS+INPUT_LAYERS+STRIP_LAYER:
			print( "Unknown layer type '%s'. double check if it is supported"%l.type )
		for b in bottom_names[n]:
			last_used[b] = i
	
	active_data = {}
	# Read all the input data
	for i, (n, l) in enumerate(zip(net._layer_names, net.layers)):
		# Initialize the layer
		if len(l.blobs) > 0:
			print( "Initializing layer '%s'"%n )
			assert l.type in PARAMETER_LAYERS, "Unsupported parameter layer"
			assert len(top_names[n]) == 1, "Exactly one output supported"
			if np.sum(np.abs(l.blobs[0].data)) <= 1e-10:
				# Fill the parameters
				initializeLayer(net, i, {b: active_data[b] for b in bottom_names[n]}, top_names[n][0], bias, type)
				
			# TODO: Estimate and rescale the values [TODO: Record and undo this scaling above]
		
		# Run the network forward
		new_data = forward(net, i, NIT, {b: active_data[b] for b in bottom_names[n]}, top_names[n])
		active_data.update(new_data)
		
		# Delete all unused data
		for k in list(active_data):
			if k not in last_used or last_used[k] == i:
				del active_data[k]

def load(net, blobs):
	for l,n in zip(net.layers, net._layer_names):
		if n in blobs:
			for b, sb in zip(l.blobs, blobs[n]):
				b.data[...] = sb

def save(net):
	import numpy as np
	r = {}
	for l,n in zip(net.layers, net._layer_names):
		if len(l.blobs) > 0:
			r[n] = [np.copy(b.data) for b in l.blobs]
	return r

def estimateHomogenety(net, bottom_names={}, top_names={}):
	# Estimate if a certain layer is homogeneous and if yes return the degree k
	# by which the output is scaled (if input is scaled by alpha then the output
	# is scaled by alpha^k). Return None if the layer is not homogeneous.
	import numpy as np
	# When was a blob last used
	last_used = {}
	# Make sure all layers are supported, and compute the range each blob is used in
	for i, (n, l) in enumerate(zip(net._layer_names, net.layers)):
		for b in bottom_names[n]:
			last_used[b] = i
	
	active_data = {}
	homogenety = {}
	# Read all the input data
	for i, (n, l) in enumerate(zip(net._layer_names, net.layers)):
		# Run the network forward
		new_data1 = forward(net, i, 1, {b: [1*d for d in active_data[b]] for b in bottom_names[n]}, top_names[n])
		new_data2 = forward(net, i, 1, {b: [2*d for d in active_data[b]] for b in bottom_names[n]}, top_names[n])
		active_data.update(new_data1)
		
		if len(new_data1) == 1:
			m = list(new_data1.keys())[0]
			d1, d2 = flattenData(new_data1[m]), flattenData(new_data2[m])
			f = np.mean(np.abs(d1), axis=0) / np.mean(np.abs(d2), axis=0)
			if 1e-3*np.mean(f) < np.std(f):
				# Not homogeneous
				homogenety[n] = None
			else:
				# Compute the degree of the homogeneous transformation
				homogenety[n] = (np.log(np.mean(np.abs(d2))) - np.log(np.mean(np.abs(d1)))) / np.log(2)
		else:
			homogenety[n] = None
		# Delete all unused data
		for k in list(active_data):
			if k not in last_used or last_used[k] == i:
				del active_data[k]
	return homogenety

def calibrateGradientRatio(net, NIT=1, bottom_names={}, top_names={}):
	import numpy as np
	# When was a blob last used
	last_used = {}
	# Find the last layer to use
	last_layer = 0
	for i, (n, l) in enumerate(zip(net._layer_names, net.layers)):
		if l.type not in STRIP_LAYER:
			last_layer = i
		for b in bottom_names[n]:
			last_used[b] = i
	# Figure out which tops are involved
	last_tops = top_names[net._layer_names[last_layer]]
	
	# Call forward and store the data of all data layers
	active_data, input_data, bottom_scale = {}, {}, {}
	# Read all the input data
	for i, (n, l) in enumerate(zip(net._layer_names, net.layers)):
		if i > last_layer: break
		# Compute the input scale for parameter layers
		if len(l.blobs) > 0:
			bottom_scale[n] = np.mean([np.mean(np.abs(active_data[b])) for b in bottom_names[n]])
		# Run the network forward
		new_data = forward(net, i, NIT, {b: active_data[b] for b in bottom_names[n]}, top_names[n])
		if l.type in INPUT_LAYERS:
			input_data.update(new_data)
		active_data.update(new_data)
		
		# Delete all unused data
		for k in list(active_data):
			if k not in last_used or last_used[k] == i:
				del active_data[k]
	output_std = np.mean(np.std(flattenData(list(active_data.values())[0]), axis=0))
	
	for it in range(10):
		# Reset the diffs
		for l in net.layers:
			for b in l.blobs:
				b.diff[...] = 0
		# Set the top diffs
		for t in last_tops:
			net.blobs[t].diff[...] = np.random.normal(0, 1, net.blobs[t].shape)
		# Compute all gradients
		net._backward(last_layer, 0)
		
		# Compute the gradient ratio
		ratio={}
		for i, (n, l) in enumerate(zip(net._layer_names, net.layers)):
			if len(l.blobs) > 0:
				assert l.type in PARAMETER_LAYERS, "Parameter layer '%s' currently not supported"%l.type
				b = l.blobs[0]
				ratio[n] = np.mean(np.abs(b.diff)) / np.mean(np.abs(b.data))
		
		# If all layers are homogeneous, then the target ratio is the geometric mean of all ratios
		# (assuming we want the same output)
		# To deal with non-homogeneous layers we scale by output_std in the hope to undo correct the
		# estimation over time.
		# NOTE: for non feed-forward networks the geometric mean might not be the right scaling factor
		target_ratio = np.exp(np.mean(np.log(np.array(list(ratio.values()))))) * (output_std)**(1. / len(ratio))
		
		# Terminate if the relative change is less than 1% for all values
		log_ratio = np.log( np.array(list(ratio.values())) )
		if np.all( np.abs(log_ratio/np.log(target_ratio) - 1) < 0.01 ):
			break
		
		# Update all the weights and biases
		active_data = {}
		# Read all the input data
		for i, (n, l) in enumerate(zip(net._layer_names, net.layers)):
			if i > last_layer: break
			# Use the stored input
			if l.type in INPUT_LAYERS:
				active_data.update({b: input_data[b] for b in top_names[n]})
			else:
				if len(l.blobs) > 0:
					# Add the scaling from the bottom to the biases
					current_scale = np.mean([np.mean(np.abs(active_data[b])) for b in bottom_names[n]])
					adj = current_scale / bottom_scale[n]
					for b in list(l.blobs)[1:]:
						b.data[...] *= adj
					bottom_scale[n] = current_scale
					
					# Scale to obtain the target ratio
					scale = np.sqrt(ratio[n] / target_ratio)
					for b in l.blobs:
						b.data[...] *= scale
					
				active_data.update(forward(net, i, NIT, {b: active_data[b] for b in bottom_names[n]}, top_names[n]))
			# Delete all unused data
			for k in list(active_data):
				if k not in last_used or last_used[k] == i:
					del active_data[k]
		
		new_output_std = np.mean(np.std(flattenData(list(active_data.values())[0]), axis=0))
		if np.abs(np.log(output_std) - np.log(new_output_std)) > 0.25:
			# If we diverge by a factor of exp(0.25) = ~1.3, then we should check if the network is really
			# homogeneous
			print( "WARNING: It looks like one or more layers are not homogeneous! Trying to correct for this..." )
			print( "         Output std = %f" % new_output_std )
		output_std = new_output_std

def netFromString(s, t=None):
	import caffe
	from tempfile import NamedTemporaryFile
	if t is None: t = caffe.TEST
	f = NamedTemporaryFile('w')
	f.write(s)
	f.flush()
	r = caffe.Net(f.name, t)
	f.close()
	return r

def layerTypes(net_proto):
	return {l.name: l.type for l in net_proto.layer}

def layerTops(net_proto):
	return {l.name: list(l.top) for l in net_proto.layer}

def layerBottoms(net_proto):
	return {l.name: list(l.bottom) for l in net_proto.layer}

def getFileList(f):
	from glob import glob
	from os import path
	return [f for f in glob(f) if path.isfile(f)]

def main():
	from argparse import ArgumentParser
	from os import path
	
	parser = ArgumentParser()
	parser.add_argument('prototxt')
	parser.add_argument('output_caffemodel')
	parser.add_argument('-l', '--load', help='Load a pretrained model and rescale it [bias and type are not supported]')
	parser.add_argument('-d', '--data', default=None, help='Image list to use [default prototxt data]')
	parser.add_argument('-b', '--bias', type=float, default=0.1, help='Bias')
	parser.add_argument('-t', '--type', default='elwise', help='Type: elwise, pca, zca, kmeans, rand (random input patches)')
	parser.add_argument('-z', action='store_true', help='Zero all weights and reinitialize')
	parser.add_argument('-cs',  action='store_true', help='Correct for scaling')
	parser.add_argument('-q', action='store_true', help='Quiet execution')
	parser.add_argument('-s', type=float, default=1.0, help='Scale the input [only custom data "-d"]')
	parser.add_argument('-bs', type=int, default=16, help='Batch size [only custom data "-d"]')
	parser.add_argument('-nit', type=int, default=10, help='Number of iterations')
	parser.add_argument('--gpu', type=int, default=0, help='What gpu to run it on?')
	args = parser.parse_args()
	
	if args.q:
		from os import environ
		environ['GLOG_minloglevel'] = '2'
	import caffe, load
	from caffe import NetSpec, layers as L
	
	caffe.set_mode_gpu()
	if args.gpu is not None:
		caffe.set_device(args.gpu)
	
	model = load.ProtoDesc(args.prototxt)
	net = NetSpec()
	if args.data is not None:
		fl = getFileList(args.data)
		if len(fl) == 0:
			print("Unknown data type for '%s'"%args.data)
			exit(1)
		from tempfile import NamedTemporaryFile
		f = NamedTemporaryFile('w')
		f.write('\n'.join([path.abspath(i)+' 0' for i in fl]))
		f.flush()
		net.data, net.label = L.ImageData(source=f.name, batch_size=args.bs, new_width=model.input_dim[-1], new_height=model.input_dim[-1], transform_param=dict(mean_value=[104,117,123], scale=args.s),ntop=2)
		net.out = model(data=net.data, label=net.label)
	else:
		net.out = model()
	
	net_proto = net.to_proto()
	n = netFromString('force_backward:true\n'+str(net_proto), caffe.TRAIN )
	layer_top = layerTops( net_proto )
	layer_bottoms = layerBottoms( net_proto )
	
	if args.load is not None:
		n.copy_from(args.load)
		# Rescale existing layers?
		if args.fix:
			magicFix(n, args.nit)

	if args.z:
		# Zero out all layers
		for l in n.layers:
			for b in l.blobs:
				b.data[...] = 0

	magicInitialize(n, args.bias, NIT=args.nit, type=args.type, top_names=layer_top, bottom_names=layer_bottoms)
	if args.cs:
		#print( estimateHomogenety(n, top_names=layer_top, bottom_names=layer_bottoms) )
		calibrateGradientRatio(n, top_names=layer_top, bottom_names=layer_bottoms)
	n.save(args.output_caffemodel)

if __name__ == "__main__":
	main()