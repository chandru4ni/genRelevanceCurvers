import sys
import cv2
import numpy as np
import caffe
import os

layers = {"conv1", "conv2", "conv3", "conv4", "conv5", "newfc6", "newfc7", "newfc8"}

layernumbers = {"conv1": "1", "conv2": "5", "conv3": "9", "conv4": "11", "newconv5": "13"}

def get_activations(input_file):

	f = open(input_file, 'r')
	input_image_file = f.read()
	input_image_file = input_image_file.split('\n')[0]
	input_image_file = input_image_file.split(' ')[0]

	model_file = 'newmodel/model_iter_60000.caffemodel'
	deploy_prototxt = 'fgvcdeploy.prototxt'

	net = caffe.Net(deploy_prototxt, model_file, caffe.TEST)
	temp = [(k, v.data.shape) for k, v in net.blobs.items()]
	
	imagemean_file = 'newmodel/2_class.npy'
	transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
	transformer.set_mean('data', np.load(imagemean_file).mean(1).mean(1))
	transformer.set_transpose('data', (2,0,1))
	transformer.set_raw_scale('data', 255.0)
	net.blobs['data'].reshape(1,3,227,227)
	img = caffe.io.load_image(input_image_file)
	net.blobs['data'].data[...] = transformer.preprocess('data', img)
	output = net.forward()

	for layername, layernumber in layernumbers.iteritems():
		a  = net.blobs[layername].data[0]
		d = []
		for i in range(len(a)):
			b = sum(map(sum, a[i]))
			d.append(b)
		d = np.asarray(d).astype(np.float)

		output_file = "act"+layername+".txt"

		with open(output_file, 'w') as f:
			np.savetxt(f, d, delimiter='\n' )


if __name__ == '__main__':
	get_activations("./outputimages/0063284.jpg")
