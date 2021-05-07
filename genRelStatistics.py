import shutil
import matplotlib.pyplot as plot
import numpy as np
import plotly.plotly as py
import plotly.tools as tls
import os
import sys
from extract_features import get_activations
from libGenRelStatistics import *

DoFlights = True
if DoFlights == True:
	layers = {"conv1", "conv2", "conv3", "conv4", "newconv5", "fc6", "fc7"}

	layernumbers = {"conv1": "1", "conv2": "5", "conv3": "9", "conv4": "11", "newconv5": "13", "fc6": "16", "fc7": "19"}
	f = open('flightfilelist.txt', 'r')
else:
	layers = {"conv1", "conv2", "ip1"}
	layernumbers = {"conv1": "1", "conv2": "5", "ip1": "9"}
	f = open('digitfilelist.txt', 'r')

flist = f.readlines()
count = 0
spacestring = "                                           "
file_header = "FILE NAME:     LAYERNAME: PureFilterCount: OthersCount: MisC(Y/N)"
outfile = open('output.txt', 'w')
outfile.write(file_header)

os.system("cp results.txt results_old.txt")
os.system("rm results.txt")

for fname in flist:
	if DoFlights == True:
		tf = open('testfilelist_flight.txt', 'w')
	else:
		tf = open('testfilelist_digit.txt', 'w')
	tf.write(fname.split('\n')[0])
	tf.close()

	tempfname = fname.split('/')[2]
	tempfname = tempfname.split(' ')[0]

	misClassFlag = ""
	#if count == 4:
	#	count = 0

	if DoFlights == True:
		os.system('./lrp_demo ./fgvc2config.txt ./testfilelist_flight.txt ./')
	else:
		os.system('./lrp_demo ./digitsconfig.txt ./testfilelist_digit.txt ./')
	#get_activations('testfilelist.txt')
	#os.system('python avgAScores.py')
	tfname = "./lrp_output/outputimages/"+tempfname+"_top10scores.txt"
	tempf = open(tfname, 'r')
	classname = tempf.read(1)
	tempf.close()
	if classname == "0":
		targetclass = "P"
		firstclass = "P"
		secondclass = "F"
		firstclassString = "Passenger"
		secondclassString = "Fighter"
	if classname == "1":
		targetclass = "F"
		firstclass = "P"
		secondclass = "F"
		firstclassString = "Passenger"
		secondclassString = "Fighter"

	for layername, layernumber in layernumbers.iteritems():
		if layername == "fc6" or layername == "fc7":
			continue
		if DoFlights == True:
			totalRelevances = 250
			topRelCount = 30
			commonSeqCount = 30
		else:
			totalRelevances = 200
			if layername == "conv1":
				topRelCount = 10
				commonSeqCount = 10
			if layername == "conv2":
				topRelCount = 20
				commonSeqCount = 20
			if layername == "ip1":
				topRelCount = 30
				commonSeqCount = 30


	resultsfilename = open("results.txt", "a")
	resultsfilename.write(tempfname+" ")
	resultsfilename.close()
	tfname = "./lrp_output/outputimages/"+tempfname+"_top10scores.txt"
	tempf = open(tfname, 'r')
	classname = tempf.read(1)
	tempf.close()
	RelAcrossLayersCommonDist(misClassFlag, targetclass, totalRelevances, topRelCount, commonSeqCount, firstclass, secondclass, firstclassString, secondclassString, classname)
	resultsfilename = open("results.txt", "a")
	resultsfilename.write("\n")
	resultsfilename.close()
	#shutil.move("RelAcrossLayers.png", './Images/'+str(targetclass)+'/seqRelAcrossLayers'+"_"+targetclass+".png")

	#layerwise_montage_images(targetclass)

	#count = count+1
	#if count == 3:
	#	exit()
outfile.write("\n")
outfile.close()
