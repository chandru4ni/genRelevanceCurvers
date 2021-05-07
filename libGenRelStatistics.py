import matplotlib.pyplot as plot
from collections import Counter
import numpy as np
import seaborn as sns
import plotly.tools as tls
import os
import sys
from matplotlib.colors import ListedColormap
import shutil
import glob

DoFlights = True
if DoFlights == True:
	layers = {"conv1", "conv2", "conv3", "conv4", "conv5", "fc6", "fc7", "fc8"}

	layernumbers = {"conv1": "1", "conv2": "5", "conv3": "9", "conv4": "11", "newconv5": "13", "fc6": "16", "fc7": "19"}
else:
	layers = {"conv1", "conv2", "ip1"}
	layernumbers = {"conv1": "1", "conv2": "5", "ip1": "9"}

def RelAndActsAcrossLayersDist(misClassFlag, targetclass, firstclass, secondclass, firstclassString, secondclassString):
  	fig = plot.figure(figsize=(10,6))
  	ax = fig.add_subplot(111)

	testinfo = firstclassString
	if misClassFlag != "":
		testinfo = "Misclassified " +  misClassFlag

	if DoFlights == True:
  		fig.suptitle("Layerwise Learning Curve - Average of Relevances\nTest Image - Flight "+testinfo, color="red")
	else:
  		fig.suptitle("Layerwise Learning Curve - Average of Relevances\nTest Image - Digit "+testinfo, color="red")

	#firstclass = "1"
	#secondclass = "6"

	inc = 0
	pfiltercount = []
	ffiltercount = []
	pffiltercount = []
	ofiltercount = []
	tpfiltercount = []
	tffiltercount = []
	tpffiltercount = []
	tofiltercount = []
	tpfilterRelsum = []
	tffilterRelsum = []
	tpffilterRelsum = []
	tofilterRelsum = []

	if DoFlights == True:
		numberOfLayers = 5
	else:
		numberOfLayers = 3

	tempnames = np.zeros(numberOfLayers)
	for layername, layernumber in layernumbers.iteritems():
		if layername == "fc6" or layername == "fc7":
			continue
		#print layername
		if DoFlights == True:
			pname = "temp/"+firstclass+"Avg"+"/"+"relScores_"+layernumbers[layername]+".txt"
			fname = "temp/"+secondclass+"Avg"+"/"+"relScores_"+layernumbers[layername]+".txt"
		else:
			pname = "temp/"+firstclass+"Avg"+"/"+firstclass+"avgRelScores_"+layernumbers[layername]+".txt"
			fname = "temp/"+secondclass+"Avg"+"/"+secondclass+"avgRelScores_"+layernumbers[layername]+".txt"
			
		tname = "relScores_"+layernumbers[layername]+".txt"
		pdata = np.genfromtxt(pname)
		fdata = np.genfromtxt(fname)
		tdata = np.genfromtxt(tname)

		pureP = []
		pureF = []
		purePF = []
		othersPF = []

		tempname = pname.split(".")[0]
		tempnames[inc] = tempname.split("_")[1]

		#print tempname.split("_")[1]
		inc = inc + 1

		for idx in range(len(pdata)):
			if pdata[idx] > 0 and fdata[idx] > 0:
				purePF.append(idx)
			if pdata[idx] > 0 and fdata[idx] == 0:
				pureP.append(idx)
			if fdata[idx] > 0 and pdata[idx] == 0:
				pureF.append(idx)
			if pdata[idx] < 0 and fdata[idx] < 0:
				othersPF.append(idx)
			if pdata[idx] < 0 and fdata[idx] > 0:
				pureF.append(idx)
			if fdata[idx] < 0 and pdata[idx] > 0:
				pureP.append(idx)
			if pdata[idx] == 0 and fdata[idx] == 0:
				othersPF.append(idx)

		tpureP = []
		tpureF = []
		tpurePF = []
		tothersPF = []

		for idx in range(len(tdata)):
			if idx in pureP:
				tpureP.append(idx)
			if idx in pureF:
				tpureF.append(idx)
			if idx in purePF:
				tpurePF.append(idx)
			if idx in othersPF:
				tothersPF.append(idx)
				
		#print pureP, pureF, purePF, othersPF
		#print len(pureP), len(pureF), len(purePF), len(othersPF)
		length = len(pureP)+len(pureF)+len(purePF)+len(othersPF)	
		pfiltercount.append(float(float(len(pureP))/float(length)))
		ffiltercount.append(float(float(len(pureF))/float(length)))
		pffiltercount.append(float(float(len(purePF))/float(length)))
		ofiltercount.append(float(float(len(othersPF))/float(length)))

		# Chandra, commented to display statistics instead of graphs
		#length = len(tpureP)+len(tpureF)+len(tpurePF)+len(tothersPF)	
		#tpfiltercount.append(float(float(len(tpureP))/float(length)))
		#tffiltercount.append(float(float(len(tpureF))/float(length)))
		#tpffiltercount.append(float(float(len(tpurePF))/float(length)))
		#tofiltercount.append(float(float(len(tothersPF))/float(length)))

		tpfiltercount.append(len(tpureP))
		tffiltercount.append(len(tpureF))
		tpffiltercount.append(len(tpurePF))
		tofiltercount.append(len(tothersPF))
	
		#print tpureP, tpureF, tpurePF, tothersPF
		#print len(tpureP), len(tpureF), len(tpurePF), len(tothersPF)

		length = len(tpureP)+len(tpureF)+len(tpurePF)+len(tothersPF)	
		ptotalRel = 0.0
		ftotalRel = 0.0
		pftotalRel = 0.0
		ototalRel = 0.0
		for i in pureP:
			ptotalRel += tdata[i]
		for i in pureF:
			ftotalRel += tdata[i]
		for i in purePF:
			pftotalRel += tdata[i]
		for i in othersPF:
			ototalRel += tdata[i]

		totalRel = abs(ptotalRel) + abs(ftotalRel) + abs(pftotalRel) + abs(ototalRel)
		tpffilterRelsum.append(float(ptotalRel/totalRel))
		tffilterRelsum.append(float(ftotalRel/totalRel))
		tpfilterRelsum.append(float(pftotalRel/totalRel))
		tofilterRelsum.append(float(ototalRel/totalRel))

	tempmindata = []
	tempmindata = (tempnames).argsort()[:numberOfLayers]
	#print tempmindata
	#print numberOfLayers
	tpfilterlist = []
	tffilterlist = []
	tpffilterlist = []
	tofilterlist = []
	for i in range(numberOfLayers):
		 #tpfilterlist.append(tpfiltercount[tempmindata[i]])
		 #tffilterlist.append(tffiltercount[tempmindata[i]])
		 #tpffilterlist.append(tpffiltercount[tempmindata[i]])
		 #tofilterlist.append(tofiltercount[tempmindata[i]])
		 tpfilterlist.append(tpfilterRelsum[tempmindata[i]])
		 tffilterlist.append(tffilterRelsum[tempmindata[i]])
		 tpffilterlist.append(tpffilterRelsum[tempmindata[i]])
		 tofilterlist.append(tofilterRelsum[tempmindata[i]])

	print "Average Relevances statistics:\n"	
	print tpfiltercount
	print tffiltercount
	print tpffiltercount
	print tofiltercount

	'''		
	x=np.arange(numberOfLayers)
	#x = x.reshape(numberOfLayers)
	if DoFlights == True:
		xlabels = ["", "Conv1", "", "Conv2", "", "Conv3", "", "Conv4", "", "Conv5"]
	else:
		xlabels = ["", "Conv1", "", "", "", "Conv2", "", "", "", "ip1"]
		
	ax.set_xticklabels(xlabels, color='b')
	#ylabels = ["", 0, 5, 10, 15, 20, 25, 30]
	#ax.set_yticklabels(ylabels, color='b')
	ax.set_xlabel("Neural Network Layers", color='r', fontsize=12)

	ax.set_ylabel("Relevances Across Layers", color='r', fontsize=12)
	ax.plot(x,tpfilterlist,c='g',marker="o",ls='--',label=firstclassString,fillstyle='none')
	ax.plot(x,tffilterlist,c='y',marker="x",ls='--',label=secondclassString,fillstyle='none')
	ax.plot(x,tpffilterlist,c='m',marker="o",ls='--',label=firstclassString+" and "+secondclassString,fillstyle='none')
	ax.plot(x,tofilterlist,c='c',marker="x",ls='--',label='others',fillstyle='none')

	plot.legend(loc=2)
	plot.draw()

	plot.savefig('RelAcrossLayers.png')

	'''

def featureMapcommonDist(misClassFlag, targetclass, totalRelevances, topRelCount, commonSeqCount, firstclass, secondclass, layername, flag, firstclassString, secondclassString):
	fig = plot.figure(figsize=(6,5))
	ax = fig.add_subplot(111)
	#fig, (ax,ax2) = plot.subplots(ncols=2, figsize=(12,5))

	testinfo = firstclassString
	if misClassFlag != "":
		testinfo = "Misclassified " + misClassFlag 

	if flag == True:
		titleinfo = "Average of Relevances"
	else:
		titleinfo = "Sequence of Relevances"
		
	if DoFlights == True:
		title = "Feature Map Distribution For "+layername+" - "+titleinfo+"\nTest Image - Flight "+testinfo
	else:
		title = "Feature Map Distribution For "+layername+" - "+titleinfo+"\nTest Image - Digit "+testinfo

	fig.suptitle(title, color="red")

	tname = "relScores_"+layernumbers[layername]+".txt"
	#print layername

	pcountlist = []
	fcountlist = []
	apcountlist = []
	afcountlist = []

	tdata = np.genfromtxt(tname)
	tmaxdata = []
	tmaxdata = (-tdata).argsort()[:topRelCount]

	if flag == True:
		if DoFlights == True:
			pname = "temp/"+firstclass+"Avg"+"/"+"relScores_"+layernumbers[layername]+".txt"
			fname = "temp/"+secondclass+"Avg"+"/"+"relScores_"+layernumbers[layername]+".txt"
		else:
			pname = "temp/"+firstclass+"Avg"+"/"+firstclass+"avgRelScores_"+layernumbers[layername]+".txt"
			fname = "temp/"+secondclass+"Avg"+"/"+secondclass+"avgRelScores_"+layernumbers[layername]+".txt"

		fdata = np.genfromtxt(fname)
		pdata = np.genfromtxt(pname)
		pmaxdata = [] 
		pmaxdata = (-pdata).argsort()[:topRelCount]
		fmaxdata = []
		fmaxdata = (-fdata).argsort()[:topRelCount]
	else:
			commonfilters = commonRelFilters(totalRelevances, firstclass, layername, topRelCount, commonSeqCount)
			pmaxdata = []
			for filt, tempcount in commonfilters:
				pmaxdata.append(filt)
			commonfilters = commonRelFilters(totalRelevances, secondclass, layername, topRelCount, commonSeqCount)
			fmaxdata = []
			for filt, tempcount in commonfilters:
				fmaxdata.append(filt)



	#print pmaxdata
	#print fmaxdata
	#print tmaxdata

	pcount = 0
	fcount = 0
	#mfilters = np.zeros(topRelCount)
	mfilters = np.empty(topRelCount)
	mfilters.fill(4)
	count = 0

	ptempdata = np.empty(topRelCount, dtype='object')
	for idx in tmaxdata:
		pele = np.abs(pmaxdata - idx).argmin()
		fele = np.abs(fmaxdata - idx).argmin()
		#print pmaxdata[pele], fmaxdata[fele], idx
		flag = 2
		if pmaxdata[pele] == idx:
			flag = 0
			mfilters[count] = 0
			pcount = pcount+1
			#ptempdata[count] ="F\n"+str(tmaxdata[count])+"\n"+str(round(tdata[idx], 2))
			ptempdata[count] =firstclassString+"\n"+str(tmaxdata[count])+"\n"+str(round(tdata[idx], 2))
			#print "marking P"

		if fmaxdata[fele] == idx:
			mfilters[count] = 1
			fcount = fcount+1
			if flag == 0:
				ptempdata[count]="BOTH\n"+str(tmaxdata[count])+"\n"+str(round(tdata[idx], 2))
				mfilters[count] = 2
				#print "marking PF"
			else:
				ptempdata[count]=secondclassString+"\n"+str(tmaxdata[count])+"\n"+str(round(tdata[idx], 2))
				#print "marking F"
			flag = 1

		if flag == 2:	
			mfilters[count] = 3
			ptempdata[count]="OTHERS\n"+str(tmaxdata[count])+"\n"+str(round(tdata[idx], 2))

		count = count+1
			#print "marking O"
	pcountlist.append(pcount) 
	fcountlist.append(fcount)

	midpoint = 2
	
	#print mfilters
	if topRelCount == 10:
		pdatachunks = [mfilters[x:x+2] for x in xrange(0, len(mfilters), 2)]
		ptempdatachunks = ptempdata.reshape(5,2)
	if topRelCount == 20:
		pdatachunks = [mfilters[x:x+4] for x in xrange(0, len(mfilters), 4)]
		ptempdatachunks = ptempdata.reshape(5,4)
	if topRelCount == 30:
		pdatachunks = [mfilters[x:x+6] for x in xrange(0, len(mfilters), 6)]
		ptempdatachunks = ptempdata.reshape(5,6)

	sns.heatmap(pdatachunks, annot=ptempdatachunks, xticklabels=False, yticklabels=False, ax=ax, cmap=['blue', 'red', 'green', '#abcabc'], fmt="", linewidths=0.30, cbar=False, center=midpoint)

	#plot.legend(loc=2)
	plot.draw()
	#plot.show()
	plot.savefig('HeatmapFor'+layername+'.png')

	return mfilters

def commonRelFilters(countImages, targetclass, inputlayername, count, commoncount):
	setRelV = []
	inc = 0
	commonSeq = []
	for layername, layernumber in layernumbers.iteritems():
		if layername == inputlayername and layername != "fc6" and layername != "fc7":
				#for i in range(countImages):
				count = 0
				for pname in glob.glob("relV/"+str(targetclass)+"/*/relScores_"+str(layernumbers[layername])+"*.txt"):
					#pname = "relV/"+str(targetclass)+"/"+str(i+1)+"/relScores_"+str(layernumbers[layername])+".txt"
					pdata = np.genfromtxt(pname)

					pmaxdata = (-pdata).argsort()[:count]
					setRelV = setRelV + pmaxdata.tolist()
					count = count+1
					if count == countImages:
						break;
				data = Counter(setRelV)
				#commonSeq.append(data.most_common(10))  # Returns all unique items and their counts
				return data.most_common(commoncount)
		inc = inc + 1

def commonNRelFilters(countImages, targetclass, inputlayername, count, commoncount):
	setRelV = []
	inc = 0
	commonSeq = []
	for layername, layernumber in layernumbers.iteritems():
		if layername == inputlayername:
				#for i in range(countImages):
				count = 0
				for pname in glob.glob("relV/"+str(targetclass)+"/*/relScores_"+str(layernumbers[layername])+"*.txt"):
					#pname = "relV/"+str(targetclass)+"/"+str(i+1)+"/relScores_"+str(layernumbers[layername])+".txt"
					pdata = np.genfromtxt(pname)

					pmaxdata = pdata.argsort()[:count]
					setRelV = setRelV + pmaxdata.tolist()
					count = count+1
					if count == countImages:
						break;

				data = Counter(setRelV)
				#commonSeq.append(data.most_common(10))  # Returns all unique items and their counts
				return data.most_common(commoncount)
		inc = inc + 1

def RelAcrossLayersCommonDist(misClassFlag, targetclass, totalRelevances, topRelCount, commonSeqCount, firstclass, secondclass, firstclassString, secondclassString, classname):
	fig = plot.figure(figsize=(10,6))
	ax = fig.add_subplot(111)

	testinfo = firstclassString
	if misClassFlag != "":
		testinfo = "Misclassified " + misClassFlag

	if DoFlights == True:
		fig.suptitle("Layerwise Learning Curve - Sequence of Relevances\nTest Image - Flight "+testinfo, color="red")
	else:
		fig.suptitle("Layerwise Learning Curve - Sequence of Relevances\nTest Image - Digit "+testinfo, color="red")


	inc = 0
	pfiltercount = []
	ffiltercount = []
	pffiltercount = []
	ofiltercount = []
	nonfiltercount = []
	tpfiltercount = []
	tffiltercount = []
	tpffiltercount = []
	tofiltercount = []
	tnonfiltercount = []
	tpfilterRelsum = []
	tffilterRelsum = []
	tpffilterRelsum = []
	tofilterRelsum = []

	if DoFlights == True:
		numberOfLayers = 5
	else:
		numberOfLayers = 3

	tempnames = np.zeros(numberOfLayers)
	pCount = 0
	potentialmisClassFlag = False
	for layername, layernumber in layernumbers.iteritems():
		print layername
		if layername == "fc6" or layername == "fc7":
			continue
		tname = "relScores_"+layernumbers[layername]+".txt"

		tdata = np.genfromtxt(tname)
		# Chandra, to obtain statistics instead of graph
		tmaxdata = (-tdata).argsort()[:topRelCount]
		ntmaxdata = (tdata).argsort()[:topRelCount]

		pmaxdata = []
		commonfilters = commonRelFilters(totalRelevances, firstclass, layername, topRelCount, commonSeqCount)
		for filt, tempcount in commonfilters:
			pmaxdata.append(filt)
		fmaxdata = []
		commonfilters = commonRelFilters(totalRelevances, secondclass, layername, topRelCount, commonSeqCount)
		for filt, tempcount in commonfilters:
			fmaxdata.append(filt)
		#npmaxdata = []
		#commonfilters = commonNRelFilters(totalRelevances, firstclass, layername, topRelCount, commonSeqCount)
		#for filt, tempcount in commonfilters:
		#	npmaxdata.append(filt)
		#nfmaxdata = []
		#commonfilters = commonNRelFilters(totalRelevances, secondclass, layername, topRelCount, commonSeqCount)
		#for filt, tempcount in commonfilters:
		#	nfmaxdata.append(filt)

		npmaxdata = pmaxdata
		nfmaxdata = fmaxdata

		pureP = []
		pureF = []
		purePF = []
		othersPF = []
		nonPF = []

		tempname = tname.split(".")[0]
		tempnames[inc] = tempname.split("_")[1]
		inc = inc + 1
		'''
		print "tmax: ", tmaxdata
		print "ntmax: ", ntmaxdata
		print "pmax: ", pmaxdata
		print "npmax: ", npmaxdata
		print "fmax: ", fmaxdata
		print "nfmax: ", nfmaxdata
		'''

		#for idx in range(len(tdata)):
		# Chandra, to obtain statistics instead of graphs

		resultsfilename = open("results.txt", "a")
		resultsPfilename = open("resultsP.txt", "a")
		resultsFfilename = open("resultsF.txt", "a")
		resultsPFfilename = open("resultsPF.txt", "a")
		resultsOfilename = open("resultsO.txt", "a")
		resultsNonfilename = open("resultsNon.txt", "a")
		for idx in range(len(tmaxdata)):
			pele = 0
			fele = 0
			npele = 0
			nfele = 0
			try:
				#pele = np.abs(pmaxdata - temp).argmin()
				# Chandra, G2S
				#pele = pmaxdata.index(idx)
				pele = pmaxdata.index(tmaxdata[idx])
			except ValueError:
				pele = 0
				pass	
			try:
				#fele = fmaxdata.index(idx)
				# Chandra, G2S
				fele = fmaxdata.index(tmaxdata[idx])
			except ValueError:
				fele = 0
				pass	
			try:
				#npele = npmaxdata.index(idx)
				# Chandra, G2S
				npele = npmaxdata.index(ntmaxdata[idx])
			except ValueError:
				npele = 0
				pass	
			try:
				#nfele = nfmaxdata.index(idx)
				# Chandra, G2S
				nfele = nfmaxdata.index(ntmaxdata[idx])
			except ValueError:
				nfele = 0
				pass	
			#print pele, fele, npele, fele, idx
			# Chandra G2S
			#if tmaxdata[idx] > 0 and pele != 0 and fele != 0:
			if pele != 0 and fele != 0:
				purePF.append(idx)
			# Chandra G2S
			#if tmaxdata[idx] > 0 and pele != 0 and fele == 0:
			if pele != 0 and fele == 0:
				pureP.append(idx)
			# Chandra G2S
			#if tmaxdata[idx] > 0 and pele == 0 and fele != 0:
			if pele == 0 and fele != 0:
				pureF.append(idx)
			# Chandra G2S
			#if tmaxdata[idx] > 0 and pele == 0 and fele == 0:
			if pele == 0 and fele == 0:
				othersPF.append(idx)
			# Chandra G2S
			#if tmaxdata[idx] == 0:
			#	othersPF.append(idx)
			# Chandra G2S
			#if ntmaxdata[idx] > 0 and npele != 0 and nfele != 0:
			'''
			if npele != 0 and nfele != 0:
				purePF.append(idx)
			# Chandra G2S
			#if ntmaxdata[idx] > 0 and npele != 0 and nfele == 0:
			if npele != 0 and nfele == 0:
				pureF.append(idx)
			# Chandra G2S
			#if ntmaxdata[idx] > 0 and npele == 0 and nfele != 0:
			if npele == 0 and nfele != 0:
				pureP.append(idx)
			# Chandra G2S
			#if ntmaxdata[idx] > 0 and pele == 0 and fele == 0:
			if npele == 0 and nfele == 0:
				othersPF.append(idx)
			'''

			if pele == 0 and fele == 0 and npele == 0 and nfele == 0:
				nonPF.append(idx)

		#print "P, F, PF, O length: ", len(pureP), len(pureF), len(purePF), len(othersPF), len(nonPF), count
		#print pureP
		#print pureF
		#print purePF
		#print othersPF
		#print nonPF

		tpureP = []
		tpureF = []
		tpurePF = []
		tothersPF = []
		tnonPF = []

		for idx in range(len(tmaxdata)):
			if idx in pureP:
				tpureP.append(idx)
			if idx in pureF:
				tpureF.append(idx)
			if idx in purePF:
				tpurePF.append(idx)
			if idx in othersPF:
				tothersPF.append(idx)
			if idx in nonPF:
				tnonPF.append(idx)
				
		length = len(tpureP)+len(tpureF)+len(tpurePF)+len(tothersPF)	

		# Chandra, commented to print the statistics instead of graphs

		#print "T P, F, PF, O length: ", len(tpureP), len(tpureF), len(tpurePF), len(tothersPF), len(tnonPF)
		#tpfiltercount.append(float(float(len(tpureP))/float(length)))
		#tffiltercount.append(float(float(len(tpureF))/float(length)))
		#tpffiltercount.append(float(float(len(tpurePF))/float(length)))
		#tofiltercount.append(float(float(len(tothersPF))/float(length)))
		tpfiltercount.append(len(pureP))
		tffiltercount.append(len(pureF))
		tpffiltercount.append(len(purePF))
		tofiltercount.append(len(othersPF))
		tnonfiltercount.append(len(nonPF))

		print classname
		if classname == "0":
			print "class name is 0"
			if (len(pureP) - len(pureF)) <= 4:
					pCount = pCount + 1
					potentialmisClassFlag = True
		if classname == "1":
			print "class name is 1"
			if (len(pureP) - len(pureF)) <= 4:
					pCount = pCount + 1
					potentialmisClassFlag = True

		#print tpureP, tpureF, tpurePF, tothersPF
		#print len(tpureP), len(tpureF), len(tpurePF), len(tothersPF)


		length = len(tpureP)+len(tpureF)+len(tpurePF)+len(tothersPF)	
		ptotalRel = 0.0
		ftotalRel = 0.0
		pftotalRel = 0.0
		ototalRel = 0.0
		for i in pureP:
			ptotalRel += tdata[i]
		for i in pureF:
			ftotalRel += tdata[i]
		for i in purePF:
			pftotalRel += tdata[i]
		for i in othersPF:
			ototalRel += tdata[i]

		totalRel = abs(ptotalRel) + abs(ftotalRel) + abs(pftotalRel) + abs(ototalRel)
		#print ptotalRel
		#print ftotalRel 
		#print pftotalRel 
		#print ototalRel
		#print totalRel 
		#print ptotalRel/totalRel
		#print ftotalRel/totalRel
		#print pftotalRel/totalRel
		#print ototalRel/totalRel
		#print totalRel/totalRel
		tpffilterRelsum.append(float(ptotalRel/totalRel))
		tffilterRelsum.append(float(ftotalRel/totalRel))
		tpfilterRelsum.append(float(pftotalRel/totalRel))
		tofilterRelsum.append(float(ototalRel/totalRel))

	tempmindata = []
	tempmindata = (tempnames).argsort()[:numberOfLayers]
	#print tempmindata
	#print numberOfLayers
	tpfilterlist = []
	tffilterlist = []
	tpffilterlist = []
	tofilterlist = []
	for i in range(numberOfLayers):
		 #tpfilterlist.append(tpfiltercount[tempmindata[i]])
		 #tffilterlist.append(tffiltercount[tempmindata[i]])
		 #tpffilterlist.append(tpffiltercount[tempmindata[i]])
		 #tofilterlist.append(tofiltercount[tempmindata[i]])
		 tpfilterlist.append(tpfilterRelsum[tempmindata[i]])
		 tffilterlist.append(tffilterRelsum[tempmindata[i]])
		 tpffilterlist.append(tpffilterRelsum[tempmindata[i]])
		 tofilterlist.append(tofilterRelsum[tempmindata[i]])
	
	print "Sequence Relevances statistics:\n"	
	if potentialmisClassFlag == True:
		resultsfilename.write("True "+str(pCount))
		resultsPfilename.write("\n"+str(tpfiltercount)+"\n")
		resultsFfilename.write(str(tffiltercount)+"\n")
		resultsPFfilename.write(str(tpffiltercount)+"\n")
		resultsOfilename.write(str(tofiltercount)+"\n")
		resultsNonfilename.write(str(tnonfiltercount)+"\n")
	resultsfilename.close()
	resultsPfilename.close()
	resultsFfilename.close()
	resultsPFfilename.close()
	resultsOfilename.close()
	resultsNonfilename.close()
	#print tpfiltercount
	#print tffiltercount
	#print tpffiltercount
	#print tofiltercount
	#print tnonfiltercount

	'''
	x=np.arange(numberOfLayers)
	#x = x.reshape(numberOfLayers)
	if DoFlights == True:
		xlabels = ["", "Conv1", "", "Conv2", "", "Conv3", "", "Conv4", "", "Conv5"]
	else:
		xlabels = ["", "Conv1", "", "", "", "Conv2", "", "", "", "ip1"]
		
	ax.set_xticklabels(xlabels, color='b')
	#ylabels = ["", 0, 5, 10, 15, 20, 25, 30]
	#ax.set_yticklabels(ylabels, color='b')
	ax.set_xlabel("Neural Network Layers", color='r', fontsize=12)

	#ax.plot(x,pcountdata,c='b',marker="o",ls='-',label='Passenger',fillstyle='none', linewidth=3)
	#ax.plot(x,fcountdata,c='r',marker="o",ls='-',label='Fighter',fillstyle='none', linewidth=3)
	ax.set_ylabel("Relevances Across Layers", color='r', fontsize=12)
	ax.plot(x,tpfilterlist,c='g',marker="o",ls='--',label=firstclassString,fillstyle='none')
	ax.plot(x,tffilterlist,c='y',marker="o",ls='--',label=secondclassString,fillstyle='none')
	ax.plot(x,tpffilterlist,c='m',marker="o",ls='--',label=firstclassString+" and "+secondclassString,fillstyle='none')
	ax.plot(x,tofilterlist,c='c',marker="o",ls='--',label='others',fillstyle='none')

	plot.legend(loc=2)
	plot.draw()

	plot.savefig('RelAcrossLayers.png')
	'''

class plotcommonCurve():
  	def __init__(self, totalRelevances, topRelCount, commonSeqCount, firstclass, secondclass, layername, flag):

		if flag != True:
			self.tname = "relScores_"+layernumbers[layername]+".txt"
			self.tdata = np.genfromtxt(self.tname)

			self.pmaxdata = [] 
			self.commonfilters = commonRelFilters(totalRelevances, firstclass, layername, topRelCount, commonSeqCount)
			for filt, tempcount in self.commonfilters:
				self.pmaxdata.append(filt)
			self.fmaxdata = []
			self.commonfilters = commonRelFilters(totalRelevances, secondclass, layername, topRelCount, commonSeqCount)
			for filt, tempcount in self.commonfilters:
				self.fmaxdata.append(filt)
			self.npmaxdata = [] 
			self.commonfilters = commonNRelFilters(totalRelevances, firstclass, layername, topRelCount, commonSeqCount)
			for filt, tempcount in self.commonfilters:
				self.npmaxdata.append(filt)
			self.nfmaxdata = []
			self.commonfilters = commonNRelFilters(totalRelevances, secondclass, layername, topRelCount, commonSeqCount)
			for filt, tempcount in self.commonfilters:
				self.nfmaxdata.append(filt)
			self.tmaxdata = []
			self.tmaxdata = (-self.tdata).argsort()[:topRelCount]

			self.xtdata = np.zeros(topRelCount, dtype=float)
			for i in range(topRelCount):
				self.xtdata[i] = self.tdata[self.tmaxdata[i]]

			self.ntmaxdata = []
			self.ntmaxdata = (self.tdata).argsort()[:topRelCount]

			self.nxtdata = np.zeros(topRelCount, dtype=float)
			for i in range(topRelCount):
				self.nxtdata[i] = self.tdata[self.ntmaxdata[i]]
			self.mfilters = np.zeros(topRelCount)
			self.nmfilters = np.zeros(topRelCount)

			self.plotCounters = \
			{
				"FirstClass" : {"poscount" : 0, "negcount": 0, "rposdata": None, "rnegdata":None, "rxposdata": None, "rxnegdata":None, "annotrposdata": None, "annotrnegdata": None}, 
				"SecondClass" : {"poscount" : 0, "negcount": 0, "rposdata": None, "rnegdata":None, "rxposdata": None, "rxnegdata":None, "annotrposdata": None, "annotrnegdata": None}, 
				"FirstAndSecond" : {"poscount":0, "negcount": 0, "rposdata": None, "rnegdata": None, "rxposdata": None, "rxnegdata": None, "annotrposdata": None, "annotrnegdata": None},
				"Others" : {"poscount":0, "negcount": 0, "rposdata": None, "rnegdata":None, "rxposdata": None, "rxnegdata":None, "annotrposdata": None, "annotrnegdata": None} 
			}
		else:
			self.tname = "relScores_"+layernumbers[layername]+".txt"
			self.tdata = np.genfromtxt(self.tname)
			if DoFlights == True:
				self.pname = "temp/"+firstclass+"Avg"+"/"+"relScores_"+layernumbers[layername]+".txt"
				self.fname = "temp/"+secondclass+"Avg"+"/"+"relScores_"+layernumbers[layername]+".txt"
			else:
				self.pname = "temp/"+firstclass+"Avg"+"/"+firstclass+"avgRelScores_"+layernumbers[layername]+".txt"
				self.fname = "temp/"+secondclass+"Avg"+"/"+secondclass+"avgRelScores_"+layernumbers[layername]+".txt"

			self.pdata = np.genfromtxt(self.pname)
			self.fdata = np.genfromtxt(self.fname)

			self.pmaxdata = [] 
			self.pmaxdata = (-self.pdata).argsort()[:topRelCount]
			self.fmaxdata = []
			self.fmaxdata = (-self.fdata).argsort()[:topRelCount]
			self.tmaxdata = []
			self.tmaxdata = (-self.tdata).argsort()[:topRelCount]

			self.npmaxdata = [] 
			self.npmaxdata = (self.pdata).argsort()[:topRelCount]
			self.nfmaxdata = []
			self.nfmaxdata = (self.fdata).argsort()[:topRelCount]

			self.tmaxdata = []
			self.tmaxdata = (-self.tdata).argsort()[:topRelCount]

			self.xtdata = np.zeros(topRelCount, dtype=float)
			for i in range(topRelCount):
				self.xtdata[i] = self.tdata[self.tmaxdata[i]]

			self.ntmaxdata = []
			self.ntmaxdata = (self.tdata).argsort()[:topRelCount]

			self.nxtdata = np.zeros(topRelCount, dtype=float)
			for i in range(topRelCount):
				self.nxtdata[i] = self.tdata[self.ntmaxdata[i]]
			self.mfilters = np.zeros(topRelCount)
			self.nmfilters = np.zeros(topRelCount)

			self.plotCounters = \
			{
				"FirstClass" : {"poscount" : 0, "negcount": 0, "rposdata": None, "rnegdata":None, "rxposdata": None, "rxnegdata":None, "annotrposdata": None, "annotrnegdata": None}, 
				"SecondClass" : {"poscount" : 0, "negcount": 0, "rposdata": None, "rnegdata":None, "rxposdata": None, "rxnegdata":None, "annotrposdata": None, "annotrnegdata": None}, 
				"FirstAndSecond" : {"poscount":0, "negcount": 0, "rposdata": None, "rnegdata": None, "rxposdata": None, "rxnegdata": None, "annotrposdata": None, "annotrnegdata": None},
				"Others" : {"poscount":0, "negcount": 0, "rposdata": None, "rnegdata":None, "rxposdata": None, "rxnegdata":None, "annotrposdata": None, "annotrnegdata": None} 
			}


	def setParamsForPlot(self):
			count = 0
			for idx in self.tmaxdata:
				pele = np.abs(self.pmaxdata - idx).argmin()
				fele = np.abs(self.fmaxdata - idx).argmin()
				flag = 2
				if self.pmaxdata[pele] == idx and self.fmaxdata[fele] != idx:
					flag = 0
					self.mfilters[count] = 0
					self.plotCounters["FirstClass"]["poscount"] += 1

				if flag == 0 and self.fmaxdata[fele] == idx and self.pmaxdata[pele] == idx:
					self.mfilters[count] = 2
					self.plotCounters["FirstAndSecond"]["poscount"] += 1
					flag = 1

				if self.fmaxdata[fele] == idx and self.pmaxdata[pele] != idx:
					self.mfilters[count] = 1
					self.plotCounters["SecondClass"]["poscount"] += 1
					flag = 1

				if flag == 2:	
					self.mfilters[count] = 3
					self.plotCounters["Others"]["poscount"] += 1
				count = count+1

			count = 0
			for idx in self.ntmaxdata:
				npele = np.abs(self.npmaxdata - idx).argmin()
				nfele = np.abs(self.nfmaxdata - idx).argmin()
				flag = 2
				if self.npmaxdata[pele] != idx and self.nfmaxdata[nfele] == idx:
					flag = 0
					self.nmfilters[count] = 0
					self.plotCounters["FirstClass"]["negcount"] += 1

				if flag == 0 and self.nfmaxdata[fele] == idx and self.npmaxdata[pele] == idx:
					self.nmfilters[count] = 2
					self.plotCounters["FirstAndSecond"]["negcount"] += 1
					flag = 1

				if self.nfmaxdata[fele] != idx and self.npmaxdata[npele] == idx:
					self.nmfilters[count] = 1
					self.plotCounters["SecondClass"]["negcount"] += 1
					flag = 1

				if flag == 2:	
					self.nmfilters[count] = 3
					self.plotCounters["Others"]["negcount"] += 1
				count = count+1

			xtmax = self.xtdata.max()
			nxtmax = self.nxtdata.min()

			for classtype in self.plotCounters:
				self.plotCounters[classtype]["rposdata"] = np.zeros(self.plotCounters[classtype]["poscount"], dtype=float)
				self.plotCounters[classtype]["rnegdata"] = np.zeros(self.plotCounters[classtype]["negcount"], dtype=float)
				self.plotCounters[classtype]["rxposdata"] = np.zeros(self.plotCounters[classtype]["poscount"], dtype=float)
				self.plotCounters[classtype]["rxnegdata"] = np.zeros(self.plotCounters[classtype]["negcount"], dtype=float)
				self.plotCounters[classtype]["annotrposdata"] = np.zeros(self.plotCounters[classtype]["poscount"], dtype=object)
				self.plotCounters[classtype]["annotrnegdata"] = np.zeros(self.plotCounters[classtype]["negcount"], dtype=object)
	
			for classtype in self.plotCounters:
				self.plotCounters[classtype]["poscount"] = 0
				self.plotCounters[classtype]["negcount"] = 0

			for i in range(len(self.mfilters)):
				if self.mfilters[i] == 0:
					pcount = self.plotCounters["FirstClass"]["poscount"]
					self.plotCounters["FirstClass"]["rposdata"][pcount] = self.xtdata[i]
					self.plotCounters["FirstClass"]["annotrposdata"][pcount] = \
						"("+str(pcount)+", "+str(round(self.xtdata[i],2))+")"
					self.plotCounters["FirstClass"]["rxposdata"][pcount] = i
					self.plotCounters["FirstClass"]["poscount"] += 1
				if self.nmfilters[i] == 0:
					npcount = self.plotCounters["FirstClass"]["negcount"]
					self.plotCounters["FirstClass"]["rnegdata"][npcount] = self.nxtdata[i]
					self.plotCounters["FirstClass"]["annotrnegdata"][npcount] = \
						"("+str(npcount)+", "+str(round(self.nxtdata[i],2))+")"
					self.plotCounters["FirstClass"]["rxnegdata"][npcount] = i
					self.plotCounters["FirstClass"]["negcount"] += 1
				if self.mfilters[i] == 1:
					fcount = self.plotCounters["SecondClass"]["poscount"]
					self.plotCounters["SecondClass"]["rposdata"][fcount] = self.xtdata[i]
					self.plotCounters["SecondClass"]["annotrposdata"][fcount] = \
						"("+str(fcount)+", "+str(round(self.xtdata[i],2))+")"
					self.plotCounters["SecondClass"]["rxposdata"][fcount] = i
					self.plotCounters["SecondClass"]["poscount"] += 1
				if self.nmfilters[i] == 1:
					nfcount = self.plotCounters["SecondClass"]["negcount"]
					self.plotCounters["SecondClass"]["rnegdata"][nfcount] = self.nxtdata[i]
					self.plotCounters["SecondClass"]["annotrnegdata"][nfcount] = \
						"("+str(nfcount)+", "+str(round(self.nxtdata[i],2))+")"
					self.plotCounters["SecondClass"]["rxnegdata"][nfcount] = i
					self.plotCounters["SecondClass"]["negcount"] += 1
				if self.mfilters[i] == 2:
					pfcount = self.plotCounters["FirstAndSecond"]["poscount"]
					self.plotCounters["FirstAndSecond"]["rposdata"][pfcount] = self.xtdata[i]
					self.plotCounters["FirstAndSecond"]["annotrposdata"][pfcount] = \
						"("+str(pfcount)+", "+str(round(self.xtdata[i],2))+")"
					self.plotCounters["FirstAndSecond"]["rxposdata"][pfcount] = i
					self.plotCounters["FirstAndSecond"]["poscount"] += 1
				if self.nmfilters[i] == 2:
					npfcount = self.plotCounters["FirstAndSecond"]["negcount"]
					self.plotCounters["FirstAndSecond"]["rnegdata"][npfcount] = self.nxtdata[i]
					self.plotCounters["FirstAndSecond"]["annotrnegdata"][npfcount] = \
						"("+str(npfcount)+", "+str(round(self.nxtdata[i],2))+")"
					self.plotCounters["FirstAndSecond"]["rxnegdata"][npfcount] = i
					self.plotCounters["FirstAndSecond"]["negcount"] += 1
				if self.mfilters[i] == 3:
					ocount = self.plotCounters["Others"]["poscount"]
					self.plotCounters["Others"]["rposdata"][ocount] = self.xtdata[i]
					self.plotCounters["Others"]["annotrposdata"][ocount] = \
						"("+str(ocount)+", "+str(round(self.xtdata[i],2))+")"
					self.plotCounters["Others"]["rxposdata"][ocount] = i
					self.plotCounters["Others"]["poscount"] += 1
				if self.nmfilters[i] == 3:
					nocount = self.plotCounters["Others"]["negcount"]
					self.plotCounters["Others"]["rnegdata"][nocount] = self.nxtdata[i]
					self.plotCounters["Others"]["annotrnegdata"][nocount] = \
						"("+str(nocount)+", "+str(round(self.nxtdata[i],2))+")"
					self.plotCounters["Others"]["rxnegdata"][nocount] = i
					self.plotCounters["Others"]["negcount"] += 1

def	relcommonCurve(misClassFlag, targetclass, totalRelevances, topRelCount, commonSeqCount, firstclass, secondclass, layername, flag, firstclassString, secondclassString):
	fig = plot.figure(figsize=(10,6))
	ax = fig.add_subplot(111)

	testinfo = firstclassString
	if misClassFlag != "":
		testinfo = "Misclassified " + misClassFlag 

	if DoFlights == True:
		title = "Relevance Curve For " + layername+"\nTest Image - Flight "+testinfo
	else:
		title = "Relevance Curve For " + layername+"\nTest Image - Digit "+testinfo
	fig.suptitle(title, color="red")

	plotRelSeqCurve = plotcommonCurve(totalRelevances, topRelCount, commonSeqCount, firstclass, secondclass, layername, flag)

	plotRelSeqCurve.setParamsForPlot()

	x=np.arange(topRelCount)
	x = x.reshape(topRelCount)
	ax.set_xlabel("High Relevance Filters", color='r', fontsize=12)
	ax.set_ylabel("Relevances", color='r', fontsize=12)

	ax.plot(x,plotRelSeqCurve.xtdata,c='b',ls='--',label="Relevances", fillstyle='none')
	ax.plot(x,plotRelSeqCurve.nxtdata,c='b',ls='--',fillstyle='none')
	pcount = plotRelSeqCurve.plotCounters["FirstClass"]["poscount"]
	ax.scatter(plotRelSeqCurve.plotCounters["FirstClass"]["rxposdata"][:pcount],
			plotRelSeqCurve.plotCounters["FirstClass"]["rposdata"][:pcount],
			c='m', marker="o", label=firstclassString)
	fcount = plotRelSeqCurve.plotCounters["SecondClass"]["poscount"]
	ax.scatter(plotRelSeqCurve.plotCounters["SecondClass"]["rxposdata"][:fcount],
			plotRelSeqCurve.plotCounters["SecondClass"]["rposdata"][:fcount],
			c='r',marker="x", label=secondclassString)
	npcount = plotRelSeqCurve.plotCounters["FirstClass"]["negcount"]
	ax.scatter(plotRelSeqCurve.plotCounters["FirstClass"]["rxnegdata"][:npcount],
			plotRelSeqCurve.plotCounters["FirstClass"]["rnegdata"][:npcount],
			c='m', marker="o")
	nfcount = plotRelSeqCurve.plotCounters["SecondClass"]["negcount"]
	ax.scatter(plotRelSeqCurve.plotCounters["SecondClass"]["rxnegdata"][:nfcount],
			plotRelSeqCurve.plotCounters["SecondClass"]["rnegdata"][:nfcount],
			c='r',marker="x")

	plot.legend(loc='upper right')
	#plot.draw()
	
	plot.savefig('HighRelFiltersFor'+layername+'.png')
