from tensorflow import keras
from tensorflow.keras import optimizers
from tensorflow.keras import backend as K
from matplotlib import pyplot as plt
import numpy as np
import scipy.io
from neuralgregnet.training import load_model
from scipy.optimize import curve_fit
#maybe implement cma instead of scipy curve_fit

aeM= load_model("networks/newWT2")
aeM.compile(optimizer=optimizers.Adam(lr=0.01),loss="mse")

genes=['Gt', 'Kni', 'Kr', 'Hb']
cutNaN=35 # to avoid all the nan values
offNaN=965
positions=np.linspace(0,100,1000)
positions=positions[cutNaN:offNaN]


def find_nearest(array, value):
	"""
	find object in array closest to value and return its index
	INPUT
	array: in which we look for value
	value: object we are looking for
	OUTPUT
	index: index for position of object in array
	"""
	array = np.asarray(array)
	idx = (np.abs(array - value)).argmin()
	return idx

cut=find_nearest(positions,0)
off=find_nearest(positions,101)

def loadGG(f="DataPetkova/Data/Gap/gap_data_raw_dorsal_wt.mat",path="DataPetkova/Data/Gap/",positions=positions,ageSort=True,age1=40,age2=44,cutPos=False,cut=cut,off=off, genotype=1, smooth=False):
	"""
	load gap genes
	INPUT
	f:file .mat that we want to upload
	path: path where the file is
	positions: positions array along AP
	ageSort: True if want to sort that data from earliest to latest and only keep the data in time window age1-age2
	age1,age2: age window you want to keep (not index , really age in minute)
	cutPos: True if want to only keep data between cut:off
	cut,off: position index you want to keep
	genotype: 1==WT, 2== mutant
	smooth: True if want to smooth the data
	OUTPUT
	SortData: array shape(nb of embryos, 4(gap genes),nb of positions)
	sortAge: list of age lenght nb of embryos. corrresponds to the ages of each embryo in SortData
	"""
	
	mat=scipy.io.loadmat(path+f,squeeze_me=True)
	#iterate through embryos
	listOfAges=[]
	genesData=[]
	for i in range(mat['data'].shape[0]):
		if mat['data'][i]['orient']==1  and mat['data'][i]['genotype']==genotype: #Dubuis et.al (2014) use asymmetrically oriented embryos.
	                                                                        #genotype ==1 is wild type, genotype ==2 is mutant
			tmpAge=mat['data'][i]['age']
			listOfAges.append(tmpAge)
			tmpKni=mat['data'][i]['Kni']
			tmpKr=mat['data'][i]['Kr']
			tmpGt=mat['data'][i]['Gt']
			tmpHb=mat['data'][i]['Hb']
			tmpGenes=np.stack((tmpGt,tmpKni,tmpKr,tmpHb),axis=0)
			genesData.append(tmpGenes)


	genesData=np.array(genesData)
	genesData=genesData[:,:,cutNaN:offNaN]      #cut to avoid nan

	#normalize
	for g in range(genesData.shape[1]):
	    #mutantsData[:,g,:]=((genesData[:,g,:]- np.amin(genesData[:,g,:]))/ (np.amax(genesData[:,g,:])-
	                            #np.amin(genesData[:,g,:])))   # old norm
		genesData[:,g,:] = (genesData[:,g,:].T-np.nanmin(genesData[:,g,:],axis=1)).T
		genesData[:,g,:] /= np.nanmax(genesData[:,g,:]) # new norm by adrien to have min at 0
	         
	#Sort by age
	SortData=genesData
	sortAge=None
	if ageSort:
		listOfAges=np.array(listOfAges)
		#sort age
		noSortAge=listOfAges.copy()
		sortAge=listOfAges
		sortAge=sorted(sortAge)
		#sort data
		SortData=[]
		for a in sortAge:
			for i in range(len(noSortAge)):     
				if a==noSortAge[i]:
					SortData.append(genesData[i,:,:])
					noSortAge[i]=None
		                   
		SortData=np.array(SortData)
		#cut for right age window
		age1I=find_nearest(sortAge,age1)
		age2I=find_nearest(sortAge,age2)
		SortData=SortData[age1I:age2I,:,:]
		sortAge=sortAge[age1I:age2I]
	
	#cut for right position range
	if cutPos:
		SortData=SortData[:,:,cut:off]

	#smooth the data
	SortData=np.array(SortData)	
	print(SortData.shape)
	if smooth and ageSort:
		#gaussian smoothing from Petkova 
		timeV=np.linspace(min(sortAge),max(sortAge),SortData.shape[0])
		Tfit=2.5
		smoothData=np.zeros(SortData.shape)
		for i in range(SortData.shape[0]):
			for j in range(SortData.shape[2]):
				for g in range(SortData.shape[1]):
					smoothData[i,g,j]=(np.nansum(SortData[:,g,j] * np.exp(-(sortAge-timeV[i])**2/(2*Tfit**2)))/
										np.sum(np.exp(-(sortAge-timeV[i])**2 / (2*Tfit**2))))
		SortData=smoothData

	if smooth and (not ageSort):
		print("Must sort age to smooth")

	return SortData,sortAge


def makeManifold(SortData,cut=cut,off=off,linear=True,aeM=aeM):
	"""
	Transfer from gap gene space to hidden nodes space
	INPUT
	SortData: From loadGG(). array of gap genes data shape (nb of embryos, 4(gap genes),nb of positions)
	cut,off: position index you want to keep
	linear:Choose if hidden nodes are calculated from linear manifold(True) or autoencoder manifold(false)
	aeM: loaded autoencoder model
	OUTPUT
	h1s,h2s: all points from SortData transformed in hidden nodes. shape (nb of embryos, nb of positions -1)
	avg1,avg2: average over embryos of h1s,h2s. Creates the average manifold. length (nb of embryos)
	"""
	plt.figure(figsize=(5,5))
	#create manifold linearly
	if linear:
		plt.xlim(-1,1)
		plt.ylim(-1,1)
		h1s=SortData[:,2,cut:off]-SortData[:,0,cut:off]
		h2s=SortData[:,3,cut:off]-SortData[:,1,cut:off]
		plt.plot(h1s,h2s, '.',markersize=2, c='pink')
		avg1=[]
		avg2=[]
		for i in range(off-cut):
			l1=h1s[:,i]
			l2=h2s[:,i]
			avg1.append(np.mean(l1))
			avg2.append(np.mean(l2))  

	# or from autoencoder
	else:
		#create manifold from autoencoder
		plt.xlim(0,1)
		plt.ylim(0,1)
		#cut for the right position range
		get_middle_layer_output = K.function([aeM.layers[0].input],[aeM.layers[1].output])
		h1s=[]
		h2s=[]
		#loop through all embryos/ages
		for i in range(SortData.shape[0]):
			#print(i)
			h1=[]
			h2=[]
			#loop trough positions
			for j in range(cut,off):
				inp=SortData[i,:,j].reshape((1,4))
				layer_output = get_middle_layer_output([inp])[0][0]
				h1.append(layer_output[0])
				h2.append(layer_output[1])
			plt.plot(h1,h2)#, '.', markersize=1)
			h1s.append(h1)
			h2s.append(h2)
		    
		h1s=np.array(h1s)
		h2s=np.array(h2s)
		avg1=[]
		avg2=[]
		#calculate average
		for i in range(off-cut):
			l1=h1s[:,i]
			l2=h2s[:,i]
			avg1.append(np.mean(l1))
			avg2.append(np.mean(l2)) 
	plt.xlabel(r"$h_1$")
	plt.ylabel(r"$h_2$")	     
	plt.plot(avg1,avg2, color="black",label="average")
	plt.legend()
	return h1s,h2s,avg1,avg2

def gauss(x,x0,sig,a):#for fit
	return a*np.exp(-(x-x0)**2/(2*sig**2))

def makeAvgMap(mutantData,Pxh,Zfunc,linear=True,meanErr=True, medianErr=True,aeM=aeM,positions=positions,cut=cut,off=off):
	"""
	Plot average bayesian map and calculate positional error.
	INPUT
	mutantData: Data of map (put wt data, mutant data or wt at other time window here)
	Pxh: bayesian probability distribution. Should always be constructed from WT 40-44 min.
			If building mutant map, choose the WT data of mutant file (genotype ==1).
	linear:Choose if hidden nodes are calculated from linear manifold(True) or autoencoder manifold(false)
	meanErr: True if you want to calculate the mean std of the fit. (Petkova fig S5)
	meanMedian: True if you want to calculate median std of the distribution. (petkova fig S1)
	aeM: loaded autoencoder model
	positions: array of position along AP axis
	OUTPUT
	prob:array shape (pos,pos). Corresponds to the probability of being at x* if you are a x given by Pxh
	std:array of mean std of fit. length( nb of embryos)
	sig:array of median std of distribution. length( nb of embryos)
	"""
	p=np.linspace(0, off-cut-1 ,off-cut)
	pos=[positions[cut+int(x)] for x in p]
	prob=[]# 2d array of average probability of being a position. This is the prediction map.
	sig=[] #collecting the allSig for all embryos
	std=[]# collecting the allStd for all embryos
	allMaps=[] # collecting the allRows for all embryos. arrays for maps for each embryo.
	#iterate through all embryos
	for emb in range(mutantData.shape[0]):
		print(emb)
		allStd=[]#all std of fits
		allSig=[]# all median sigma
		allRow=[]# all rows of prediction along x*
		#iterate through all positions along AP axis x
		for x in p:
			x=int(x)
			row=[]
			#linear manifold
			if linear:
				layer_output=[mutantData[emb,2,cut+x]-mutantData[emb,0,cut+x],mutantData[emb,3,cut+x]-mutantData[emb,1,cut+x]]
		    # or autoencoder
			else:
				inp=SortData[emb,:,cut+x].reshape((1,4))
				layer_output = get_middle_layer_output([inp])[0][0]

			#for normalization
			Z=Zfunc(layer_output[0],layer_output[1])
			#iterate through all possible position x*
			for y in p:
				y=int(y)
				row.append(Pxh(positions[cut:off][y], layer_output[0],layer_output[1],Z))
			allRow.append(row)
			#calculating positional error with fit like figure S5 of Petkova
			if meanErr:
				try:
					argmax=np.argmax(row)
					#loacalized fit
					if argmax<40:
						width=argmax
					else:
						width=40
					 #edge case
					if argmax<3:
						popt,pcov = curve_fit(gauss,pos[:40],row[:40], p0=[pos[argmax],0.5, 1/(np.sqrt(2*np.pi*1**2))])
					else:
						popt,pcov = curve_fit(gauss,pos[argmax-width:argmax+width],row[argmax-width:argmax+width], p0=[pos[argmax],0.5, 1/(np.sqrt(2*np.pi*1**2))])

				except RuntimeError:
					popt=[np.nan,np.nan]
					print("Error - curve_fit failed")
				#if fit failed
				if (pcov[0,0] == np.nan) or (pcov[1,1] == np.nan)  or (pcov[1,1] >100) or (pcov[0,0]>100):
					allStd.append(np.nan )
				else:    
					allStd.append(abs(popt[1]))
			# other mesure of precision like  figure S1i
			if medianErr:
				#median sigma
				mu=np.sum([row[i]*pos[i] for i in range(len(pos))])
				sigma=np.sqrt(np.sum([((pos[i]-mu)**2) * row[i] for i in range(len(pos))]))
				allSig.append(sigma)
		if meanErr:
			allStd=np.array(allStd)
			std.append(np.nanmean(allStd))
		if medianErr:
			allSig=np.array(allSig)
			sig.append(np.nanmedian(allSig))
	    
		allMaps.append(allRow)
	#average all maps    
	prob=np.mean(allMaps,axis=0)
	    
	print("mean std of localized fit")
	print(np.nanmean(std))
	print("mean(over embryos) of median(over x) std of distributions")
	print(np.nanmean(sig))
	#plot map
	plt.figure(figsize=(6,5))
	conf=plt.contourf(pos,pos,prob.T, 50,cmap='jet', extend='max',vmax=0.04)#CAREFUL prob needs to be transposed
	conf.cmap.set_over("red")
	cb1=plt.colorbar(conf)
	cb1.ax.set_ylabel(r"$P(x^*|{h_i})$")
	plt.xlabel("x")
	plt.ylabel("x*")

	return prob,std,sig
	
