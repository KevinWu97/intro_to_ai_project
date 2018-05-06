import os

#modified from Berkeley's provided code, retrieves all labels from a file
def getLabels(baseFilePath, percent):
	labelPath = os.path.join(baseFilePath, "labels")
	labelsFull = []
	if(os.path.exists(labelPath)): 
		fin = [l[:-1] for l in open(labelPath).readlines()]
		labelsFull = []
		for line in fin:
			if line == '':
				break
			labelsFull.append(int(line))

	return labelsFull

#retrieves data from images with indices taken from imageList, converted to features
def getFeatures(baseFilePath, prefix, imageList):
	#read image files and generate features, append to matrix
	featureMatrix = []
	for i in range(len(imageList)):
		imagePath = prefix + str(imageList[i]) + ".txt"
		featurePath = os.path.join(baseFilePath, imagePath)
		featureList = []
		with open(featurePath, "r") as f:
			for line in f:
				for c in line:
					if c == ' ':
						featureList.append(0)
					else:
						featureList.append(1)
		featureMatrix.append(featureList)
	return featureMatrix