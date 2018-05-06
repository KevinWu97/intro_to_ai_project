import math
import os

#modified from Berkeley's provided code, retrieves all labels from a file
def getLabels(baseFilePath):
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

#retrieves all data from images, converted to features
def getAllFeatures(baseFilePath, imageType, imageCount):
	#read image files and generate features, append to matrix
	featureMatrix = []
	for i in range(imageCount):
		imagePath = iamgeType + str(i) + ".txt"
		featurePath = os.path.join(baseFilePath, imagePath)
		featureList = []
		featureMatrix.append(getFeatureSingle(featurePath, imageType))
	return featureMatrix

#retrieves data from images with indices taken from imageList, converted to features
def getFeatures(baseFilePath, imageType, imageList):
	#read image files and generate features, append to matrix
	featureMatrix = []
	for i in range(len(imageList)):
		imagePath = imageType + str(imageList[i]) + ".txt"
		featurePath = os.path.join(baseFilePath, imagePath)
		featureList = []
		featureMatrix.append(getFeatureSingle(featurePath, imageType))
	return featureMatrix

#converts a single image to features
def getFeaturesSingle(filePath, imageType):
	#generic features
	featureList = []
	with open(filePath, "r") as f:
		for line in f:
			for c in line:
				if c == ' ':
					featureList.append(0)
				else:
					featureList.append(1)

	#specialized features

	return featureList