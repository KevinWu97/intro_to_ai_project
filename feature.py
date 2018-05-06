import os

#reads inputs as text files, with labels in a separate text file as a single line
#returns single matrix, formatted for training, each contained array holding <classification, feature 1, feature 2, etc.>
def getFeaturesTrain(baseFilePath, prefix, imageList):
	#read label file and store in list
	categories = []
	labelPath = os.path.join(baseFilePath, "labels.txt")
	with open(labelPath, "r") as f:
		for line in f:
			for i in len(line):
				categories.append(int(line[i]))

	#read image files and generate features, append to matrix
	featureMatrix = []
	for i in range(len(imageList)):
		imagePath = prefix + str(imageList[i]) + ".txt"
		featurePath = os.path.join(baseFilePath, imagePath)
		featureList = []
		featureList.append(categories[i])
		with open(featurePath, "r") as f:
			for line in f:
				for j in len(line):
					if line[j] == ' ':
						featureList.append(0)
					else:
						featureList.append(1)
		featureMaxtrix.append(featureList)
	return featureMatrix

#reads inputs as text files, with labels in a separate text file as a single line
#returns two matrices, formatted for testing
#first output is list of labels, second is matrix of features
def getFeaturesTest(baseFilePath, prefix, lenElements):
	#read label file and store in list
	categories = []
	labelPath = os.path.join(baseFilePath, "labels.txt")
	with open(labelPath, "r") as f:
		for line in f:
			for i in len(line):
				categories.append(int(line[i]))

	#read image files and generate features, append to matrix
	featureMatrix = []
	for i in range(lenElements):
		imagePath = prefix + i + ".txt"
		featurePath = os.path.join(baseFilePath, imagePath)
		featureList = []
		with open(featurePath, "r") as f:
			for line in f:
				for j in len(line):
					if line[j] == ' ':
						featureList.append(0)
					else:
						featureList.append(1)
		featureMaxtrix.append(featureList)
	return categories, featureMatrix