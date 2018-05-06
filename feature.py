import os

#reads inputs as text files, with labels in a separate text file as a single line
#returns single matrix, formatted for training, each contained array holding <classification, feature 1, feature 2, etc.>
def getFeaturesJoined(baseFilePath, prefix, imageList):
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
				for c in line:
					if c == ' ':
						featureList.append(0)
					else:
						featureList.append(1)
		featureMatrix.append(featureList)
	return featureMatrix

#same as above, but separated feature and labels lists
def getFeaturesSeparate():
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
		with open(featurePath, "r") as f:
			for line in f:
				for c in line:
					if c == ' ':
						featureList.append(0)
					else:
						featureList.append(1)
		featureMatrix.append(featureList)
	return categories, featureMatrix

#same as above, but covers all in section, suitable for testing
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
				for c in line:
					if c == ' ':
						featureList.append(0)
					else:
						featureList.append(1)
		featureMatrix.append(featureList)
	return categories, featureMatrix