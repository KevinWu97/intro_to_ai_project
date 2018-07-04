import argparse
import math
import os
import random

from timeit import default_timer as timer

def initWeights(features, classes):
	allWeights = {}
	for uniqClass in set(classes):
		allWeights[uniqClass] = [1] * len(features[0])

	return allWeights

def perceptron(featureVec, allWeights):
	scores = {}
	for myClass in allWeights:
		classScore = 0
		for featInd in range(0, len(featureVec)):
			classScore = classScore + (featureVec[featInd] * allWeights[myClass][featInd])

		scores[myClass] = classScore

	return scores

def bestClass(scores):
	bestClass = None
	bestVal = None
	for myClass in scores:
		if bestVal is None or scores[myClass] > bestVal:
			bestClass = myClass
			bestVal = scores[bestClass]

	return bestClass

def perceptronAlg(allFeatures, allWeights, trainClasses):
	correctArr = [False] * len(allFeatures)
	for ind in range(0, len(allFeatures)):
		currScores = perceptron(allFeatures[ind], allWeights)
		guessClass = bestClass(currScores)
		correctArr[ind] = (guessClass == trainClasses[ind])

	return correctArr

def getImageNames(imageType, category, numImages):
	# example: facedata_train_split
	imageDir = imageType + "data_" + category + "_split"

	# array of full paths to each image, given the parent folder
	imageNames = [""] * numImages
	for i in range(0, numImages):
		imageNames[i] = os.path.join(imageDir, imageType + str(i) + ".txt")

	return imageNames

def getFeatures(imageFileName):
	with open(imageFileName, 'r') as imageFile:
		allLines = imageFile.readlines()
		# don't count new line
		lineWidth = len(allLines[0]) - 1

		# pixel data from 2d image is stored as 1d array
		pixelData = [0] * (lineWidth * len(allLines))
		currInd = 0
		for line in allLines:
			for i in range(0, len(line) - 1):
				if line[i] == ' ':
					pixelData[currInd + i] = 0
				else:
					pixelData[currInd + i] = 1
			currInd = currInd + (len(line) - 1)

	return pixelData

def getAllFeatures(imageType, category, numImages):
	# feature list is a 2d array of pixel data, with 0/1 representing empty space or not
	imageNames = getImageNames(imageType, category, numImages)
	allFeatures = [[0]] * len(imageNames)

	for ind in range(0, len(imageNames)):
		allFeatures[ind] = getFeatures(imageNames[ind])

	return allFeatures

def getClasses(imageType, category):
	# naturally the labeling system isn't consistent so we need a bunch of ifs
	if category == "train":
		if imageType == "face":
			labelFileName = "facedatatrainlabels"
		elif imageType == "digit":
			labelFileName = "traininglabels"
	elif category == "test":
		if imageType == "face":
			labelFileName = "facedatatestlabels"
		elif imageType == "digit":
			labelFileName = "testlabels"

	# file consists of numbers, 1 per line
	# transfer that to an array
	with open(labelFileName, 'r') as labelFile:
		allLines = labelFile.readlines()
		classData = [0] * len(allLines)

		for ind in range(0, len(allLines)):
			classData[ind] = int(allLines[ind])

	return classData

def sumFeatures(feat1, feat2):
	return [f1 + f2 for (f1, f2) in zip(feat1, feat2)]

def getCountProbs(allFeatures, trainClasses, allClasses):
	# smoothing value
	k = 1
	countsByClass = []
	for classInd in range(0, len(allClasses)):
		countArr = [0] * len(allFeatures[0])

		for sampleInd in range(0, len(trainClasses)):
			if trainClasses[sampleInd] == allClasses[classInd]:
				
				featureVec = allFeatures[sampleInd]
				for featInd in range(0, len(featureVec)):
					countArr[featInd] = countArr[featInd] + featureVec[featInd]

		for featInd in range(0, len(allFeatures[0])):
			countArr[featInd] = countArr[featInd] + k

		# normalize
		totalSum = sum(countArr)
		for featInd in range(0, len(countArr)):
			countArr[featInd] = math.log(countArr[featInd] / totalSum)

		countsByClass.append(countArr)

	return countsByClass

def dotProd(v1, v2):
	retVal = 0
	for ind in range(0, len(v1)):
		retVal = retVal + (v1[ind] * v2[ind])
	
	return retVal

def calcProbs(allFeatures, classProbs, countProbs):
	finalProbs = []
	for sampleInd in range(0, len(allFeatures)):
		currProbs = {}
		featureVec = allFeatures[sampleInd]

		for classInd in range(0, len(classProbs)):
			currProbs[classInd] = classProbs[classInd] + dotProd(featureVec, countProbs[classInd])

		finalProbs.append(currProbs)

	return finalProbs

def main_naivebayes(imageType, trainPct):
	print("Naive bayes using {} data with {} training data".format(imageType, trainPct))
	start = timer()

	# training setup
	allTrainClasses = getClasses(imageType, "train")
	allTrainFeatures = getAllFeatures(imageType, "train", len(allTrainClasses))

	numToUse = int(trainPct * len(allTrainClasses))
	trainClasses = [0] * numToUse
	trainFeatures = [[0]] * numToUse

	currInd = 0
	for randInd in random.sample(range(0, len(allTrainClasses)), numToUse):
		trainClasses[currInd] = allTrainClasses[randInd]
		trainFeatures[currInd] = allTrainFeatures[randInd]
		currInd = currInd + 1

	allClasses = list(set(trainClasses))

	# training phase
	classProbs = [0] * len(allClasses)
	for classInd in range(0, len(allClasses)):
		numFound = 0
		for sampleInd in range(0, len(trainClasses)):
			if allClasses[classInd] == trainClasses[sampleInd]:
				numFound = numFound + 1

		classProbs[classInd] = math.log(numFound / len(trainClasses))

	countProbs = getCountProbs(trainFeatures, trainClasses, allClasses)

	# testing phase
	testClasses = getClasses(imageType, "test")
	testFeatures = getAllFeatures(imageType, "test", len(testClasses))

	allScores = calcProbs(testFeatures, classProbs, countProbs)
	correctClasses = [False] * len(testFeatures)
	for ind in range(0, len(testFeatures)):
		guessClass = bestClass(allScores[ind])
		correctClasses[ind] = (guessClass == testClasses[ind])
	
	end = timer()
	print(sum(correctClasses) / len(correctClasses))
	print("{} seconds elapsed".format(end - start))

	return classProbs, countProbs

def main_perceptron(imageType, trainPct):
	print("Perceptron using {} data with {} training data".format(imageType, trainPct))
	start = timer()

	# training setup
	allTrainClasses = getClasses(imageType, "train")
	allTrainFeatures = getAllFeatures(imageType, "train", len(allTrainClasses))

	numToUse = int(trainPct * len(allTrainClasses))
	trainClasses = [0] * numToUse
	trainFeatures = [[0]] * numToUse

	currInd = 0
	for randInd in random.sample(range(0, len(allTrainClasses)), numToUse):
		trainClasses[currInd] = allTrainClasses[randInd]
		trainFeatures[currInd] = allTrainFeatures[randInd]
		currInd = currInd + 1

	allWeights = initWeights(trainFeatures, trainClasses)

	# training iteration
	for i in range(0, 20):
		# print(allWeightsFace[100:110]) # faces debug
		# print(allWeights[1][100:110]) # digit debug

		correctClasses = perceptronAlg(trainFeatures, allWeights, trainClasses)
		print(sum(correctClasses) / len(correctClasses))

		for ind in range(0, len(correctClasses)):
			# if incorrect, adjust weights
			if not correctClasses[ind]:
				for myClass in allWeights:
					# increase weights for the correct class
					if myClass == trainClasses[ind]:
						for wInd in range(0, len(allWeights[myClass])):
							allWeights[myClass][wInd] = allWeights[myClass][wInd] + trainFeatures[ind][wInd]
					# decrease weights for all other classes
					else:
						for wInd in range(0, len(allWeights[myClass])):
							allWeights[myClass][wInd] = allWeights[myClass][wInd] - trainFeatures[ind][wInd]

	print('------------------------')
	# testing phase
	testClasses = getClasses(imageType, "test")
	testFeatures = getAllFeatures(imageType, "test", len(testClasses))

	correctClasses = perceptronAlg(testFeatures, allWeights, testClasses)
	end = timer()
	print(sum(correctClasses) / len(correctClasses))
	print("{} seconds elapsed".format(end - start))

	return allWeights

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description='AI Final Project')
	parser.add_argument(type=str, dest = "alg",
					choices = ['perceptron', 'naivebayes'],
					help="algorithm to use: 'perceptron' or 'naivebayes'")
	parser.add_argument(type=str, dest = "imageType",
					choices = ['face', 'digit'],
					help="dataset to use: 'face' or 'digit'")
	parser.add_argument(type=float, dest = "trainPct",
					help="training set percentage to use")
	args = parser.parse_args()
	alg = args.alg
	imageType = args.imageType
	trainPct = args.trainPct

	if alg == 'naivebayes':
		classProbs, countProbs = main_naivebayes(imageType, trainPct)
	elif alg == 'perceptron':
		allWeights = main_perceptron(imageType, trainPct)

	inStr = input()
	while (inStr != 'quit'):
		fileIndex = int(inStr)

		testClasses = getClasses(imageType, "test")
		fileName = getImageNames(imageType, "test", len(testClasses))[fileIndex]

		# https://stackoverflow.com/questions/8084260/python-printing-a-file-to-stdout
		with open(fileName, 'r') as testFile:
			print(testFile.read())

		if alg == 'naivebayes':
			scores = calcProbs([getFeatures(fileName)], classProbs, countProbs)[0]
		elif alg == 'perceptron':
			scores = perceptron(getFeatures(fileName), allWeights)

		print('Actual label: {}'.format(testClasses[fileIndex]))
		print('Label chosen: {}'.format(bestClass(scores)))

		inStr = input()

	# trainPcts = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
	# for pct in trainPcts:
	# 	main_naivebayes("digit", pct)

	# for pct in trainPcts:
	# 	main_naivebayes("face", pct)