import argparse
import itertools
import math
import os
import random

from functools import reduce

from timeit import default_timer as timer

# https://stackoverflow.com/questions/952914/making-a-flat-list-out-of-list-of-lists-in-python
myFlatten = lambda l: list(itertools.chain(*l))

def initWeights(features, classes):
	return {uniqClass: [1] * len(features[0]) for uniqClass in set(classes)}

def perceptron(featureVec, allWeights):
	return {cInd: sum([f * w for f, w in zip(featureVec, allWeights[cInd])]) for cInd in allWeights}

def perceptronAlg(allFeatures, allWeights, trainClasses):
	classScores = [perceptron(featureVec, allWeights) for featureVec in allFeatures]

	guessClasses = [max(scores, key = scores.get) for scores in classScores]
	return [actual == guess for (actual, guess) in zip(trainClasses, guessClasses)]

def getImageNames(imageType, category, numImages):
	# example: facedata_train_split
	imageDir = imageType + "data_" + category + "_split"

	return [os.path.join(imageDir, imageType + str(i) + ".txt") for i in range(0, numImages)]

def getFeatures(imageFileName):
	with open(imageFileName, 'r') as imageFile:
		pixelData = myFlatten([[0 if c == ' ' else 1 for c in lineStr[:-1]] for lineStr in imageFile])

	return pixelData

def getAllFeatures(imageType, category, numImages):
	imageNames = getImageNames(imageType, category, numImages)
	return [getFeatures(fileName) for fileName in imageNames]

def getClasses(imageType, category):
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

	with open(labelFileName, 'r') as labelFile:
		classData = [int(l) for l in labelFile.readlines()]

	return classData

def sumFeatures(feat1, feat2):
	return [f1 + f2 for (f1, f2) in zip(feat1, feat2)]

def getCountProbs(allFeatures, trainClasses, allClasses):
	# group feature vecs by class
	featuresByClass = [[allFeatures[ind] for ind, trainClass in enumerate(trainClasses) if trainClass == classId] for classId in allClasses]
	# sum all features together to produce counts
	countsByClass = [reduce(sumFeatures, featureList) for featureList in featuresByClass]
	# smooth counts (k = 1)
	countsByClass = [[c + 1 for c in featureCounts] for featureCounts in countsByClass]
	# normalize and log-scale counts
	return [[math.log(count) - math.log(sum(featureCounts)) for count in featureCounts] for featureCounts in countsByClass]

def dotProd(v1, v2):
	return sum([e1 * e2 for (e1, e2) in zip(v1, v2)])

def calcProbs(allFeatures, classProbs, countProbs):
	return [{cInd: cProb + dotProd(featureVec, countProbs[cInd]) for cInd, cProb in enumerate(classProbs)} for featureVec in allFeatures]

def main_naivebayes(imageType, trainPct):
	print("Naive bayes using {} data with {} training data".format(imageType, trainPct))
	start = timer()

	# training setup
	trainClasses = getClasses(imageType, "train")
	trainFeatures = getAllFeatures(imageType, "train", len(trainClasses))

	numToUse = int(trainPct * len(trainClasses))

	trainClasses, trainFeatures = zip(*random.sample(list(zip(trainClasses, trainFeatures)), numToUse))
	allClasses = list(set(trainClasses))

	# training phase
	classProbs = [sum([classId == trainClass for trainClass in trainClasses]) / len(trainClasses) for classId in allClasses]
	classProbs = [math.log(classProb) for classProb in classProbs]

	countProbs = getCountProbs(trainFeatures, trainClasses, allClasses)

	# testing phase
	testClasses = getClasses(imageType, "test")
	testFeatures = getAllFeatures(imageType, "test", len(testClasses))

	allScores = calcProbs(testFeatures, classProbs, countProbs)
	classGuesses = [max(scores, key = scores.get) for scores in allScores]
	correctClasses = [c1 == c2 for (c1, c2) in zip(classGuesses, testClasses)]
	
	end = timer()
	print(sum(correctClasses) / len(correctClasses))
	print("{} seconds elapsed".format(end - start))

	return classProbs, countProbs

def main_perceptron(imageType, trainPct):
	# training setup
	print("Perceptron using {} data with {} training data".format(imageType, trainPct))
	start = timer()
	trainClasses = getClasses(imageType, "train")
	trainFeatures = getAllFeatures(imageType, "train", len(trainClasses))

	numToUse = int(trainPct * len(trainClasses))

	trainClasses, trainFeatures = zip(*random.sample(list(zip(trainClasses, trainFeatures)), numToUse))

	allWeights = initWeights(trainFeatures, trainClasses)

	# training iteration
	for i in range(0, 20):
		# print(allWeightsFace[100:110]) # faces debug
		# print(allWeights[1][100:110]) # digit debug

		correctClasses = perceptronAlg(trainFeatures, allWeights, trainClasses)
		# print(sum(correctClasses) / len(correctClasses))

		for ind, guess in enumerate(correctClasses):
			if not guess:
				for cInd in allWeights:
					if cInd == trainClasses[ind]:
						allWeights[cInd] = [w + trainFeatures[ind][j] for j, w in enumerate(allWeights[cInd])]
					else:
						allWeights[cInd] = [w - trainFeatures[ind][j] for j, w in enumerate(allWeights[cInd])]

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
		print('Label chosen: {}'.format(max(scores, key = scores.get)))

		inStr = input()

	# trainPcts = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
	# for pct in trainPcts:
	# 	main_naivebayes("digit", pct)

	# for pct in trainPcts:
	# 	main_naivebayes("face", pct)