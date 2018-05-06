import itertools
import os

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

if __name__ == "__main__":
	imageType = "digit"

	# training setup
	trainClasses = getClasses(imageType, "train")
	trainFeatures = getAllFeatures(imageType, "train", len(trainClasses))
	allWeights = initWeights(trainFeatures, trainClasses)

	# training iteration
	for i in range(0, 20):
		# print(allWeightsFace[100:110]) # faces debug
		print(allWeights[1][100:110]) # digit debug

		correctClasses = perceptronAlg(trainFeatures, allWeights, trainClasses)
		print(sum(correctClasses) / len(correctClasses))

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
	print(sum(correctClasses) / len(correctClasses))
