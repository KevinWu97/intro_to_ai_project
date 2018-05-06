import itertools
import os

# https://stackoverflow.com/questions/952914/making-a-flat-list-out-of-list-of-lists-in-python
myFlatten = lambda l: list(itertools.chain(*l))

def initWeights(features):
	return [1] * len(features[0])

def perceptron(features, weights):
	return [sum([weights[i] * f for i, f in enumerate(featureVec)]) for featureVec in features]

def perceptronAlg(allFeatures, allWeightsFace, allWeightsNonFace, trainClasses):
	pFace = perceptron(allFeatures, allWeightsFace)
	pNonFace = perceptron(allFeatures, allWeightsNonFace)

	guessClasses = [1 if face >= nonFace else 0 for (face, nonFace) in zip(pFace, pNonFace)]
	return [actual == guess for (actual, guess) in zip(trainClasses, guessClasses)]

def getImageNames(category, numImages):
	if category == "train":
		imageDir = "facedata_train_split"
	elif category == "test":
		imageDir = "facedata_test_split"

	return [os.path.join(imageDir, "face" + str(i) + ".txt") for i in range(0, numImages)]

def getFeatures(imageFileName):
	with open(imageFileName, 'r') as imageFile:
		pixelData = myFlatten([[0 if c == ' ' else 1 for c in lineStr[:-1]] for lineStr in imageFile])

	return pixelData

def getAllFeatures(category, numImages):
	imageNames = getImageNames(category, numImages)
	return [getFeatures(fileName) for fileName in imageNames]

def getClasses(category):
	if category == "train":
		labelFileName = "facedatatrainlabels" # os.path.join("imageDir", )
	elif category == "test":
		labelFileName = "facedatatestlabels"

	with open(labelFileName, 'r') as labelFile:
		classData = [int(l) for l in labelFile.readlines()]

	return classData

if __name__ == "__main__":
	imageType = "face"

	# training setup
	trainClasses = getClasses("train")
	multFactors = [2*i - 1 for i in trainClasses]

	trainFeatures = getAllFeatures("train", len(trainClasses))

	allWeightsFace = initWeights(trainFeatures)
	allWeightsNonFace = initWeights(trainFeatures)

	# training iteration
	for i in range(0, 20):
		# print(allWeightsFace[100:110]) # faces debug
		print(allWeightsFace[100:110]) # digit debug

		correctClasses = perceptronAlg(trainFeatures, allWeightsFace, allWeightsNonFace, trainClasses)
		print(sum(correctClasses) / len(correctClasses))

		for ind, guess in enumerate(correctClasses):
			if not guess:
				allWeightsFace = [w + multFactors[ind] * trainFeatures[ind][j] for j, w in enumerate(allWeightsFace)]
				allWeightsNonFace = [w - multFactors[ind] * trainFeatures[ind][j] for j, w in enumerate(allWeightsNonFace)]

	print('------------------------')
	# testing phase
	testClasses = getClasses("test")
	testFeatures = getAllFeatures("test", len(testClasses))

	correctClasses = perceptronAlg(testFeatures, allWeightsFace, allWeightsNonFace, testClasses)
	print(sum(correctClasses) / len(correctClasses))
