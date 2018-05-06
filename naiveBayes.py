import math

class naiveBayes(object):
	def __init__(self):
		self.classProb = []
		self.featureProb = []
		self.smoothing = 1

	def train(self, numClass, sampleSet):
		print(len(sampleSet))
		#get count of features
		featureCount = len(sampleSet[0]) - 1

		#create matrix featureProb[classification][feature number], initialized to smoothing for all values
		self.featureProb = []
		for i in range(numClass):
			temp = []
			for i in range(featureCount):
				temp.append(self.smoothing)
			self.featureProb.append(temp)

		#create matrix self.classProb[classification], initialized to 0 for all values (no smoothing)
		self.classProb = []
		for i in range(numClass):
			self.classProb.append(0)

		#for each element in the set get classification, then add +1 to each feature present (for binary)
		#additionally, increment frequency of classifications
		for i in range(len(sampleSet)):
			currClassification = sampleSet[i][0]
			self.classProb[currClassification] += 1
			for j in range(featureCount):
				self.featureProb[currClassification][j] += sampleSet[i][j + 1]

		#normalize for self.featureProb
		for i in range(featureCount):
			total = 0
			for j in range(numClass):
				total += self.featureProb[j][i]
			for j in range(numClass):
				self.featureProb[j][i] /= total

		print(self.classProb)
		#normalize for self.classProb
		total = 0
		for i in range(numClass):
			total += self.classProb[i]
		for i in range(numClass):
			self.classProb[i] /= total

		print(self.classProb)

	#checks list of images reduced to features, returns labels
	def test(self, testCases):
		testResults = []
		for test in testCases:
			if len(self.classProb) == 0:
				print("Classifier not initialized, please initialize first.\n")
			else:
				#preset the max value and the index of its classification to 0
				maxClass = 0
				maxVal = 0
				
				#initialize max value
				maxVal = math.log(self.classProb[0])
				for i in range(len(self.featureProb[0])):
					if test[i] == 1:
						maxVal += math.log(self.featureProb[0][i])
					else:
						maxVal += math.log(1 - self.featureProb[0][i])

				#iterate through possible classifications, selecting for max value
				for i in range(1, len(self.classProb)):
					temp = math.log(self.classProb[i])
					for j in range(len(self.featureProb[0])):
						if test[j] == 1:
							maxVal += math.log(self.featureProb[i][j])
						else:
							maxVal += math.log(1 - self.featureProb[i][j])
					if temp > maxVal:
						maxClass = i
						maxVal = temp
				testResults.append(maxClass)
		return testResults




