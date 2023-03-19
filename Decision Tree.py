import math
import numpy as np

# load files
trainFile = open(r"pendigits_training.txt")
testFile = open(r"pendigits_test.txt")

trainData = []
trainDataFeatures = []
trainTarget = []

testData = []
testDataFeatures = []
testTarget = []

# split both train and test file into features and target column
for l in trainFile:
    temp = l.split()
    res = [eval(i) for i in temp]
    trainData.append(res)

for l in testFile:
    temp = l.split()
    res = [eval(i) for i in temp]
    testData.append(res)

for outerIndex in range(len(trainData)):
    tempList = []
    for innerListIndex in range(len(trainData[outerIndex])):
        if innerListIndex == len(trainData[outerIndex]) - 1:
            trainTarget.append(trainData[outerIndex][innerListIndex])
        else:
            tempList.append(trainData[outerIndex][innerListIndex])
    trainDataFeatures.append(tempList)

for outerIndex in range(len(testData)):
    tempList = []
    for innerListIndex in range(len(testData[outerIndex])):
        if innerListIndex == len(testData[outerIndex]) - 1:
            testTarget.append(testData[outerIndex][innerListIndex])
        else:
            tempList.append(testData[outerIndex][innerListIndex])
    testDataFeatures.append(tempList)


# normalizing training features
# F(v) = (v - mean) / std
def normalization(trainDataFeatures, testDataFeatures):
    normTrainDataFeatures = []
    normTestDataFeatures = []
    # Normalizing training data
    for i in range(len(trainDataFeatures[0])):
        sumMean = 0
        for j in range(len(trainDataFeatures)):
            sumMean += trainDataFeatures[j][i]
        mean = sumMean / len(trainDataFeatures)
        std = 0
        totSTD = 0
        for v in range(len(trainDataFeatures)):
            std += ((trainDataFeatures[v][i] - mean) ** 2)
        totSTD = math.sqrt(std / len(trainDataFeatures))
        tempList = []
        for k in range(len(trainDataFeatures)):
            tempList.append((trainDataFeatures[k][i] - mean) / totSTD)
        normTrainDataFeatures.append(tempList)
        array = np.array(normTrainDataFeatures)
        transposed_array = array.T
        transNormTrainDataFeatures = transposed_array.tolist()

        # Normalizing Test data
    for i in range(len(testDataFeatures[0])):
        sumMean = 0
        for j in range(len(testDataFeatures)):
            sumMean += testDataFeatures[j][i]
        mean = sumMean / len(testDataFeatures)
        std = 0
        totSTD = 0
        for v in range(len(testDataFeatures)):
            std += ((testDataFeatures[v][i] - mean) ** 2)
        totSTD = math.sqrt(std / len(testDataFeatures))
        tempList = []
        for k in range(len(testDataFeatures)):
            tempList.append((testDataFeatures[k][i] - mean) / totSTD)
        normTestDataFeatures.append(tempList)
        array = np.array(normTestDataFeatures)
        transposed_array = array.T
        transNormTestDataFeatures = transposed_array.tolist()
    return transNormTrainDataFeatures, transNormTestDataFeatures


# calc distance between all training objects and each testing obj
def eculidean_distance(trainFeatures, testObj, trainTarget):
    distances = []
    for i in range(len(trainFeatures)):
        sum = 0
        for j in range(len(trainFeatures[i])):
            sum += ((trainFeatures[i][j] - testObj[j]) ** 2)
        sqrtVar = math.sqrt(sum)
        distances.append([sqrtVar, trainTarget[i]])
    distances.sort()
    return distances


def getClass(classes):
    # if k = 1 , means only one neighbour with one label
    if len(classes) == 1:
        return classes[0]
    else:  # k >1 , more than one neighbour
        mostRepeatedClass = {}
        for j in classes:
            if j in mostRepeatedClass:
                mostRepeatedClass[j] += 1
            else:
                mostRepeatedClass[j] = 1
        maxVal = max(mostRepeatedClass.values())
        classLabels = [k for k, v in mostRepeatedClass.items() if v == maxVal]
        # #means that one label has max votes
        if len(classLabels) == 1:
            return classLabels[0]
        else:
            # tie is broken --> since equal votes
            occurrenceDict = {}
            for val in range(len(classLabels)):
                occurrenceDict[classLabels[val]] = trainTarget.index(classLabels[val])
            # get first occurrence
            return next(iter(occurrenceDict))


totLabels = []
distanceResult = []
normTrainDataFeatures, normTestDataFeatures = normalization(trainDataFeatures, testDataFeatures)
for i, j in enumerate(normTestDataFeatures):
    distanceResult = eculidean_distance(normTrainDataFeatures, j, trainTarget)
    labelTemp = []
    for k in range(1, 10):
        classes = []
        for neighbour in distanceResult[:k]:
            classes.append(neighbour[1])
        labelTemp.append(getClass(classes))
    totLabels.append(labelTemp)

acc = []
for k in range(1, 10):
    exact = 0
    print("k = ", k)
    for indx, val in enumerate(totLabels):
        print("Predicted --> ", val[k - 1], "Actual --> ", testTarget[indx])
        if val[k - 1] == testTarget[indx]:
            exact += 1
    acc.append((exact / len(testTarget)) * 100)
    print("Accuracy at k = ", k, "is: ", acc[k - 1])
    print("Matched results at k = ", k, "is: ", exact, "from", len(testDataFeatures))





