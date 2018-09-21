# process the dataset's data

# shape_params: always ['blue', 'green', 'cyan', 'magenta', 'yellow', 'red']
# decoration: "digit" or "logo"; an even split
# notes:
#    start: start state
#    finish: end state
#    notes: natural language instructions
#    type: A0 (single); A1 (short multi) A2 (long)
#    users: users who annotated, I guess
# states: stores the states; 1 more than A0s in notes
#    each state has up to 20 items in it
#       each item is an x position, y position, z position (y is always 0.1)
# filename: different files
# images: some images related to filename
# side_length: always 0.1524
import json
import copy
import re
import uuid

from collections import Counter

import numpy as np

# preprocess a word
def preprocessWord(word):
    return word.lower()

# preprocess a sentence
def preprocessSentence(sent):
    newSent = re.sub(r'([,\.\'\)\(;\"!?])', r' \1 ', sent)
    newSent = re.sub(r'\s+', ' ', newSent)
    newSent = re.split(r'\s', newSent)
    newSent = list(map(preprocessWord, newSent))
    return newSent

# get the map of words to indices
def getWordMap(digitDataPoints, logoDataPoints):
    word_counter = Counter()

    for dat in [digitDataPoints, logoDataPoints]:
        for d in dat:
            instructionArray = preprocessSentence(d["instruction"])
            word_counter.update(instructionArray)

    NUM_COPIES = 2 # each data point has a correct and incorrect end
    UNK_THRESHOLD = 2*NUM_COPIES

    words = [w for w, _ in word_counter.most_common() if word_counter[w] >= UNK_THRESHOLD]
    unkWords = [w for w, _ in word_counter.most_common() if word_counter[w] < UNK_THRESHOLD]
    allWords = [w for w, _ in word_counter.most_common()]

    print("===")
    print("Total number of tokens: %d" % (len(allWords)))
    print("Number of represented tokens: %d" % (len(words)))
    print("Number of UNK tokens: %d" % len(unkWords))

    #print([w for w, _ in word_counter.most_common() if (word_counter[w] < 3 and word_counter[w] > 1)])


    # index 0 reserved for UNK
    myMap = {}
    for i in range(len(words)):
        word = words[i]
        myMap[word] = i + 1

    return myMap, words


# return if two states are the same
def isSameState(state1, state2):
    l1 = len(state1)
    l2 = len(state2)
    if not(l1 == l2):
        return False

    for i in range(l1):
        x1, y1, z1 = state1[i]
        x2, y2, z2 = state2[i]

        if (not(x1 == x2) or not(y1 == y2) or not(z1 == z2)):
            return False
    return True


# get the list of datasets
dsets = ["train", "dev", "test"]#["dev"]#
wordMap = {}
words = []

examplesOutput = []

# for each dataset...
for dset in dsets:

    print("== " + dset)

    # load the original version of the dataset
    filename = "data/%sset.json" % dset
    f = open(filename, 'r')
    rawData = f.read()
    lines = rawData.split("\n")

    # go through the lines, extracting single instruction transitions
    digitDataPoints = []
    logoDataPoints = []

    exampleInstructions = []

    for line in lines:
        if (line == ""):
            continue
        lineData = json.loads(line)

        isDigits = lineData["decoration"] == "digit"

        dataPoints = []
        for note in lineData["notes"]:

            # only include 1-step transitions
            if (note["type"] == "A0"):
                if (np.random.randint(0, 49) == 0):
                    exampleInstructions.append(note["notes"])


                startState = copy.deepcopy(lineData["states"][note["start"]])
                endState = copy.deepcopy(lineData["states"][note["finish"]])

                # all transitions move a single block, so we know this works
                blockMoved = -1
                for i in range(len(startState)):
                    sX, sY, sZ = startState[i]
                    eX, eY, eZ = endState[i]

                    if (not(sX == eX) or not(sY == eY) or not(sZ == eZ)):
                        blockMoved = i
                        break

                # make sure we don't swap examples of the same instruction
                instructionID = str(uuid.uuid4())
                for instruction in note["notes"]:
                    pointID = str(uuid.uuid4())

                    dataPoint = {
                        "decoration": lineData["decoration"],
                        "startState": startState,
                        "endState": endState,
                        "instructionID": instructionID,
                        "numBlocks": len(startState),
                        "blockMoved": blockMoved,
                        "instruction": instruction,
                        "changeType": 0, # 0 is no, 1 is yes, 2 is yes and same block
                        "id": pointID
                    }
                    dataPoints.append(dataPoint)
        if isDigits:
            digitDataPoints.extend(dataPoints)
        else:
            logoDataPoints.extend(dataPoints)
    numDig = len(digitDataPoints)
    numLog = len(logoDataPoints)
    print("Number of total data points: %d" % (numLog + numDig))
    print("Number of digit data points: %d" % numDig)
    print("Number of logo data points: %d" % numLog)

    # store some example instructions
    numExamples = 0
    for e in exampleInstructions:
        numExamples += len(e)
        examplesOutput.extend(e)
        examplesOutput.append("================\n")

    print(numExamples)

    # shuffle data and add new points
    for d in [digitDataPoints, logoDataPoints]:
        np.random.shuffle(d)
        numDiffEnds = 0
        numPoints = len(d)
        dSplit  = int(numPoints)/2
        numSameBlockDiffEnd = 0
        numDiffBlockDiffEnd = 0

        # for each point, add a new one with a different end state,
        # either with the same block moved (50%) or a different block moved (50%)
        for i in range(numPoints):
             point = d[i]
             newPoint = copy.deepcopy(point)

             properMoved = newPoint["blockMoved"]

             # move new block
             if i < dSplit:
                 moved = properMoved
                 while moved == properMoved or moved >= len(newPoint["endState"]):
                     moved = np.random.randint(0, 20)

                 newPoint["blockMoved"] = moved
                 newPoint["changeType"] = 1
                 numDiffBlockDiffEnd += 1
             else: #move same block
                 moved = properMoved

                 newPoint["changeType"] = 2
                 numSameBlockDiffEnd += 1

             oldX = newPoint["endState"][moved][0]
             oldZ = newPoint["endState"][moved][2]
             newX = (2*np.random.random()) - 1
             newZ = (2*np.random.random()) - 1
             while ((np.abs(newX - oldX) < 0.2) and (np.abs(newZ - oldZ) < 0.2)):
                 newX = (2*np.random.random()) - 1
                 newZ = (2*np.random.random()) - 1


             newPoint["endState"] = copy.deepcopy(newPoint["startState"])
             newPoint["endState"][moved] = [newX, 0.1, newZ]


             newPoint["id"] = str(uuid.uuid4())
             d.append(newPoint)
             numDiffEnds += 1

        print("%d/%d/%d" % ((len(d) - numDiffEnds), numDiffBlockDiffEnd, numSameBlockDiffEnd))

        # run some accounting
        trueCount = 0
        falseDiffCount = 0
        falseSameCount = 0
        for i in range(len(d)):
            c = d[i]["changeType"]
            if c == 0:
                trueCount += 1
                d[i]["correctEnd"] = 1
            elif c == 1:
                falseDiffCount += 1
                d[i]["correctEnd"] = 0
            elif c == 2:
                falseSameCount += 1
                d[i]["correctEnd"] = 0
            else:
                print("Error: changed status was not 0 or 1")
        print("%d/%d/%d" % (trueCount, falseDiffCount, falseSameCount))

    # add extra blocks if there are fewer than 20
    missingPointVal = -1000.0
    for dat in [digitDataPoints, logoDataPoints]:
        for d in dat:
            myLen = len(d["startState"])
            d["startState"] = copy.deepcopy(d["startState"])
            d["endState"] = copy.deepcopy(d["endState"])
            for i in range(myLen, 20):
                point = [missingPointVal, missingPointVal, missingPointVal]
                d["startState"].append(point)
                d["endState"].append(point)
            # myLen = len(d["startState"])
            # if not(myLen == 20):
            #     print(myLen)
            # myLen = len(d["endState"])
            # if not(myLen == 20):
            #     print(myLen)

    # word counting
    if (dset == "train"):
        wordMap, words = getWordMap(digitDataPoints, logoDataPoints)

    vocabSize = len(wordMap)
    for dat in [digitDataPoints, logoDataPoints]:
        for d in dat:
            # for each point, get a bag of words and instruction vector
            wordArray = preprocessSentence(d["instruction"])

            newArr = []
            bagOfWords = np.zeros((vocabSize+1))
            for w in wordArray:
                if w in wordMap:
                    wordIndex = wordMap[w]
                    newArr.append(wordIndex)
                    bagOfWords[wordIndex] += 1
                else:
                    newArr.append(0)
            d["instructionVector"] = newArr
            d["bagOfWords"] = bagOfWords.tolist()



    # write data files
    allDataPoints = []
    allDataPoints.extend(digitDataPoints)
    allDataPoints.extend(logoDataPoints)

    filename = "data/%s_digit.json" % dset
    out_file = open(filename, 'w')
    out_file.write(json.dumps(digitDataPoints))
    out_file.close()

    filename = "data/%s_logo.json" % dset
    out_file = open(filename, 'w')
    out_file.write(json.dumps(logoDataPoints))
    out_file.close()

    filename = "data/%s_all.json" % dset
    out_file = open(filename, 'w')
    out_file.write(json.dumps(allDataPoints))
    out_file.close()

out_file = open("output/randomInstructions.txt", 'w')
out_file.write("\n".join(examplesOutput))
out_file.close()


if True:
    # prep glove
    # distribute uniformly at random between -1 and 1
    print(vocabSize)
    baseWordEmbeddings = (2*np.random.rand(vocabSize+1, 300)-1)

    filename = "glove/glove.840B.300d.txt"
    f = open(filename, 'r')
    # rawData = f.read()
    # lines = rawData.split("\n")

    wordsFound = 0
    i = 0
    foundList = np.zeros((vocabSize+1))
    foundList[0] = 1
    for line in f:
        split = line.split(" ")
        gWord = split[0]
        dims = split[1:]
        for word in words:
            if gWord == word:
                print(word)
                wordsFound += 1
                index = wordMap[word]
                foundList[index] = 1
                for j, d in enumerate(dims):
                    baseWordEmbeddings[index][j] = float(d)
        if i % 100000 == 0:
            print("%d: %s" % (int(i/100000), gWord))
        i += 1

    print("Found %d of %d" % (wordsFound, len(words)))
    print("Words not found:")
    for i, f in enumerate(foundList):
        if (f == 0):
            print(words[i-1])



    out = json.dumps(baseWordEmbeddings.tolist())
    out_file = open("glove/embed.json", 'w')
    out_file.write(out)
    out_file.close()
