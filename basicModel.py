# store information about a general model
# and some baselines
import os
import numpy as np
from sklearn import linear_model, svm

SET_NAMES = ["Train", "Dev", "Test"]

# check if the given file path exists, and if not create it.
# based on Krumelur's answer to
# http://stackoverflow.com/questions/12517451/python-automatically-creating-directories-with-file-output
def check_and_create_path(filename):
    if (not os.path.exists(os.path.dirname(filename))):
        try:
            os.makedirs(os.path.dirname(filename))
        except OSError as exc: # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise

# write content to the file at filename. Make the directory path to the given
# file if it does not exist.
def safeWrite(filename, content, dumpJSON=False):
    check_and_create_path(filename)
    out_file = open(filename, "w")
    if dumpJSON:
        content = json.dumps(content)
    out_file.write(content)
    out_file.close()


# Superclass for all modesl
class Model(object):
    def __init__(self):
        super().__init__()
        self.name = "DEFAULT"
        self.trained = False
        self.predicted = False
        self.guesses = {
            "digit": {},
            "logo": {},
            "all": {}
        }
        self.reportInfo = {
            "digit": {},
            "logo": {},
            "all": {}
        }

    def getName(self):
        return self.name

    # children  need to implement a train function

    # get the feature vector for a given point
    def getFeatureVector(self, point):
        return []

    # given a train, dev, and test set, make predictions for all of them
    # then report the information
    def predictAll(self, train, dev, test, decoration):
        if not(self.trained):
            raise("Model is not trained yet!")

        dsets = [train, dev, test]

        for k in range(len(dsets)):
            dset = dsets[k]
            setName = SET_NAMES[k]
            # 0-2 are change types
            # 3, 4 are decorations
            numPoints = [0, 0, 0, 0, 0]
            correctPoints = [0, 0, 0, 0, 0]
            incorrectList = [[], [], [], [], []]

            for i in range(len(dset)):
                point = dset[i]
                pid = point["id"]
                decType = point["decoration"]
                decIndex = 3
                if (decType == "logo"):
                    decIndex = 4

                fv = self.getFeatureVector(point)
                guess = self.predict(fv)
                trueLabel = point["correctEnd"]
                changeType = point["changeType"]

                numPoints[changeType] += 1
                numPoints[decIndex] += 1
                if (trueLabel == guess):
                    correctPoints[changeType] += 1
                    correctPoints[decIndex] += 1
                else:
                    incorrectList[changeType].append(pid)
                    incorrectList[decIndex].append(pid)

            # save info on number of correct
            self.reportInfo[decoration][setName] = [numPoints, correctPoints, incorrectList]

            # save info on failures
            fname = "output/mistakes/%s/%s_%s.txt" % (self.name, decoration, setName)
            output = []
            for k, name in enumerate(["NoChange", "WrongFinal", "Digit", "Logo"]):
                output.append("====== %s ======" % name)
                output.extend(incorrectList[k])
                output.append("")

            safeWrite(fname, "\n".join(output))

        self.predicted = True

    # return a textual report on the evaluation results for this dataset
    # reportTest is true if the report should include test results
    def getTextReport(self, decoration, reportTest):
        report = []
        setNames = SET_NAMES
        if not(reportTest):
            setNames = setNames[:-1]

        lineNames = ["%s:" % self.name]
        for name in ["  (NoChange", "(DiffBlockMoved", "(SameBlockMoved", "     (Digit", "      (Logo"]:
            lineNames.append("    %s):" % name)

        for i, lineName in enumerate(lineNames):
            line = lineName
            for setName in setNames:
                nums, correct, incorrectList = self.reportInfo[decoration][setName]

                if (i == 0):
                    num = np.sum(nums[:3])
                    corr = np.sum(correct[:3])
                else:
                    num = nums[i-1]
                    corr = correct[i-1]

                if (num == 0):
                    pct = 0
                else:
                    pct = (100.0*corr)/num

                line += " %.2f%% (%d/%d) ||" % (pct, corr, num)
            report.append(line)


        report.append("---")

        return "\n".join(report)


    # get this model's row for the latex table reporting results by
    # decoration type
    def getDecorationLatexRow(self, reportTest):

        line = "%s &" % self.name

        for decoration in [0, 2, 3]:#["all", "digit", "logo"]:
            for setName in SET_NAMES[1:]:
                # if we aren't reporting text, show ???
                if (not(reportTest) and setName == SET_NAMES[-1]):
                    line += " --\\% &"
                else:
                    nums, correct, incorrectList = self.reportInfo["all"][setName]

                    if (decoration == 0):
                        num = np.sum(nums[0:3])
                        corr = np.sum(correct[0:3])
                    else:
                        num = nums[decoration]
                        corr = correct[decoration]

                    if (num == 0):
                        pct = 0
                    else:
                        pct = (100.0*corr)/num

                    line += " %.2f\\%% &" % pct

        # remove last tab
        line = line[:-2]
        line += " \\\\\\hline"

        return line

    # get this model's row for the latex table reporting results by subset
    def getSubsetLatexRow(self, reportTest):

        line = "%s &" % self.name

        decoration = "all"
        for i in range(3):
            for setName in SET_NAMES[1:]:
                # if we aren't reporting text, show ???
                if (not(reportTest) and setName == SET_NAMES[-1]):
                    line += " --\\% &"
                else:
                    nums, correct, incorrectList = self.reportInfo[decoration][setName]

                    if (i == 0):
                        num = np.sum(nums[0:3])
                        corr = np.sum(correct[0:3])
                    else:
                        num = nums[i-1]
                        corr = correct[i-1]

                    pct = (100.0*corr)/num

                    line += " %.2f\\%% &" % pct

        # remove last tab
        line = line[:-2]

        line += " \\\\\\hline"

        return line

    # given a test set, evaluate the model
    # "positive" means that the instruction is correct
    # returns true positives, false positives, false negatives, true negatives
    def evaluate(self, testSet):
        if not(self.trained):
            raise("Model is not trained yet!")
        # true label is first
        num_pp = 0
        num_np = 0
        num_pn = 0
        num_nn = 0
        pp_list = []
        np_list = []
        pn_list = []
        nn_list = []

        for i in range(len(testSet)):
            point = testSet[i]
            trueLabel = point["correctEnd"]

            # start = point["startState"]
            # end = point["endState"]
            # ins = point["instruction"]
            fv = self.getFeatureVector(point)
            guess = self.predict(fv)

            if (trueLabel == 0 and guess == 0):
                num_nn += 1
                nn_list.append(i)
            elif (trueLabel == 0): # guess = 1
                num_np += 1
                np_list.append(i)
            elif (guess == 0): # trueLabel = 1
                num_pn += 1
                pn_list.append(i)
            else: # trueLabel = 1, guess = 1
                num_pp += 1
                pp_list.append(i)
        nums = [num_pp, num_np, num_pn, num_nn]
        lists = [pp_list, np_list, pn_list, nn_list]

        return nums, lists


# Baseline that just guesses "Yes, this transition is correct"
class GuessYes(Model):
    def __init__(self):
        super().__init__()

        self.name = "GuessYes"

    # predict that this is the correct end
    def predict(self, fv):
        return 1


    def train(self, train, dev):
        self.trained = True

# Baseline that just guesses "No, this transition is incorrect"
class GuessNo(Model):
    def __init__(self):
        super().__init__()

        self.name = "GuessNo"

    # predict that this is the correct end
    def predict(self, fv):
        return 0


    def train(self, train, dev):
        self.trained = True

# Baseline that just guesses randomly
class GuessRandom(Model):
    def __init__(self):
        super().__init__()

        self.name = "GuessRandom"

    # predict that this is the correct end
    def predict(self, fv):
        return np.random.randint(0, 2)


    def train(self, train, dev):
        self.trained = True

# Superclass for a model that uses bag of words and a sklearn model
class sklearn_BOW(Model):
    def __init__(self):
        super().__init__()

        self.name = "sklearn_BOW"
        self.model = None

    # get the feature vector for a given point
    def getFeatureVector(self, point):
        # print(len(point["startState"]))
        # print(len(point["endState"]))
        # print(len(point["bagOfWords"]))
        start = np.array(point["startState"])
        start = start.flatten()
        end = np.array(point["endState"])
        end = end.flatten()
        ins = np.array(point["bagOfWords"])
        fv = np.append(np.append(start, end), ins)
        fv = fv.flatten()
        return fv

    # predict that this is the correct end
    def predict(self, fv):
        pred = self.model.predict([fv])
        return pred


    def train(self, train, dev):
        fvSize = len(self.getFeatureVector(train[0]))
        numPoints = len(train)
        X = np.zeros((numPoints, fvSize))
        y = np.zeros((numPoints))
        for i in range(numPoints):
            point = train[i]
            fv = self.getFeatureVector(point)
            label = point["correctEnd"]
            X[i] = fv
            y[i] = label

        self.model.fit(X, y)

        self.trained = True

# Logistic regression baseline
class LogisticRegression(sklearn_BOW):
    def __init__(self):
        super().__init__()

        self.name = "LogisticRegression"
        self.model = linear_model.LogisticRegression()


# SVM baseline with a linear kernel
class Linear_SVM(sklearn_BOW):
    def __init__(self):
        super().__init__()

        self.name = "LinearSVM"
        self.model = svm.LinearSVC()
