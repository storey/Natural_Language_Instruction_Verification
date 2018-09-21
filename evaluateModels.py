# evaluate a set of models
import json

import numpy as np
import dynet as dy

from basicModel import GuessYes, GuessRandom, LogisticRegression, Linear_SVM
from model import NeuralBaseline, MovedLearner, GuessEndv2, FullView

# Models to run
modelBluePrints = [
    lambda : GuessYes(),
    lambda : GuessRandom(),
    lambda : NeuralBaseline(dy.Model(), "rel", False, 128, 0.1, 0.01),
    lambda : LogisticRegression(),
    lambda : Linear_SVM(),
    lambda : MovedLearner(),
    lambda : FullView(dy.Model(), "rel", False, 64, 128, 0.1, 0.01, True, True, True),
    lambda : GuessEndv2(dy.Model(), "rel", 128, 128, 0.0, 0.01, True),
    lambda : GuessEndv2(dy.Model(), "rel", 128, 128, 0.0, 0.01, False),
]

models = []
for fun in modelBluePrints:
    mod = fun()
    models.append(mod)


cleanTextReport = []
fullTextReport = []

decorationTypes = ["all"]
for dtype in decorationTypes:
    print("Working on type: %s" % dtype)

    header = "========= %s =========" % dtype
    cleanTextReport.append(header)
    fullTextReport.append(header)

    # load data
    filename = "data/train_%s.json" % dtype
    f = open(filename, 'r')
    train = json.loads(f.read())

    filename = "data/dev_%s.json" % dtype
    f = open(filename, 'r')
    dev = json.loads(f.read())

    filename = "data/test_%s.json" % dtype
    f = open(filename, 'r')
    test = json.loads(f.read())

    # train and evaluate each model
    for model in models:
        mName = model.getName()

        print("  Model %s..." % mName)

        model.train(train, dev)

        model.predictAll(train, dev, test, dtype)

        cleanReport = model.getTextReport(dtype, False)
        fullReport = model.getTextReport(dtype, True)

        cleanTextReport.append(cleanReport)
        fullTextReport.append(fullReport)

# save the text report
out_file = open("output/textReport.txt", 'w')
out_file.write("\n".join(cleanTextReport))
out_file.close()

out_file = open("output/restricted/fullTextReport.txt", 'w')
out_file.write("\n".join(fullTextReport))
out_file.close()

# save the latex table reports
cleanDecorationLatexTable = []
fullDecorationLatexTable = []

cleanSubsetLatexTable = []
fullSubsetLatexTable = []

for model in models:
    cleanLatex = model.getDecorationLatexRow(False)
    fullLatex = model.getDecorationLatexRow(True)

    cleanDecorationLatexTable.append(cleanLatex)
    fullDecorationLatexTable.append(fullLatex)

    cleanLatex = model.getSubsetLatexRow(False)
    fullLatex = model.getSubsetLatexRow(True)

    cleanSubsetLatexTable.append(cleanLatex)
    fullSubsetLatexTable.append(fullLatex)

out_file = open("output/decorationTable.txt", 'w')
out_file.write("\n".join(cleanDecorationLatexTable))
out_file.close()

out_file = open("output/restricted/decorationTable.txt", 'w')
out_file.write("\n".join(fullDecorationLatexTable))
out_file.close()

out_file = open("output/subsetTable.txt", 'w')
out_file.write("\n".join(cleanSubsetLatexTable))
out_file.close()

out_file = open("output/restricted/subsetTable.txt", 'w')
out_file.write("\n".join(fullSubsetLatexTable))
out_file.close()
