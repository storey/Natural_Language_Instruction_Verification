# evaluate a single model and send an email when it is done
# python3 evaluateSingleModel.py --dynet-mem 2048
import json
import numpy as np

import dynet as dy

from basicModel import GuessYes
from model import WorldConcat, FullView, NeuralBaseline, MovedLearner, GuessEnd, GuessEndv2, SourceBlockPredictor

from graphNeuralLoss import check_and_create_path
from sendEmail import sendEmail

modelBluePrints = [
    #lambda : GuessYes(),
    #lambda : MovedLearner(),
    # lambda : NeuralBaseline(dy.Model(), "rel", False, 64, 0.1, 0.01),
    # lambda : FullView(dy.Model(), "rel", False, 64, 64, 0.1, 0.01, True, False, False),
    # lambda : GuessEnd(dy.Model(), "rel", True, 128, 64, 0.1, 0.01, True),
    #lambda : GuessEndv2(dy.Model(), "rel", 128, 0.1, 0.01, True),
    #lambda : WorldConcat(dy.Model(), "rel", False, 64, 64, 0.1, 0.01, True, True, True),
    #lambda : WorldConcat(dy.Model(), "rel", False, 64, 64, 0.1, 0.01, False),
    #lambda : SourceBlockPredictor(dy.Model(), "rel", False, 64, 0.3, 0.01, True),
    #lambda : GuessEndv2(dy.Model(), "rel", 64, 64, 0.1, 0.01, True),
    #lambda : GuessEndv2(dy.Model(), "rel", 256, 128, 0.1, 0.01, True),
    # lambda : GuessEndv2(dy.Model(), "rel", 128, 128, 0.1, 0.01, True),
    # lambda : GuessEndv2(dy.Model(), "rel", 128, 128, 0.2, 0.01, True),
    # lambda : GuessEndv2(dy.Model(), "rel", 128, 128, 0.0, 0.01, True),
    # lambda : SourceBlockPredictor(dy.Model(), embed=False),
    # lambda : GuessEndv2(dy.Model(), "rel", 128, 128, 0.0, 0.01, False),
    lambda : FullView(dy.Model(), "rel", False, 64, 128, 0.1, 0.01, True, True, True),
    #lambda : NeuralBaseline(dy.Model(), "rel", False, 128, 0.1, 0.01)
]

models = []
for fun in modelBluePrints:
    mod = fun()
    models.append(mod)

cleanTextReport = []

decorationTypes = ["all"]
for dtype in decorationTypes:
    # open the data files
    filename = "data/train_%s.json" % dtype
    f = open(filename, 'r')
    train = json.loads(f.read())

    filename = "data/dev_%s.json" % dtype
    f = open(filename, 'r')
    dev = json.loads(f.read())

    filename = "data/test_%s.json" % dtype
    f = open(filename, 'r')
    test = json.loads(f.read())

    # for each model, train it and print results on validation set
    for model in models:
        mName = model.getName()

        print("  Model %s..." % mName)

        model.train(train, dev)
        print("--training done.")
        model.predictAll(train, dev, test, dtype)
        print("--predicting done.")

        cleanReport = model.getTextReport(dtype, False)

        print(cleanReport)
        cleanTextReport.append(cleanReport)

        # save the text reports
        modelName = mName
        fname = "output/%s/textReport.txt" % modelName
        check_and_create_path(fname)
        out_file = open(fname, 'w')
        out_file.write("\n".join(cleanTextReport))
        out_file.close()

        fname = "output/%s/epochReport.txt" % modelName
        check_and_create_path(fname)
        out_file = open(fname, 'w')
        out_file.write(model.getTrainReport())
        out_file.close()

        fname = "output/%s/graph.pdf" % modelName
        model.saveTrainGraph(fname)

        sendEmail(mName, cleanReport)

print("==========")
print("\n".join(cleanTextReport))
