# store our specific models
# parts of this code were adapted from the
# official Dynet MNIST and BiLSTM examples.
import numpy as np
import dynet as dy
import time
import json
from sklearn import svm

from basicModel import Model
import graphNeuralLoss

PRETRAIN_C_EPOCHS = 100
PRETRAIN_L_EPOCHS = 5
PRETRAIN_BOTH_EPOCHS = 100
NUM_EPOCHS = 200
EPOCH_REPORT = 5
ITERATION_REPORT = 2000

MINIBATCH = 1

# vocab size plus UNK
VOCAB_SIZE = 667 + 1

WORLD_STATE_DIM = 40

# Classes shared by neural models
class NeuralModel(Model):
    def __init__(self):
        super().__init__()
        self.trainReport = []
        self.loadModel = False

    # need to include a classifier function with this signature
    #def _classify(self, x, training=False):

    # do a round of training
    def _trainRound(self, trainIndices, train, classifyFunc, lossFunc):
        np.random.shuffle(trainIndices)
        i = 0
        N = len(trainIndices)

        startTime = time.time()

        nextN = ITERATION_REPORT
        loss_sum = []
        while i < N:
            dy.renew_cg()
            start = i
            end = min(i + MINIBATCH, N)

            losses = []
            for j in range(start, end):
                trainPoint = train[j]
                label = trainPoint[-1]
                x = trainPoint

                out = classifyFunc(x, training=True)
                # usually dy.pickneglogsoftmax
                loss = lossFunc(out, label)
                losses.append(loss)
            batchLoss = dy.esum(losses)/MINIBATCH
            batchLoss.backward()
            self.sgd.update()
            loss_sum.append(batchLoss.scalar_value())

            if (i >= nextN):
                nextN += ITERATION_REPORT
                print("%.0f" % (100.0*i/N), end="; ", flush=True)


            i = end
        meanLoss = np.mean(loss_sum)
        endTime = time.time()
        timeDiff = endTime - startTime
        print("\n  Training Error: %f (%.0f seconds)" % (meanLoss, timeDiff), end="; ")
        return meanLoss

    # predict for the given points
    def _predict(self, data, classifyFunc):
        correct = 0
        i = 0
        N = len(data)

        startTime = time.time()

        nextN = ITERATION_REPORT
        while i < N:
            dy.renew_cg()

            start = i
            end = min(i + MINIBATCH, N)

            scores = []
            for j in range(start, end):
                point = data[j]
                label = point[-1]

                x = point
                out = classifyFunc(x)

                scores.append([out, label])

            # evaluate batch at once
            dy.forward([out for out, _ in scores])


            for out, label in scores:
                pred = np.argmax(out.npvalue())
                if pred == label:
                    correct += 1

            i = end
        endTime = time.time()

        acc = 100.0*correct/N
        speed = N/(endTime - startTime)
        res = "%.2f" % acc
        return acc, res

    # given a point, grab flattened versions of the start and end state,
    # with irrelevant y coordinate removed.
    def _getStartEnd(self, point):
        fun = lambda x: [x[0], x[2]]
        ss = list(map(fun, point["startState"]))
        es = list(map(fun, point["endState"]))
        start = np.array(ss)
        start = start.flatten()
        end = np.array(es)
        end = end.flatten()
        return start, end

    # get the feature vector for CNN pretraining
    def _getPretrainCFeatureVector(self, point):
        start, end = self._getStartEnd(point)
        #fv_cnn = np.append(start, end)
        fv_lstm = point["instructionVector"]
        return [start, end, fv_lstm, point["blockMoved"]]

    # get the feature vector for LSTM pretraining
    def _getPretrainLFeatureVector(self, point):
        start, end = self._getStartEnd(point)
        #fv_cnn = np.append(start, end)
        fv_lstm = point["instructionVector"]
        return [start, end, fv_lstm, point["blockMoved"]]


    # get the feature vector for a given point
    # this is the start state, end state, instruction vector,
    # and label for this point
    def getFeatureVector(self, point):
        start, end = self._getStartEnd(point)
        #fv_cnn = np.append(start, end)
        fv_lstm = point["instructionVector"]
        return [start, end, fv_lstm, point["correctEnd"]]

    # if report is true, report on this epoch
    def _epochReport(self, reportEpoch, epoch, train_err, train, dev, classifyFunc, predictFunc, pretrain):
        if (reportEpoch):

            #d_acc, d_str = predictFunc(train, classifyFunc)
            d_acc, d_str = predictFunc(dev, classifyFunc)
            print("  Dev Accuracy: %s" % (d_str))
            res = (
                True,
                epoch,
                train_err,
                d_acc
            )

            if not(pretrain):
                if (self.bestDevAcc == None or d_acc > self.bestDevAcc):
                    if self.bestDevAcc == None:
                        print("%f better than nothin, saving model..." % (d_acc))
                    else:
                        print("%f better than %f, saving model..." % (d_acc, self.bestDevAcc))
                    self.bestDevAcc = d_acc
                    self.saveModel()
        else:
            print("")
            res = (
                False,
                epoch,
                train_err,
                -1
            )
        return res

    def _train(self, train, dev, classifyFunc, lossFunc, predictFunc, pretrain=False, epochs=NUM_EPOCHS):
        trainIndices = list(range(len(train)))

        infoReport = []

        print("Before training:")
        self._epochReport(True, 0, -1, train, dev, classifyFunc, predictFunc, pretrain)

        for epoch in range(epochs):
            reportEpoch = ((epoch + 1) % EPOCH_REPORT == 0) or (epoch == epochs - 1)

            print("Epoch %d: " % (epoch + 1))#, end="")

            train_err = self._trainRound(trainIndices, train, classifyFunc, lossFunc)

            infoReport.append(self._epochReport(reportEpoch, epoch, train_err, train, dev, classifyFunc, predictFunc, pretrain))


        if not(pretrain):
            # load best model
            self.loadBestModel()

            self.trainReport.append(infoReport)
            self.trained = True


    def train(self, in_train, in_dev):
        if self.loadModel:
            self.loadBestModel()
            self.trained = True
            return
        # default loss function
        lossFunc = dy.pickneglogsoftmax
        predFunc = self._predict

        loadPretrained = False#True#
        if loadPretrained:
            self.loadPreprocess()
        else:
            if (self.pretrainC):
                train = []
                for x in in_train:
                    train.append(self._getPretrainCFeatureVector(x))

                dev = []
                for x in in_dev:
                    dev.append(self._getPretrainCFeatureVector(x))

                cfunc = self._classifyPreC
                self._train(train, dev, cfunc, lossFunc, predFunc, pretrain=True, epochs=PRETRAIN_C_EPOCHS)

                print("======== CNN pretraining done.")

            if (self.pretrainL):
                train = []
                for x in in_train:
                    if (x["changeType"] == 0):
                        train.append(self._getPretrainLFeatureVector(x))

                dev = []
                for x in in_dev:
                    if (x["changeType"] == 0):
                        dev.append(self._getPretrainLFeatureVector(x))

                lfunc = self._classifyPreL
                self._train(train, dev, lfunc, lossFunc, predFunc, pretrain=True, epochs=PRETRAIN_L_EPOCHS)

                print("======== LSTM pretraining done.")
            if (self.pretrainC or self.pretrainL):
                self.savePreprocess()

        # train the end state to learn the proper transformation
        if (self.pretrainC and self.pretrainL):
            def prepPoint(x):
                cx = self._getPretrainCFeatureVector(x)
                lx = self._getPretrainLFeatureVector(x)


                cState = self._classifyPreC(cx).npvalue()

                lState = self._classifyPreL(lx).npvalue()

                point = [cState, lState, x["correctEnd"]]
                return point

            print("Calculating data points...")
            train = []
            for i, x in enumerate(in_train):
                train.append(prepPoint(x))

                if i % 500 == 0:
                    print("%d" % (int(i/500)), end=" ", flush=True)
            print("")

            dev = []
            for i, x in enumerate(in_dev):
                dev.append(prepPoint(x))

                if i % 500 == 0:
                    print("%d" % (int(i/500)), end=" ", flush=True)
            print("\nTrain points calculated.")

            lfunc = self._classifyPreBoth
            self._train(train, dev, lfunc, lossFunc, predFunc, pretrain=True, epochs=PRETRAIN_BOTH_EPOCHS)

            print("======== Upper pretraining done.")

        train = []
        for x in in_train:
            train.append(self.getFeatureVector(x))

        dev = []
        for x in in_dev:
            dev.append(self.getFeatureVector(x))

        classifyFunc = self._classify
        self._train(train, dev, classifyFunc, lossFunc, predFunc)


    # print a textual report of the training epochs
    def getTrainReport(self):
        allOut = []
        for tr in self.trainReport:
            out = []
            out.append("")
            out.append("========")
            out.append("Epoch Number: Train Loss, Dev Error")
            bestErr = 0
            bestEpoch = -1
            for info in tr:
                if (info[0]):
                    if (info[3] > bestErr):
                        _, bestEpoch, _, bestErr = info
                    out.append("Epoch %d: %f, %.2f" % (info[1:]))

            out[0] = "Best Epoch: %d (dev error %.2f)" % (bestEpoch, bestErr)
            allOut.extend(out)
        return "\n".join(allOut)

    # write a graph of the training loss and dev accuracy during training.
    def saveTrainGraph(self, fname):
        for i, tr in enumerate(self.trainReport):
            X_1 = []
            Y_1 = []
            X_2 = []
            Y_2 = []

            for hasDev, epoch, train_err, dev_acc in tr:
                X_1.append(epoch)
                Y_1.append(train_err)
                if hasDev:
                    X_2.append(epoch)
                    Y_2.append(dev_acc)

            saveOutput = True
            saveDir = ""
            myFilename = fname[:-4] + ("_%d.pdf" % i)
            graphNeuralLoss.plotChart(X_1, X_2, Y_1, Y_2, saveOutput=saveOutput, saveDir=saveDir, saveName=myFilename, title="Model Performance", xAxisName="Epoch", widthInches=11)


    # predict whether this has the correct end state
    def predict(self, fv):
        dy.renew_cg()

        input = fv
        out = self._classify(input)

        dy.forward([out])

        pred = np.argmax(out.npvalue())

        return pred


#===============================================================================
#===============================================================================
#===============================================================================

# Model that just learns which block moved using an LSTM
class SourceBlockPredictor(NeuralModel):
    # act is activation function; "sig", "rel", "tan"
    # sl is True if we use a second lstm layer
    # ldim is number of base dimensions to use in LSTM
    # noise is noise to apply to embeddings
    # lr is learning rate
    # embed is true if we initialize with word embeddings
    def __init__(self, m, act="rel", sl=False, ldim=64, noise=0.3, lr=0.01, embed=True):
        super().__init__()

        if sl:
            hlString = "_2L"
        else:
            hlString = ""
        if embed:
            embString = "_embed"
        else:
            embString ="_noembed"


        self.name = "SourceBlockPredictor_%s%s_%d_%.2f_%.2f%s" % (act, hlString, ldim, noise, lr, embString)

        self.activation = act
        self.secondLayer = sl
        self.baseLDimensions = ldim
        self.noise = noise
        self.learningRate = lr
        self.pretrainL = False

        self.sgd = dy.SimpleSGDTrainer(m, learning_rate=self.learningRate)


        if (self.activation == "tan"): # tanh
            self.actFun = dy.tanh
        elif (self.activation == "rel"): # ReLU
            self.actFun = dy.rectify
        else: # self.activation == "sig" # sigmoid
            self.actFun = dy.logistic
        # ===== Constants

        LSTM_BASE_DIM = self.baseLDimensions#512

        # LSTM
        self.LSTM_EMBED_DIM = 300 # equal to GloVe
        self.LSTM_HALF_HIDDEN_DIM = LSTM_BASE_DIM
        self.LSTM_HIDDEN_DIM = 2*self.LSTM_HALF_HIDDEN_DIM
        self.LSTM_OUTPUT_DIM = 20 # Number of blocks

        # ===== Parameters

        if (embed):
            f = open("glove/embed.json", 'r')
            arr = np.array(json.loads(f.read()))
            myInit = dy.NumpyInitializer(arr)
        else: # default initialization
            myInit = dy.GlorotInitializer()

        self.LSTM_E = m.add_lookup_parameters((VOCAB_SIZE, self.LSTM_EMBED_DIM), init=myInit)
        self.LSTM_H = m.add_parameters((self.LSTM_OUTPUT_DIM, self.LSTM_HIDDEN_DIM))
        self.LSMT_builders = [
            dy.LSTMBuilder(1, self.LSTM_EMBED_DIM, self.LSTM_HALF_HIDDEN_DIM, m),
            dy.LSTMBuilder(1, self.LSTM_EMBED_DIM, self.LSTM_HALF_HIDDEN_DIM, m),
        ]

        if (self.secondLayer):
            self.LSMT_builders2 = [
                dy.LSTMBuilder(1, self.LSTM_HIDDEN_DIM, self.LSTM_HALF_HIDDEN_DIM, m),
                dy.LSTMBuilder(1, self.LSTM_HIDDEN_DIM, self.LSTM_HALF_HIDDEN_DIM, m),
            ]


        self.paramFname = "models/%s.model" % self.name
        self.paramFnamePreprocess = "models/%s_prep.model" % self.name
        self.bestDevAcc = 0
        self.saveModel = lambda : m.save(self.paramFname)
        self.loadBestModel = lambda : m.populate(self.paramFname)
        self.savePreprocess = lambda : m.save(self.paramFnamePreprocess)
        self.loadPreprocess = lambda : m.populate(self.paramFnamePreprocess)

    def _classify(self, x, training=False):
        l_E = self.LSTM_E
        l_H = dy.parameter(self.LSTM_H)
        right_init, left_init = [builder.initial_state() for builder in self.LSMT_builders]

        if (self.secondLayer):
            right_init2, left_init2 = [builder.initial_state() for builder in self.LSMT_builders2]

        # == LSTM
        l_x = x[0]
        embs = [l_E[w] for w in l_x]
        if (training):
            embs = [dy.noise(we, self.noise) for we in embs]

        # fw = [x.output() for x in right_init.add_inputs(embs)]
        # bw = [x.output() for x in left_init.add_inputs(reversed(embs))]
        fw = right_init.transduce(embs)
        bw = left_init.transduce(reversed(embs))

        fb = dy.concatenate([fw[-1], bw[-1]])
        if (self.secondLayer):
            layer1Output = []
            for i in range(len(fw)):
                bwInd = -1*(i+1)
                piece = dy.concatenate([fw[i], bw[bwInd]])
                layer1Output.append(piece)

            fw2 = right_init2.transduce(layer1Output)
            bw2 = left_init2.transduce(reversed(layer1Output))

            fb = dy.concatenate([fw2[-1], bw2[-1]])
        l_out = l_H * self.actFun(fb)

        out = l_out

        return out

    # do a round of training
    def _trainRound(self, trainIndices, train, classifyFunc, lossFunc):
        np.random.shuffle(trainIndices)
        i = 0
        N = len(trainIndices)

        startTime = time.time()

        nextN = ITERATION_REPORT
        loss_sum = []
        while i < N:
            dy.renew_cg()
            start = i
            end = min(i + MINIBATCH, N)

            losses = []
            for j in range(start, end):
                trainPoint = train[j]
                label = trainPoint[-1]
                x = trainPoint

                out = classifyFunc(x, training=True)
                loss = lossFunc(out, label)
                losses.append(loss)
            batchLoss = dy.esum(losses)/MINIBATCH
            batchLoss.backward()
            self.sgd.update()
            loss_sum.append(batchLoss.scalar_value())

            if (i >= nextN):
                nextN += ITERATION_REPORT
                print("%.0f" % (100.0*i/N), end="; ", flush=True)


            i = end
        meanLoss = np.mean(loss_sum)
        endTime = time.time()
        timeDiff = endTime - startTime
        print("\n  Training Error: %f (%.0f seconds)" % (meanLoss, timeDiff), end="; ")
        return meanLoss

    # predict for the given points
    def _predict(self, data, classifyFunc):
        correct = 0
        i = 0
        N = len(data)

        startTime = time.time()

        nextN = ITERATION_REPORT
        while i < N:
            dy.renew_cg()

            start = i
            end = min(i + MINIBATCH, N)

            scores = []
            for j in range(start, end):
                point = data[j]
                label = point[-1]

                x = point
                out = classifyFunc(x) #dy.softmax(self._classify(x))

                scores.append([out, label])

            # evaluate batch at once
            dy.forward([out for out, _ in scores])


            for out, label in scores:
                pred = np.argmax(out.npvalue())
                # if (i < 5):
                #     print(out.npvalue().tolist())
                #     print(label, pred)
                if pred == label:
                    correct += 1

            # if (i >= nextN):
            #     nextN += ITERATION_REPORT
            #     print("%.0f" % (100.0*i/N), end="; ")

            i = end
        endTime = time.time()

        acc = 100.0*correct/N
        speed = N/(endTime - startTime)
        return acc, ("%f" % acc)

    # get the feature vector for a given point
    # this is the start state, end state, instruction vector,
    # and label for this point
    def getFeatureVector(self, point):
        fv_lstm = point["instructionVector"]
        return [fv_lstm, point["blockMoved"]]

    # load a pretrained model
    def loadPretrainedModel(self):
        self.loadBestModel()
        self.trained = True

    # train predictor
    def train(self, in_train, in_dev):
        preLoad = False
        if preLoad:
            self.loadModel()
            return

        train = []
        for x in in_train:
            if (x["changeType"] == 0):
                train.append(self.getFeatureVector(x))

        dev = []
        for x in in_dev:
            if (x["changeType"] == 0):
                dev.append(self.getFeatureVector(x))

        classifyFunc = self._classify
        lossFunc = dy.pickneglogsoftmax
        predFunc = self._predict
        # train the LSTM portion
        self._train(train, dev, classifyFunc, lossFunc, predFunc, epochs=50)
        print("Best: %f" % self.bestDevAcc)

    # predict that this is the correct end
    def predict(self, fv):
        dy.renew_cg()

        input = fv
        out = self._classify(input)

        dy.forward([out])

        predBlockMoved = np.argmax(out.npvalue())

        return predBlockMoved

#===============================================================================
#===============================================================================
#===============================================================================

# Full View Model
class FullView(NeuralModel):
    # act is activation function; "sig", "rel", "tan"
    # shl is True if we use a second hidden layer
    # cdim is number of base dimensions to use in CNN
    # ldim is number of base dimensions to use in LSTM
    # drop is dropout
    # lr is learning rate
    # embed is true if we initialize with word embeddings
    def __init__(self, m, act="sig", shl=False, cdim=64, ldim=64, drop=0.3, lr=0.01, embed=False, pretrainC=False, pretrainL=False):
        super().__init__()

        if shl:
            hlString = "_2HL"
        else:
            hlString = ""
        if embed:
            embString = "_embed"
        else:
            embString ="_noembed"
        if pretrainC:
            precString = "_pc"
        else:
            precString =""
        if pretrainL:
            prelString = "_pl"
        else:
            prelString =""

        self.name = "FullView2_%s%s_%d_%d_%.2f_%.2f%s%s%s" % (act, hlString, cdim, ldim, drop, lr, embString, precString, prelString)

        self.activation = act
        self.secondHiddenLayer = shl
        self.baseCDimensions = cdim
        self.baseLDimensions = ldim
        self.dropout = drop
        self.learningRate = lr
        self.pretrainC = pretrainC
        self.pretrainL = pretrainL

        self.loadModel = True

        self.sgd = dy.SimpleSGDTrainer(m, learning_rate=self.learningRate)


        if (self.activation == "tan"): # tanh
            self.actFun = dy.tanh
        elif (self.activation == "rel"): # ReLU
            self.actFun = dy.rectify
        else: # self.activation == "sig" # sigmoid
            self.actFun = dy.logistic
        # ===== Constants

        CNN_BASE_DIM = self.baseCDimensions#512
        LSTM_BASE_DIM = self.baseLDimensions#512
        # Worldstate CNN
        self.WCNN_INPUT_DIM = WORLD_STATE_DIM*2
        self.WCNN_HIDDEN_DIM = CNN_BASE_DIM
        self.WCNN_OUTPUT_DIM = CNN_BASE_DIM
        self.WCNN_DROPOUT = self.dropout

        # LSTM
        self.LSTM_EMBED_DIM = 300 # equal to GloVe
        self.LSTM_HALF_HIDDEN_DIM = LSTM_BASE_DIM
        self.LSTM_HIDDEN_DIM = 2*self.LSTM_HALF_HIDDEN_DIM
        self.LSTM_OUTPUT_DIM = LSTM_BASE_DIM

        # Final CNN
        self.FCNN_INPUT_DIM = self.WCNN_OUTPUT_DIM + self.LSTM_OUTPUT_DIM
        self.FCNN_HIDDEN_DIM = 256#CNN_BASE_DIM
        self.FCNN_OUTPUT_DIM = 2
        self.FCNN_DROPOUT = self.dropout

        # ===== Parameters

        self.WCNN_W1 = m.add_parameters((self.WCNN_HIDDEN_DIM, self.WCNN_INPUT_DIM))
        self.WCNN_h1bias = m.add_parameters((self.WCNN_HIDDEN_DIM, ))
        self.WCNN_WF = m.add_parameters((self.WCNN_OUTPUT_DIM, self.WCNN_HIDDEN_DIM))
        if (self.secondHiddenLayer):
            self.WCNN_h2bias = m.add_parameters((self.WCNN_HIDDEN_DIM, ))
            self.WCNN_W2 = m.add_parameters((self.WCNN_HIDDEN_DIM, self.WCNN_HIDDEN_DIM))


        if (embed):
            f = open("glove/embed.json", 'r')
            arr = np.array(json.loads(f.read()))
            myInit = dy.NumpyInitializer(arr)
        else: # default initialization
            myInit = dy.GlorotInitializer()

        self.LSTM_E = m.add_lookup_parameters((VOCAB_SIZE, self.LSTM_EMBED_DIM), init=myInit)
        self.LSTM_H = m.add_parameters((self.LSTM_OUTPUT_DIM, self.LSTM_HIDDEN_DIM))
        self.LSMT_builders = [
            dy.LSTMBuilder(1, self.LSTM_EMBED_DIM, self.LSTM_HALF_HIDDEN_DIM, m),
            dy.LSTMBuilder(1, self.LSTM_EMBED_DIM, self.LSTM_HALF_HIDDEN_DIM, m),
        ]
        if (self.secondHiddenLayer):
            self.LSTM_h2bias = m.add_parameters((self.LSTM_HIDDEN_DIM, ))
            self.LSTM_W2 = m.add_parameters((self.LSTM_HIDDEN_DIM, self.LSTM_HIDDEN_DIM))

        self.FCNN_W1 = m.add_parameters((self.FCNN_HIDDEN_DIM, self.FCNN_INPUT_DIM))
        self.FCNN_h1bias = m.add_parameters((self.FCNN_HIDDEN_DIM, ))
        self.FCNN_WF = m.add_parameters((self.FCNN_OUTPUT_DIM, self.FCNN_HIDDEN_DIM))
        if (self.secondHiddenLayer):
            self.FCNN_h2bias = m.add_parameters((self.FCNN_HIDDEN_DIM, ))
            self.FCNN_W2 = m.add_parameters((self.FCNN_HIDDEN_DIM, self.FCNN_HIDDEN_DIM))

        # these aren't needed anymore but we need them to load the pretrained params.
        if (self.pretrainC):
            self.WCNN_P = m.add_parameters((20, self.WCNN_OUTPUT_DIM))
        if (self.pretrainL):
            self.LSTM_P = m.add_parameters((20, self.LSTM_OUTPUT_DIM))
        if (self.pretrainL and self.pretrainC):
            self.FCNN_P = m.add_parameters((20*20, self.FCNN_HIDDEN_DIM))

        self.paramFname = "models/%s.model" % self.name
        self.paramFnamePreprocess = "models/%s_prep.model" % self.name
        self.bestDevAcc = 0
        self.saveModel = lambda : m.save(self.paramFname)
        self.loadBestModel = lambda : m.populate(self.paramFname)
        self.savePreprocess = lambda : m.save(self.paramFnamePreprocess)
        self.loadPreprocess = lambda : m.populate(self.paramFnamePreprocess)


    def _classifyPreC(self, x, training=False):
        w_W1, w_h1bias, w_WF = dy.parameter(self.WCNN_W1, self.WCNN_h1bias, self.WCNN_WF)

        if (self.secondHiddenLayer):
            w_W2, w_h2bias = dy.parameter(self.WCNN_W2, self.WCNN_h2bias)

        # == CNN
        w_xs = [dy.inputVector(w_x) for w_x in x[:2]]
        w_in = dy.concatenate(w_xs)
        h = self.actFun(w_W1 * w_in + w_h1bias)
        # apply dropout
        if training:
            h = dy.dropout(h, self.WCNN_DROPOUT)
        if (self.secondHiddenLayer):
            h = self.actFun(w_W2 * h + w_h2bias)
        w_out = w_WF * h
        out = w_out


        return out

    def _classifyPreL(self, x, training=False):
        l_E = self.LSTM_E
        l_H = dy.parameter(self.LSTM_H)
        right_init, left_init = [builder.initial_state() for builder in self.LSMT_builders]

        if (self.secondHiddenLayer):
            l_W2, l_h2bias = dy.parameter(self.LSTM_W2, self.LSTM_h2bias)


        # == LSTM
        l_x = x[2]
        embs = [l_E[w] for w in l_x]
        if (training):
            embs = [dy.noise(we, 0.3) for we in embs]

        # fw = [x.output() for x in right_init.add_inputs(embs)]
        # bw = [x.output() for x in left_init.add_inputs(reversed(embs))]
        fw = right_init.transduce(embs)
        bw = left_init.transduce(reversed(embs))


        fb = dy.concatenate([fw[-1], bw[-1]])
        if (self.secondHiddenLayer):
            fb = self.actFun(l_W2 * fb + l_h2bias)
        l_out = self.actFun(l_H * fb)

        out = l_out

        return out

    def _classifyPreBoth(self, x, training=False):
        f_W1, f_h1bias, f_WF = dy.parameter(self.FCNN_W1, self.FCNN_h1bias, self.FCNN_WF)

        if (self.secondHiddenLayer):
            f_W2, f_h2bias = dy.parameter(self.FCNN_W2, self.FCNN_h2bias)

        # == CNN
        f_xs = [dy.inputVector(f_x) for f_x in x[:2]]
        f_in = dy.concatenate(f_xs)
        h = self.actFun(f_W1 * f_in + f_h1bias)

        if (self.secondHiddenLayer):
            h = self.actFun(f_W2 * h + f_h2bias)
        f_out = f_WF * h
        out = f_out

        return out

    def _classify(self, x, training=False):
        w_W1, w_h1bias, w_WF = dy.parameter(self.WCNN_W1, self.WCNN_h1bias, self.WCNN_WF)

        l_E = self.LSTM_E
        l_H = dy.parameter(self.LSTM_H)
        right_init, left_init = [builder.initial_state() for builder in self.LSMT_builders]

        f_W1, f_h1bias, f_WF = dy.parameter(self.FCNN_W1, self.FCNN_h1bias, self.FCNN_WF)

        if (self.secondHiddenLayer):
            w_W2, w_h2bias = dy.parameter(self.WCNN_W2, self.WCNN_h2bias)
            l_W2, l_h2bias = dy.parameter(self.LSTM_W2, self.LSTM_h2bias)
            f_W2, f_h2bias = dy.parameter(self.FCNN_W2, self.FCNN_h2bias)

        # == CNN
        w_xs = [dy.inputVector(w_x) for w_x in x[:2]]
        w_in = dy.concatenate(w_xs)
        h = self.actFun(w_W1 * w_in + w_h1bias)
        # apply dropout
        if training:
            h = dy.dropout(h, self.WCNN_DROPOUT)
        if (self.secondHiddenLayer):
            h = self.actFun(w_W2 * h + w_h2bias)
        w_out = w_WF * h

        # == LSTM
        l_x = x[2]
        embs = [l_E[w] for w in l_x]
        if (training):
            embs = [dy.noise(we, 0.1) for we in embs]

        # fw = [x.output() for x in right_init.add_inputs(embs)]
        # bw = [x.output() for x in left_init.add_inputs(reversed(embs))]
        fw = right_init.transduce(embs)
        bw = left_init.transduce(reversed(embs))


        fb = dy.concatenate([fw[-1], bw[-1]])
        if (self.secondHiddenLayer):
            fb = self.actFun(l_W2 * fb + l_h2bias)
        l_out = self.actFun(l_H * fb)

        # == Combiner
        f_x = dy.concatenate([w_out, l_out])
        # ReLU
        h = self.actFun(f_W1 * f_x + f_h1bias)
        # apply dropout
        if training:
            h = dy.dropout(h, self.FCNN_DROPOUT)

        if (self.secondHiddenLayer):
            h = self.actFun(f_W2 * h + f_h2bias)

        out = f_WF * h

        return out


#===============================================================================
#===============================================================================
#===============================================================================

# Full View Model initialized such that it imitates the MovedLearner model
# initially. Spoiler alert: it immediately forgets everything during training.
class FullViewForcedInitialization(NeuralModel):
    # act is activation function; "sig", "rel", "tan"
    # shl is True if we use a second hidden layer
    # cdim is number of base dimensions to use in CNN
    # ldim is number of base dimensions to use in LSTM
    # drop is dropout
    # lr is learning rate
    # embed is true if we initialize with word embeddings
    def __init__(self, m, act="sig", shl=False, cdim=64, ldim=64, drop=0.3, lr=0.01, embed=False, pretrainC=False, pretrainL=False):
        super().__init__()

        if shl:
            hlString = "_2HL"
        else:
            hlString = ""
        if embed:
            embString = "_embed"
        else:
            embString ="_noembed"
        if pretrainC:
            precString = "_pc"
        else:
            precString =""
        if pretrainL:
            prelString = "_pl"
        else:
            prelString =""

        self.name = "FullView_v2_%s%s_%d_%d_%.2f_%.2f%s%s%s" % (act, hlString, cdim, ldim, drop, lr, embString, precString, prelString)

        self.activation = act
        self.secondHiddenLayer = shl
        self.baseCDimensions = cdim
        self.baseLDimensions = ldim
        self.dropout = drop
        self.learningRate = lr
        self.pretrainC = pretrainC
        self.pretrainL = pretrainL

        #self.loadModel = True

        self.sgd = dy.SimpleSGDTrainer(m, learning_rate=self.learningRate)


        if (self.activation == "tan"): # tanh
            self.actFun = dy.tanh
        elif (self.activation == "rel"): # ReLU
            self.actFun = dy.rectify
        else: # self.activation == "sig" # sigmoid
            self.actFun = dy.logistic
        # ===== Constants

        CNN_BASE_DIM = self.baseCDimensions#512
        LSTM_BASE_DIM = self.baseLDimensions#512
        # Worldstate CNN
        self.WCNN_INPUT_DIM = WORLD_STATE_DIM*2
        self.WCNN_HIDDEN_DIM = CNN_BASE_DIM
        self.WCNN_OUTPUT_DIM = CNN_BASE_DIM
        self.WCNN_DROPOUT = self.dropout

        # LSTM
        self.LSTM_EMBED_DIM = 300 # equal to GloVe
        self.LSTM_HALF_HIDDEN_DIM = LSTM_BASE_DIM
        self.LSTM_HIDDEN_DIM = 2*self.LSTM_HALF_HIDDEN_DIM
        self.LSTM_OUTPUT_DIM = LSTM_BASE_DIM

        # Final CNN
        self.FCNN_INPUT_DIM = self.WCNN_OUTPUT_DIM + self.LSTM_OUTPUT_DIM
        self.FCNN_HIDDEN_DIM = 256#CNN_BASE_DIM
        self.FCNN_OUTPUT_DIM = 2
        self.FCNN_DROPOUT = self.dropout

        # ===== Parameters

        A = np.zeros((self.WCNN_HIDDEN_DIM, self.WCNN_INPUT_DIM))
        for i in range(40):
            A[i][i] = 1
            A[i][i+40] = -1
            A[i+40][i] = - 1
            A[i+40][i+40] = 1
        B = np.zeros((self.WCNN_OUTPUT_DIM, self.WCNN_HIDDEN_DIM))
        for i in range(20):
            B[i][2*i] = 10
            B[i][(2*i)+1] = 10
            B[i][40 + 2*i] = 10
            B[i][40 + (2*i)+1] = 10
        self.WCNN_W1 = m.add_parameters((self.WCNN_HIDDEN_DIM, self.WCNN_INPUT_DIM), init=dy.NumpyInitializer(A))
        self.WCNN_h1bias = m.add_parameters((self.WCNN_HIDDEN_DIM, ), init=dy.ConstInitializer(0))
        self.WCNN_WF = m.add_parameters((self.WCNN_OUTPUT_DIM, self.WCNN_HIDDEN_DIM), init=dy.NumpyInitializer(B))
        if (self.secondHiddenLayer):
            self.WCNN_h2bias = m.add_parameters((self.WCNN_HIDDEN_DIM, ))
            self.WCNN_W2 = m.add_parameters((self.WCNN_HIDDEN_DIM, self.WCNN_HIDDEN_DIM))


        if (embed):
            f = open("glove/embed.json", 'r')
            arr = np.array(json.loads(f.read()))
            myInit = dy.NumpyInitializer(arr)
        else: # default initialization
            myInit = dy.GlorotInitializer()

        self.LSTM_E = m.add_lookup_parameters((VOCAB_SIZE, self.LSTM_EMBED_DIM), init=myInit)
        self.LSTM_H = m.add_parameters((self.LSTM_OUTPUT_DIM, self.LSTM_HIDDEN_DIM))
        self.LSMT_builders = [
            dy.LSTMBuilder(1, self.LSTM_EMBED_DIM, self.LSTM_HALF_HIDDEN_DIM, m),
            dy.LSTMBuilder(1, self.LSTM_EMBED_DIM, self.LSTM_HALF_HIDDEN_DIM, m),
        ]
        if (self.secondHiddenLayer):
            self.LSTM_h2bias = m.add_parameters((self.LSTM_HIDDEN_DIM, ))
            self.LSTM_W2 = m.add_parameters((self.LSTM_HIDDEN_DIM, self.LSTM_HIDDEN_DIM))


        C = np.zeros((self.FCNN_HIDDEN_DIM, self.FCNN_INPUT_DIM))
        for i in range(20):
            C[i][i] = 1
            C[i][self.WCNN_OUTPUT_DIM+i] = 1
        D = np.zeros((self.FCNN_OUTPUT_DIM, self.FCNN_HIDDEN_DIM))
        for i in range(20):
            D[1][i] = 1
        self.FCNN_W1 = m.add_parameters((self.FCNN_HIDDEN_DIM, self.FCNN_INPUT_DIM), init=dy.NumpyInitializer(C))
        self.FCNN_h1bias = m.add_parameters((self.FCNN_HIDDEN_DIM, ), init=dy.ConstInitializer(-1))
        self.FCNN_WF = m.add_parameters((self.FCNN_OUTPUT_DIM, self.FCNN_HIDDEN_DIM), init=dy.NumpyInitializer(D))
        self.FCNN_hfbias = m.add_parameters((self.FCNN_OUTPUT_DIM, ), init=dy.NumpyInitializer(np.array([0.3, 0])))
        if (self.secondHiddenLayer):
            self.FCNN_h2bias = m.add_parameters((self.FCNN_HIDDEN_DIM, ))
            self.FCNN_W2 = m.add_parameters((self.FCNN_HIDDEN_DIM, self.FCNN_HIDDEN_DIM))

        self.paramFname = "models/%s.model" % self.name
        self.paramFnamePreprocess = "models/%s_prep.model" % self.name
        self.bestDevAcc = 0
        self.saveModel = lambda : m.save(self.paramFname)
        self.loadBestModel = lambda : m.populate(self.paramFname)
        self.savePreprocess = lambda : m.save(self.paramFnamePreprocess)
        self.loadPreprocess = lambda : m.populate(self.paramFnamePreprocess)


    def _classifyPreC(self, x, training=False):
        w_W1, w_h1bias, w_WF = dy.parameter(self.WCNN_W1, self.WCNN_h1bias, self.WCNN_WF)

        if (self.secondHiddenLayer):
            w_W2, w_h2bias = dy.parameter(self.WCNN_W2, self.WCNN_h2bias)

        # == CNN
        w_xs = [dy.inputVector(w_x) for w_x in x[:2]]
        w_in = dy.concatenate(w_xs)
        h = self.actFun(w_W1 * w_in + w_h1bias)
        # apply dropout
        if training:
            h = dy.dropout(h, self.WCNN_DROPOUT)
        if (self.secondHiddenLayer):
            h = self.actFun(w_W2 * h + w_h2bias)
        w_out = dy.tanh(w_WF * h)
        out = w_out

        return out

    def _classifyPreL(self, x, training=False):
        l_E = self.LSTM_E
        l_H = dy.parameter(self.LSTM_H)
        right_init, left_init = [builder.initial_state() for builder in self.LSMT_builders]

        if (self.secondHiddenLayer):
            l_W2, l_h2bias = dy.parameter(self.LSTM_W2, self.LSTM_h2bias)


        # == LSTM
        l_x = x[2]
        embs = [l_E[w] for w in l_x]
        if (training):
            embs = [dy.noise(we, 0.3) for we in embs]

        # fw = [x.output() for x in right_init.add_inputs(embs)]
        # bw = [x.output() for x in left_init.add_inputs(reversed(embs))]
        fw = right_init.transduce(embs)
        bw = left_init.transduce(reversed(embs))


        fb = dy.concatenate([fw[-1], bw[-1]])
        if (self.secondHiddenLayer):
            fb = self.actFun(l_W2 * fb + l_h2bias)
        l_out = self.actFun(l_H * fb)

        out = l_out

        return out

    def _classifyPreBoth(self, x, training=False):
        f_W1, f_h1bias, f_WF, f_hfbias = dy.parameter(self.FCNN_W1, self.FCNN_h1bias, self.FCNN_WF, self.FCNN_hfbias)

        if (self.secondHiddenLayer):
            f_W2, f_h2bias = dy.parameter(self.FCNN_W2, self.FCNN_h2bias)

        # == CNN
        f_xs = [dy.inputVector(x[0]), dy.softmax(dy.inputVector(x[1]))]
        f_in = dy.concatenate(f_xs)
        h = self.actFun(f_W1 * f_in + f_h1bias)
        # apply dropout
        # if training:
        #     h = dy.dropout(h, self.WCNN_DROPOUT)
        if (self.secondHiddenLayer):
            h = self.actFun(f_W2 * h + f_h2bias)
        f_out = f_WF * h + f_hfbias
        out = f_out

        return out

    def _classify(self, x, training=False):
        w_W1, w_h1bias, w_WF = dy.parameter(self.WCNN_W1, self.WCNN_h1bias, self.WCNN_WF)

        l_E = self.LSTM_E
        l_H = dy.parameter(self.LSTM_H)
        right_init, left_init = [builder.initial_state() for builder in self.LSMT_builders]

        f_W1, f_h1bias, f_WF, f_hfbias = dy.parameter(self.FCNN_W1, self.FCNN_h1bias, self.FCNN_WF, self.FCNN_hfbias)

        if (self.secondHiddenLayer):
            w_W2, w_h2bias = dy.parameter(self.WCNN_W2, self.WCNN_h2bias)
            l_W2, l_h2bias = dy.parameter(self.LSTM_W2, self.LSTM_h2bias)
            f_W2, f_h2bias = dy.parameter(self.FCNN_W2, self.FCNN_h2bias)

        # == CNN
        w_xs = [dy.inputVector(w_x) for w_x in x[:2]]
        w_in = dy.concatenate(w_xs)
        h = self.actFun(w_W1 * w_in + w_h1bias)
        # apply dropout
        if training:
            h = dy.dropout(h, self.WCNN_DROPOUT)
        if (self.secondHiddenLayer):
            h = self.actFun(w_W2 * h + w_h2bias)
        w_out = dy.tanh(w_WF * h)

        # == LSTM
        l_x = x[2]
        embs = [l_E[w] for w in l_x]
        if (training):
            embs = [dy.noise(we, 0.1) for we in embs]

        # fw = [x.output() for x in right_init.add_inputs(embs)]
        # bw = [x.output() for x in left_init.add_inputs(reversed(embs))]
        fw = right_init.transduce(embs)
        bw = left_init.transduce(reversed(embs))


        fb = dy.concatenate([fw[-1], bw[-1]])
        if (self.secondHiddenLayer):
            fb = self.actFun(l_W2 * fb + l_h2bias)
        l_out = self.actFun(l_H * fb)

        # == Combiner
        f_x = dy.concatenate([w_out, dy.softmax(l_out)])
        # ReLU
        h = self.actFun(f_W1 * f_x + f_h1bias)
        # apply dropout
        if training:
            h = dy.dropout(h, self.FCNN_DROPOUT)

        if (self.secondHiddenLayer):
            h = self.actFun(f_W2 * h + f_h2bias)

        out = f_WF * h + f_hfbias

        return out

#===============================================================================
#===============================================================================
#===============================================================================

# Guess End v2, based on Piˇsl and Mareˇcek 2017
class GuessEndv2(NeuralModel):
    # act is activation function; "sig", "rel", "tan"
    # shl is True if we use a second hidden layer
    # cdim is number of base dimensions to use in CNN
    # ldim is number of base dimensions to use in LSTM
    # drop is dropout
    # lr is learning rate
    # embed is true if we initialize with word embeddings
    def __init__(self, m, act="sig", ldim=64, l2dim=64, drop=0.3, lr=0.01, embed=False):
        super().__init__()

        if embed:
            embString = "_embed"
        else:
            embString ="_noembed"


        self.name = "GuessEndv2_%s_%d_%d_%.2f_%.2f%s" % (act, ldim, l2dim, drop, lr, embString)

        self.activation = act
        self.secondHiddenLayer = False
        self.baseLDimensions = ldim
        self.baseLDimensions2= l2dim
        self.dropout = drop
        self.learningRate = lr
        self.pretrainL = False
        self.embed = embed

        self.sgd = dy.SimpleSGDTrainer(m, learning_rate=self.learningRate)


        if (self.activation == "tan"): # tanh
            self.actFun = dy.tanh
        elif (self.activation == "rel"): # ReLU
            self.actFun = dy.rectify
        else: # self.activation == "sig" # sigmoid
            self.actFun = dy.logistic
        # ===== Constants


        LSTM_BASE_DIM = self.baseLDimensions#512
        LSTM_BASE_DIM2 = self.baseLDimensions2
        NUM_BLOCKS = 20

        self.LSTM_EMBED_DIM = 300 # equal to GloVe


        self.blockPredictor = None

        # LSTM
        self.LSTM_HALF_HIDDEN_DIM = LSTM_BASE_DIM
        self.LSTM_HIDDEN_DIM = 2*self.LSTM_HALF_HIDDEN_DIM
        self.LSTM_OUTPUT_DIM = self.LSTM_HIDDEN_DIM

        # Reference block
        self.RB_LSTM_INPUT_DIM = self.LSTM_OUTPUT_DIM
        self.RB_LSTM_HALF_HIDDEN_DIM = LSTM_BASE_DIM2
        self.RB_LSTM_HIDDEN_DIM = 2*self.RB_LSTM_HALF_HIDDEN_DIM
        self.RB_LSTM_OUTPUT_DIM = NUM_BLOCKS

        # Relative position to reference
        self.RP_LSTM_INPUT_DIM = self.LSTM_OUTPUT_DIM
        self.RP_LSTM_HALF_HIDDEN_DIM = LSTM_BASE_DIM2
        self.RP_LSTM_HIDDEN_DIM = 2*self.RP_LSTM_HALF_HIDDEN_DIM
        self.RP_LSTM_OUTPUT_DIM = 2

        # ===== Parameters

        if (embed):
            f = open("glove/embed.json", 'r')
            arr = np.array(json.loads(f.read()))
            myInit = dy.NumpyInitializer(arr)
        else: # default initialization
            myInit = dy.GlorotInitializer()

        # Instruction representation LSTM
        self.LSTM_E = m.add_lookup_parameters((VOCAB_SIZE, self.LSTM_EMBED_DIM), init=myInit)
        self.LSMT_builders = [
            dy.LSTMBuilder(1, self.LSTM_EMBED_DIM, self.LSTM_HALF_HIDDEN_DIM, m),
            dy.LSTMBuilder(1, self.LSTM_EMBED_DIM, self.LSTM_HALF_HIDDEN_DIM, m),
        ]

        # Reference Block
        self.RB_LSTM_H = m.add_parameters((self.RB_LSTM_OUTPUT_DIM, self.RB_LSTM_HIDDEN_DIM))
        self.RB_LSMT_builders = [
            dy.LSTMBuilder(1, self.RB_LSTM_INPUT_DIM, self.RB_LSTM_HALF_HIDDEN_DIM, m),
            dy.LSTMBuilder(1, self.RB_LSTM_INPUT_DIM, self.RB_LSTM_HALF_HIDDEN_DIM, m),
        ]


        # Relative Position  LSTM
        self.RP_LSTM_H = m.add_parameters((self.RP_LSTM_OUTPUT_DIM, self.RP_LSTM_HIDDEN_DIM))
        self.RP_LSMT_builders = [
            dy.LSTMBuilder(1, self.RP_LSTM_INPUT_DIM, self.RP_LSTM_HALF_HIDDEN_DIM, m),
            dy.LSTMBuilder(1, self.RP_LSTM_INPUT_DIM, self.RP_LSTM_HALF_HIDDEN_DIM, m),
        ]

        self.paramFname = "models/%s.model" % self.name
        self.paramFnamePreprocess = "models/%s_prep.model" % self.name
        self.bestDevAcc = None
        self.saveModel = lambda : m.save(self.paramFname)
        self.loadBestModel = lambda : m.populate(self.paramFname)
        self.savePreprocess = lambda : m.save(self.paramFnamePreprocess)
        self.loadPreprocess = lambda : m.populate(self.paramFnamePreprocess)

    # Movement of Block
    def _classifyPhase2(self, x, training=False):
        l_E = self.LSTM_E
        right_init, left_init = [builder.initial_state() for builder in self.LSMT_builders]

        rb_H = dy.parameter(self.RB_LSTM_H)
        rb_right_init, rb_left_init = [builder.initial_state() for builder in self.RB_LSMT_builders]

        rp_H = dy.parameter(self.RP_LSTM_H)
        rp_right_init, rp_left_init = [builder.initial_state() for builder in self.RP_LSMT_builders]


        # == Instruction Representation  LSTM
        l_x = x[2]
        embs = [l_E[w] for w in l_x]
        if (training):
            embs = [dy.noise(we, 0.1) for we in embs]

        fw = right_init.transduce(embs)
        bw = left_init.transduce(reversed(embs))

        instruction_rep = []
        for i in range(len(fw)):
            bwInd = -1*(i+1)
            piece = dy.concatenate([fw[i], bw[bwInd]])
            instruction_rep.append(piece)



        # == Reference Position LSTM
        fw = rb_right_init.transduce(instruction_rep)
        bw = rb_left_init.transduce(reversed(instruction_rep))

        fb = dy.concatenate([fw[-1], bw[-1]])
        rb_act = rb_H * self.actFun(fb)
        rb_weights = dy.sparsemax(rb_act)

        # grab state from in_state
        in_state = dy.inputTensor(x[1])
        rb_out = in_state*rb_weights

        # == Relative Position LSTM
        fw = rp_right_init.transduce(instruction_rep)
        bw = rp_left_init.transduce(reversed(instruction_rep))

        fb = dy.concatenate([fw[-1], bw[-1]])
        rp_out = rp_H * self.actFun(fb)

        out = rb_out + rp_out

        return out

    # get the block that was moved
    def _getBlockMoved(self, start, end):
        moved = -1
        for i in range(len(start)):
            if not(start[i] == end[i]):
                moved = int(i/2)
                break
        return moved

    # predict for the given points for phase 1
    def _predictAllPhase2(self, data, classifyFunc):
        correctBlocks = 0
        sumDistance = 0
        summaryOutput = []
        i = 0
        N = len(data)

        startTime = time.time()

        dists = np.zeros(N)

        nextN = ITERATION_REPORT
        while i < N:
            dy.renew_cg()

            start = i
            end = min(i + MINIBATCH, N)

            scores = []
            for j in range(start, end):
                point = data[j]
                label = point[-1]

                x = point
                out = classifyFunc(x) #dy.softmax(self._classify(x))

                scores.append([out, label, point])

            # evaluate batch at once
            dy.forward([out for out, _, _ in scores])


            for out, label, point in scores:

                endBlockLoc = label
                predBlockLoc = out.npvalue()
                dist = np.sqrt(np.sum(np.square(predBlockLoc-endBlockLoc)))

                sumDistance += dist
                dists[i] = dist

                if i < 5:
                    summaryOutput.append("GuessEnd: (%.4f, %.4f), ActualEnd: (%.4f, %.4f)" % (predBlockLoc[0], predBlockLoc[1], endBlockLoc[0], endBlockLoc[1]))

            i = end
        endTime = time.time()

        avgDist = np.mean(dists)
        medianDist = np.median(dists)

        speed = N/(endTime - startTime)
        score = (-1*(avgDist+1000*medianDist))
        res = "Average Distance: %.4f, Median Distance: %.4f, Score: %.4f" % (avgDist, medianDist, score)

        res = [res]
        res.extend(summaryOutput)
        res = "\n".join(res)
        return score, res

    # predict for phase 1
    def _predictPhase1(self, fv):
        return self.blockPredictor.predict(fv)

    # predict for phase 2
    def _predictPhase2(self, fv):
        dy.renew_cg()

        input = fv
        out = self._classifyPhase2(input)

        dy.forward([out])

        pred = out.npvalue()

        return pred

    # properly reshape the start array
    def _reshapeStart(self, start):
        return start.reshape(20,2).T

    # get feature vector for phase 2
    def _getPhase1FeatureVector(self, point):
        moved = point["blockMoved"]
        fv_lstm = point["instructionVector"]
        return [fv_lstm, moved]

    # get feature vector for phase 2
    def _getPhase2FeatureVector(self, point):
        start, end = self._getStartEnd(point)
        #fv_cnn = np.append(start, end)
        fv_lstm = point["instructionVector"]
        movedGuess = point["blockMoved"]
        #diff = end-start
        target = end[movedGuess*2:movedGuess*2+2]
        properStart = self._reshapeStart(start)
        return [movedGuess, properStart, fv_lstm, target]

    # get feature vector for phase 3
    def _getPhase3FeatureVector(self, point):
        start, end = self._getStartEnd(point)
        fv_lstm = point["instructionVector"]
        movedGuess = self._predictPhase1([fv_lstm])
        properStart = self._reshapeStart(start)
        movedEndLocation = self._predictPhase2([movedGuess, properStart, fv_lstm, end])
        change = np.zeros(start.shape)
        change[movedGuess*2] = movedEndLocation[0]
        change[movedGuess*2+1] = movedEndLocation[1]
        endGuess = start + change
        return [np.abs(endGuess-end), point["correctEnd"]]

    def train(self, in_train, in_dev):
        # Load a source block predictor
        self.blockPredictor = SourceBlockPredictor(dy.Model(), embed=self.embed)
        self.blockPredictor.loadPretrainedModel()

        predFunc = self._predict
        loadPretrained = True#False#
        if loadPretrained:
            self.loadPreprocess()
            print("Calculating data points...", end=" ")
            train = []
            for i, x in enumerate(in_train):
                if (x["changeType"] == 0):
                    train.append(self._getPhase2FeatureVector(x))

                    if i % 500 == 0:
                        print("%d" % (int(i/500)), end=" ", flush=True)

            dev = []
            for i, x in enumerate(in_dev):
                if (x["changeType"] == 0):
                    dev.append(self._getPhase2FeatureVector(x))

                    if i % 500 == 0:
                        print("%d" % (int(i/500)), end=" ", flush=True)
            print("")
            classifyFunc = self._classifyPhase2
            val,str = self._predictAllPhase2(train, classifyFunc)
            print("Train", val, str)
            val,str = self._predictAllPhase2(dev, classifyFunc)
            print("Dev", val, str)
        else:
            # train predictor
            print("Calculating data points...", end=" ")
            train = []
            for i, x in enumerate(in_train):
                if (x["changeType"] == 0):
                    train.append(self._getPhase2FeatureVector(x))

                    if i % 500 == 0:
                        print("%d" % (int(i/500)), end=" ", flush=True)

            dev = []
            for i, x in enumerate(in_dev):
                if (x["changeType"] == 0):
                    dev.append(self._getPhase2FeatureVector(x))

                    if i % 500 == 0:
                        print("%d" % (int(i/500)), end=" ", flush=True)
            print("")

            classifyFunc = self._classifyPhase2
            lossFunc = lambda pred, label: dy.l1_distance(pred, dy.inputVector(label))#dy.squared_distance(pred, dy.inputVector(label))
            predFunc2 = self._predictAllPhase2
            self.bestDevAcc = None
            self._train(train, dev, classifyFunc, lossFunc, predFunc2, epochs=200) # 500

            print("======== Phase 2 training done.")
            val,str = self._predictAllPhase2(train, classifyFunc)
            print("Train", val, str)
            val,str = self._predictAllPhase2(dev, classifyFunc)
            print("Dev", val, str)
            print("========")

            self.savePreprocess()

        for x in in_dev[-11:-1]:
            start, end = self._getStartEnd(x)
            movedActual = x["blockMoved"]
            instruction = x["instructionVector"]
            movedGuess = self._predictPhase1([instruction])
            print("GuessMoved: %d, ActualMoved: %d" % (movedGuess, movedActual))
            properStart = self._reshapeStart(start)
            movedEndLocation = self._predictPhase2([movedGuess, properStart, instruction, end])
            actualEndLocation = end[2*movedGuess:2*movedGuess+2]
            print("GuessEnd: (%.4f, %.4f), ActualEnd: (%.4f, %.4f)" % (movedEndLocation[0], movedEndLocation[1], actualEndLocation[0], actualEndLocation[1]))


        # train differentiator
        print("Calculating data points...", end=" ")
        train = []
        trainY = []
        for i, p in enumerate(in_train):
            x, y = self._getPhase3FeatureVector(p)
            train.append(x)
            trainY.append(y)

            if i % 500 == 0:
                print("%d" % (int(i/500)), end=" ", flush=True)

        dev = []
        devY = []
        for i, p in enumerate(in_dev):
            x, y = self._getPhase3FeatureVector(p)
            dev.append(x)
            devY.append(y)

            if i % 500 == 0:
                print("%d" % (int(i/500)), end=" ", flush=True)
        train = np.array(train)
        trainY = np.array(trainY)
        dev = np.array(dev)
        devY = np.array(devY)
        print("points calculated.")

        print("Optimizing SVM")
        bestAcc = 0
        bestParams = [None, None]
        cs = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
        for penalty in ["l1", "l2"]:
            for c in cs:
                if penalty=="l1":
                    tempSVM = svm.LinearSVC(penalty=penalty, C=c, dual=False)
                else:
                    tempSVM = svm.LinearSVC(penalty=penalty, C=c)
                tempSVM.fit(train, trainY)

                devPreds = tempSVM.predict(dev)
                devAcc = 100.0*np.sum(devPreds == devY)/len(devPreds)
                print("  Dev Accuracy (penalty=%s, c=%f): %f" % (penalty, c, devAcc))

                if devAcc > bestAcc:
                    bestAcc = devAcc
                    bestParams = [penalty, c]
        bestPen = bestParams[0]
        bestC = bestParams[1]
        print("Best model: penalty=%s, c=%f" % (bestPen, bestC))


        if bestPen=="l1":
            self.svm = svm.LinearSVC(penalty=bestPen, C=bestC, dual=False)
        else:
            self.svm = svm.LinearSVC(penalty=bestPen, C=bestC)
        self.svm.fit(train, trainY)

        devPreds = self.svm.predict(dev)
        devAcc = 100.0*np.sum(devPreds == devY)/len(devPreds)
        print("  Dev Accuracy: %f" % (devAcc))

        self.trained = True

    # predict that this is the correct end
    def predict(self, fv):
        dy.renew_cg()

        # phase 1 prediction of block moved
        instruction = fv[2]
        movedGuess = self._predictPhase1([instruction])
        # get phase 2 prediction of final location
        start = fv[0]

        properStart = self._reshapeStart(start)
        movedEndLocation = self._predictPhase2([movedGuess, properStart, instruction])
        change = np.zeros(start.shape)
        change[movedGuess*2] = movedEndLocation[0]
        change[movedGuess*2+1] = movedEndLocation[1]
        endGuess = start + change

        # phase 3: use SVM to predict whether this is the correct end or not.
        end = fv[1]

        x = np.abs(endGuess-end)
        pred = self.svm.predict([x])

        return pred



#===============================================================================
#===============================================================================
#===============================================================================

# Model that just learns which block moved using an LSTM
class MovedLearner(NeuralModel):
    def __init__(self):
        super().__init__()
        self.blockPredictor = None
        self.name = "MovedLearner"


    def train(self, in_train, in_dev):
        self.blockPredictor = SourceBlockPredictor(dy.Model())
        self.blockPredictor.loadPretrainedModel()
        self.trained = True

    # get the block that was moved
    def _getBlockMoved(self, point):
        start, end, _, _ = point
        moved = -1
        for i in range(len(start)):
            if not(start[i] == end[i]):
                moved = int(i/2)
                break
        return moved

    # predict that this is the correct end
    def predict(self, fv):
        instruction = fv[2]
        predBlockMoved = self.blockPredictor.predict([instruction, None])

        blockMoved = self._getBlockMoved(fv)

        if (predBlockMoved == blockMoved):
            return 1
        else:
            return 0


#===============================================================================
#===============================================================================
#===============================================================================

# Model that uses only the start and end state
class NeuralBaseline(NeuralModel):
    # act is activation function; "sig", "rel", "tan"
    # shl is True if we use a second hidden layer
    # cdim is number of base dimensions to use in CNN
    # drop is dropout
    # lr is learning rate
    # embed is true if we initialize with word embeddings
    def __init__(self, m, act="sig", shl=False, cdim=64, drop=0.3, lr=0.01):
        super().__init__()

        if shl:
            hlString = "_2HL"
        else:
            hlString = ""

        self.name = "NeuralBaseline2_%s%s_%d_%.2f_%.2f" % (act, hlString, cdim, drop, lr)

        self.activation = act
        self.secondHiddenLayer = shl
        self.baseCDimensions = cdim
        self.dropout = drop
        self.learningRate = lr
        self.pretrainC = False
        self.pretrainL = False

        self.loadModel = True

        self.sgd = dy.SimpleSGDTrainer(m, learning_rate=self.learningRate)


        if (self.activation == "tan"): # tanh
            self.actFun = dy.tanh
        elif (self.activation == "rel"): # ReLU
            self.actFun = dy.rectify
        else: # self.activation == "sig" # sigmoid
            self.actFun = dy.logistic
        # ===== Constants

        CNN_BASE_DIM = self.baseCDimensions#512
        # Worldstate CNN
        self.WCNN_INPUT_DIM = WORLD_STATE_DIM*2
        self.WCNN_HIDDEN_DIM = CNN_BASE_DIM*2
        self.WCNN_OUTPUT_DIM = 2
        self.WCNN_DROPOUT = self.dropout

        # ===== Parameters

        self.WCNN_W1 = m.add_parameters((self.WCNN_HIDDEN_DIM, self.WCNN_INPUT_DIM))
        self.WCNN_h1bias = m.add_parameters((self.WCNN_HIDDEN_DIM, ))
        self.WCNN_WF = m.add_parameters((self.WCNN_OUTPUT_DIM, self.WCNN_HIDDEN_DIM))
        # if (self.secondHiddenLayer):
        #     self.WCNN_h2bias = m.add_parameters((self.WCNN_HIDDEN_DIM, ))
        #     self.WCNN_W2 = m.add_parameters((self.WCNN_HIDDEN_DIM, self.WCNN_HIDDEN_DIM))

        self.paramFname = "models/%s.model" % self.name
        self.bestDevAcc = 0
        self.saveModel = lambda : m.save(self.paramFname)
        self.loadBestModel = lambda : m.populate(self.paramFname)


    def _classify(self, x, training=False):
        w_W1, w_h1bias, w_WF = dy.parameter(self.WCNN_W1, self.WCNN_h1bias, self.WCNN_WF)

        # if (self.secondHiddenLayer):
        #     w_W2, w_h2bias = dy.parameter(self.WCNN_W2, self.WCNN_h2bias)

        # == CNN
        w_xs = [dy.inputVector(w_x) for w_x in x[:2]]
        w_in = dy.concatenate(w_xs)
        h = self.actFun(w_W1 * w_in + w_h1bias)
        # apply dropout
        if training:
            h = dy.dropout(h, self.WCNN_DROPOUT)
        if (self.secondHiddenLayer):
            h = self.actFun(w_W2 * h + w_h2bias)
        out = w_WF * h

        return out
