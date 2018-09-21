# Utilities for graphing
import os
import errno

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

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

# Plot training loss and development accuracy by epoch
def plotChart(X_1, X_2, Y_1, Y_2, showLabels=False, title="", xAxisName="", yAxisName="", widthInches=8, heightInches=8, saveOutput=False, saveDir="", saveName=""):
    plt.clf()
    fig, ax1 = plt.subplots()
    if(saveOutput):
        fig.set_size_inches((widthInches), (heightInches))
    plt.title(title)

    plt.xlabel(xAxisName)
    line_1, = ax1.plot(X_1, Y_1, "C0-", label='Training Loss')
    ax1.set_ylabel('Loss', color='C0')


    ax2 = ax1.twinx()
    line_2, = ax2.plot(X_2, Y_2, "C1-", label='Dev Accuracy')
    ax2.set_ylabel('Accuracy', color='C1')

    plt.xticks(range(0,len(X_1)+1,5))
    plt.legend(handles=[line_1, line_2])
    #plt.plot(X, Y1, "-", X, Y2, "-")#, s=dotSize, c=C, cmap=plt.get_cmap("Set1"))
    #plt.legend()

    fig.tight_layout()

    if saveOutput:
        filename = saveDir + saveName
        check_and_create_path(filename)
        pp = PdfPages(filename)
        pp.savefig()
        pp.close()
    else:
        plt.show()
    plt.close()

if __name__ == '__main__':
    # generate dummy data
    X_1 = []
    X_2 = []
    Y_1 = []
    Y_2 = []

    for i in range(100):
        X_1.append(i)
        Y_1.append(np.random.randint(0, 1000 - i*10))

        if (i % 5 == 0):
            X_2.append(i)
            Y_2.append(np.random.randint(i*10, 1000))


    saveOutput = True
    saveDir = "output/visualizations/"
    plotChart(X_1, X_2, Y_1, Y_2, saveOutput=saveOutput, saveDir=saveDir, saveName="test.pdf", title="Model Accuracy", xAxisName="Epoch", yAxisName="Accuracy", widthInches=11)
