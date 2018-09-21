# Verifying Correct Completion of Natural Language Instructions

## About

This was my final research project for an NLP research seminar at Cornell during the Spring 2018. It examined taking natural language instructions in a block environment and determining whether a given start and end state represented correct completion of the instruction. My hypothesized approach did not work better than previous approaches, which is how research goes when you have to stop after 2 months instead of trying something else. It was still a really fun project to work on and I got to practice with various neural architectures as well. See the final report, `report.pdf`, for more details.

## Files

The `data/` folder should include trainset.json, devset.json, and testset.json from [Bisk et al.'s blocks dataset](https://groundedlanguage.github.io/).

the `glove/` folder should have glove.840B.300d.txt (GloVe trained on 840B words with 300 dimensions) in it.

Create a set of preprocessed data with `processData.py`, which will be stored in the `data/` folder, along with embeddings which will be stored in the `glove/` folder.

Run the full suite of models with  `python3 evaluateModels.py`.

Run a model or two (and get an email when each finishes) with `python3 evaluateSingleModel.py`.

Results will be stored in `output/`, with test results included only in `output/restricted/`.

To view the superclass the models and all our non-neural baselines, see `basicModel.py`. For the neural models and baselines, see `model.py`.

Saved model parameters can be found in `models/`.
