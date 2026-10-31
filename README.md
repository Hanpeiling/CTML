# Cross-task multimodal learning model(CTML)

## About
This repository contains the code and resources of the following article:
*Cross-task multimodal learning model for predicting eight sublethal effects associated with 108 effect measurement indicators across multiple fish species**

## Overview of the framework
We developed a novel cross-task multimodal learning (CTML) model integrating molecular structure, species, toxicokinetic, environmental, and endpoint-context features to predict NOEC and LOEC values for eight sublethal effects (biochemistry, development, genetics, growth, histology, hormones, morphology, and reproduction) associated with 108 effect measurement indicators across multiple fish species. The CTML model combined multimodal feature encoding with cross-task information transfer, allowing toxicity knowledge from other sublethal effects of the same chemical to be transferred to the target-effect task.


Please refer to the following steps to run the sublethal effect models

## Setup environment

```bash
pip install -r requirements.txt
```

## Evaluate the CTML model

```bash
python run_ctml.py --evaluate
```


## Train a new CTML model

```bash
python run_ctml.py --train --device auto
```

