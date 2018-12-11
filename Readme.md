# Predicting eye fixation

## Preprocessing 

## Learning and visualization

### train the baseline model

python train_basel.py

### visualizing the prediction of baseline model

First define the path for the baseline model to load.

python visualize_eyemove_pred.py

### train the prior on ImageNet

python train_prior2.py

The prior model checkpoint will automatically saved as prior2nmodel_best.pth.tar

### visualizing the prediction of trained prior mask
First define the path for the saved prior mask model

python visualize_prior2.py

### train the bayesian model of incoporating the mask

python train_incorporate.py

## Evaluation

