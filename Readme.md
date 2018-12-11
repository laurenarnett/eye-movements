# Predicting eye fixation locations

Data from:

N. Wilming, S. Onat, J. Ossandón, A. Acik, T. C. Kietzmann, K. Kaspar, R. R. Gamiero, A. Vormberg, P. König. Data from: An extensive dataset of eye movements during viewing of complex images. https://doi.org/10.5061/dryad.9pf75. Dryad Digital Repository, 2017.


## Preprocessing 

### visualize the fixation locations

python visualization/visualize_fixations.py path/to/data_file path/to/img_dir path/to/output_dir

### preprocess the ground truth labels

python preprocessing/preprocess.py path/to/data_file path/to/img_dir path/to/output_dir


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

python eval.py path/to/output_from_bayes_model.npy path/to/output_from_baseline_model.npy path/to/figure_output
