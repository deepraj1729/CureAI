# CureAI
Applications of ML in Medical Sciences

## 1. Breast Cancer Classifier
- Dataset: Breast_cancer_wisconsindata.csv
- Features: id, clump_thickness, unif_cell_size, unif_cell_shape, marg_adhesion, single_epith_cell_size, bare_nuclei, bland_chrom, norm_nucleoli, mitoses, class
- Initial Shape = (699,11)
- Trainable shape = (699,9)
- Output = 2 [0,1] or [1,0]
- training-testing split = (0.2)

## Models (Type = Sequential):
- Deep Neural Network, DNN (4 layers)
- Convolutional 1D Deep Neural Network (Conv1D, 2 convolution,1 maxpooling, 1 flatten, 2 Dense layers)

# Training and Validation graphs per epochs:

### DNN Model:

<img src="https://github.com/deepraj1729/CureAI/blob/master/training_graphs/DNN_30.png" width = "200" height = "100"> <img src="https://github.com/deepraj1729/CureAI/blob/master/training_graphs/DNN_50.png" width = "200" height = "100"> <img src="https://github.com/deepraj1729/CureAI/blob/master/training_graphs/DNN_100.png" width = "200" height = "100"> 
 
### CNN1d Model:

<img src="https://github.com/deepraj1729/CureAI/blob/master/training_graphs/CNN1d_30.png" width = "200" height = "100"> <img src="https://github.com/deepraj1729/CureAI/blob/master/training_graphs/CNN1d_50.png" width = "200" height = "100"> <img src="https://github.com/deepraj1729/CureAI/blob/master/training_graphs/CNN1d_100.png" width = "200" height = "100"> 

### Conclusion:
Clearly, the dataset is small for reducing the validation loss close to training loss, yet, the `Conv1d` model trained better than the `DNN model` for the most part
