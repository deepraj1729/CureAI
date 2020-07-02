from functions.loadData import loadDf
from functions.preprocess import dropCol,handleMissing,normalizeData
from classifiers.models import DNN,CNN1d
from sklearn.model_selection import train_test_split
import numpy as np

pathToDataset = r"dataset/breast_cancer_wisconsindata.csv"
classes = ['benign','malignant']

#Saved model path
model_DNN_path = r"BreastCancerDNN/"
model_CNN_path = r"BreastCancerCNN/"
model_version = "v1"

#Load dataset
data = loadDf(pathToDataset)

#Handle missing data and drop unimportant columns
data = handleMissing(data)
data = dropCol(data,['id'])

#dataframe into X and y values in numpy arrays
y = np.array(data['class'])
X = np.array(dropCol(data,['class']),dtype='float64')

# Normalized data
X_norm = normalizeData(X)

#Splitting data into training and testing set
x_train ,x_test , y_train, y_test = train_test_split(X_norm,y, test_size = 0.1)  #Noralized split for cross-validation
# x_train ,x_test , y_train, y_test = train_test_split(X,y, test_size = 0.1)     #Direct Split for cross-validation


""" models - DNN"""                               #Run any of the two models
## Deep Neural Network 
# model = DNN()

# Train model
# model.trainModel(x_train ,y_train,validation_data=(x_test, y_test),classes = classes ,epochs = 10,batch_size = 10,verbose=1)

# Validate model (in a creative way)
# model.validate(x_test,y_test,classes=classes)

# Plot training vs Validation graph
# model.visualize()

# Load trained DNN model
# model.loadModel(model_DNN_path+model_version)

# Model configurations
# model.model_config()

#Predict model
# model.predict(x_test,classes=classes)


""" models - CNN1d"""
# ## Convolutional 1D Deep Neural Network 
model = CNN1d()

# #Train model
model.trainModel(x_train ,y_train,validation_data=(x_test, y_test),classes = classes ,epochs = 100,batch_size = 10,verbose=1)

# # # Validate model (in a creative way)
# model.validate(x_test,y_test,classes=classes)

# #Plot training vs Validation graph
model.visualize()

# # Load trained CNN model
# model.loadModel(model_CNN_path + model_version)

# #Predict model
# model.predict(x_test,classes=classes)



