import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}
from tensorflow.python.util import deprecation
#Avoid Deprecation warnings
deprecation._PRINT_DEPRECATION_WARNINGS = False
import tensorflow as tf 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, BatchNormalization, Flatten, Conv1D, AveragePooling1D, MaxPooling1D, Dropout
import numpy as np
from prettytable import PrettyTable
from sklearn import model_selection
import time
import matplotlib.pyplot as plt

saved_model_path = "./saved_model/"

class DNN:
    def __init__(self):
        self.model = None
        self.classes = []
        self.modelHistory = []
        self.epochs = 0
        print("\nDeep Neural Network Model initialized......")
    
    def model_config(self,model_path): 
        self.model = tf.keras.models.load_model(model_path)
        self.model.summary()

    def loadModel(self,model_path):
        self.model = tf.keras.models.load_model(model_path)

    def trainModel(self,x_train,y_train,validation_data=(None,None),classes=[],epochs = 50,batch_size = 10,verbose = 0):
        disp = PrettyTable(['Training Breast Cancer ANN model.....'])
        
        x_test,y_test = validation_data
        self.classes = classes
        self.epochs = epochs
        inp_shape = len(x_train[0])

        self.model = Sequential()
        self.model.add(Flatten(input_shape=(inp_shape,)))
        self.model.add(Dense(300,activation = "relu"))
        self.model.add(Dense(200,activation = "relu"))
        self.model.add(Dense(40,activation = "relu"))
        self.model.add(Dense(len(self.classes),activation = "sigmoid"))

        self.model.compile(optimizer = "adam",
            loss = 'sparse_categorical_crossentropy',
            metrics=['accuracy'])

        disp.add_row(['Laying the pipeline for the model'])
        print("\n{}\n".format(disp))
        self.model.summary()
        print("\n\n")
        time.sleep(1)
        

        self.modelHistory = self.model.fit(x_train ,y_train , epochs =epochs,batch_size = batch_size,validation_data = (x_test,y_test),verbose =verbose)

        self.model.save(saved_model_path + "BreastCancerANN.h5")
        print("\n\n-----+-----+-----+-----+-----+-----+-----+------+------+-----+------+------")
        print("                         Saving trained Model......")
        print("-----+-----+-----+-----+-----+-----+-----+------+------+-----+------+------")
        print("Model saved in disc as \'BreastCancerANN.h5\' file in path: {}".format(saved_model_path))
        print("-----+-----+-----+-----+-----+-----+-----+------+------+-----+------+------\n")
    
    def validate(self,x,y,classes=[]):
        print("\n\nValidating Model")

        if(len(self.classes) ==0):
            if(len(classes)==0):
                print("No classes provided....")
                print("Exiting program....")
                exit()
            self.classes = classes

        #validation
        val_loss,val_acc = self.model.evaluate(x,y)
        ta_val = PrettyTable(['Loss (%)', 'Accuracy (%)'])
        ta_val.add_row([val_loss*100,val_acc*100])

        #validation table
        print("\n\n{}".format(ta_val))

        #prediction
        predict = self.model.predict(x)
        print("\n\nValidating prediction:\n")

        t_tot = PrettyTable(['Sl.No.', 'Argmax Value','Expected Value','Predicted Tag','Expected Tag','Result'])
        for i in range(len(predict)):
            argMax = np.argmax(predict[i])
            tag = self.classes[argMax]
            result = "Correct" if y[i] == argMax else "Wrong"
            t_tot.add_row([i+1,argMax,y[i],tag,y[i],result])

        #Total table
        print(t_tot)
    
    def predict(self,x,classes=[]):
        print("\nPredicting model based on the given inputs...")
        if(len(self.classes) ==0):
            if(len(classes)==0):
                print("No classes provided....")
                print("Exiting program....")
                exit()
            self.classes = classes

        #prediction
        predict = self.model.predict(x)
        print("\n\nPrediction:\n")

        t_tot = PrettyTable(['Sl.No.', 'Argmax Value','Predicted Tag'])
        for i in range(len(predict)):
            argMax = np.argmax(predict[i])
            tag = self.classes[argMax]
            t_tot.add_row([i+1,argMax,tag])

        #Total table
        print(t_tot)
    
    def visualize(self):
        plt.plot(self.modelHistory.history['loss'])
        plt.plot(self.modelHistory.history['val_loss'])
        plt.title('model train vs validation loss for DNN model, epochs = {}'.format(self.epochs))
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'validation'], loc='upper right')
        plt.savefig('training_graphs/DNN_{}.png'.format(self.epochs))
        plt.show()
        print("Graph saved in 'training_graphs' directory")



class CNN1d:
    def __init__(self):
        self.model = None
        self.classes = []
        self.modelHistory  = []
        self.epochs = 0
        print("\nConvolutional 1D Model initialized......")
    
    def model_config(self,model_path): 
        self.model = tf.keras.models.load_model(model_path)
        self.model.summary()

    def loadModel(self,model_path):
        self.model = tf.keras.models.load_model(model_path)

    def trainModel(self,x_train,y_train,validation_data=(None,None),classes=[],epochs = 50,batch_size = 10,verbose = 0):
        disp = PrettyTable(['Training Breast Cancer CNN model.....'])

        x_test,y_test = validation_data
        self.epochs = epochs

        #X shape must be (num_of_rows,1,num_of_features)
        x_train = x_train.reshape(len(x_train),1,len(x_train[0]))
        x_test = x_test.reshape(len(x_test),1,len(x_test[0]))
    

        if(len(classes)==0):
            print("No classes provided....")
            print("Exiting program....")
            exit()

        self.classes = classes

        #input shape = (1,num_of_feautures)
        inp_shape = (x_train.shape[1],x_train.shape[2])

        self.model = Sequential()
        self.model.add(Conv1D(filters=64, kernel_size=1, activation='relu', input_shape=inp_shape))
        self.model.add(Conv1D(filters=128, kernel_size=1, activation='relu'))
        # model.add(Dropout(0.5))
        self.model.add(MaxPooling1D(pool_size=1))
        self.model.add(Flatten())
        self.model.add(Dense(100, activation='relu'))
        self.model.add(Dense(len(self.classes), activation='sigmoid'))

        self.model.compile(optimizer = "adam",
            loss = 'sparse_categorical_crossentropy',
            metrics=['accuracy'])

        disp.add_row(['Laying the pipeline for the model'])
        print("\n{}\n".format(disp))
        self.model.summary()
        print("\n\n")
        time.sleep(1)
        

        self.modelHistory = self.model.fit(x_train ,y_train , epochs =epochs,batch_size = batch_size,validation_data = (x_test,y_test),verbose =verbose)

        self.model.save(saved_model_path + "BreastCancerCNN.h5")
        print("\n\n-----+-----+-----+-----+-----+-----+-----+------+------+-----+------+------")
        print("                         Saving trained Model......")
        print("-----+-----+-----+-----+-----+-----+-----+------+------+-----+------+------")
        print("Model saved in disc as \'BreastCancerCNN.h5\' file in path: {}".format(saved_model_path))
        print("-----+-----+-----+-----+-----+-----+-----+------+------+-----+------+------\n")
    
    def validate(self,x,y,classes=[]):
        print("\n\nValidating Model")

        #X shape must be (num_of_rows,1,num_of_features)
        x = x.reshape(len(x),1,len(x[0]))

        if(len(self.classes) ==0):
            if(len(classes)==0):
                print("No classes provided....")
                print("Exiting program....")
                exit()
            self.classes = classes

        #validation
        val_loss,val_acc = self.model.evaluate(x,y)
        ta_val = PrettyTable(['Loss (%)', 'Accuracy (%)'])
        ta_val.add_row([val_loss*100,val_acc*100])

        #validation table
        print("\n\n{}".format(ta_val))

        #prediction
        predict = self.model.predict(x)

        print("\n\nValidating prediction:\n")

        t_tot = PrettyTable(['Sl.No.', 'Argmax Value','Expected Value','Predicted Tag','Expected Tag','Result'])
        for i in range(len(predict)):
            argMax = np.argmax(predict[i])
            tag = self.classes[argMax]
            result = "Correct" if y[i] == argMax else "Wrong"
            t_tot.add_row([i+1,argMax,y[i],tag,y[i],result])

        #Total table
        print(t_tot)

    
    def predict(self,x,classes=[]):
        print("\nPredicting model based on the given inputs...")

        #X shape must be (num_of_rows,1,num_of_features)
        x = x.reshape(len(x),1,len(x[0]))

        if(len(self.classes) ==0):
            if(len(classes)==0):
                print("No classes provided....")
                print("Exiting program....")
                exit()
            self.classes = classes

        #prediction
        predict = self.model.predict(x)
        print("\n\nPrediction:\n")

        t_tot = PrettyTable(['Sl.No.', 'Argmax Value','Predicted Tag'])
        for i in range(len(predict)):
            argMax = np.argmax(predict[i])
            tag = self.classes[argMax]
            t_tot.add_row([i+1,argMax,tag])

        #Total table
        print(t_tot)
    
    def visualize(self):
        plt.plot(self.modelHistory.history['loss'])
        plt.plot(self.modelHistory.history['val_loss'])
        plt.title('model train vs validation loss for Conv1D model, epochs = {}'.format(self.epochs))
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'validation'], loc='upper right')
        plt.savefig('training_graphs/CNN1d_{}.png'.format(self.epochs))
        plt.show()
        print("Graph saved in 'training_graphs' directory")
