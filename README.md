# Developing a Neural Network Regression Model

## AIM
To develop a neural network regression model for the given dataset.

## THEORY
A neural network can be used to solve a problem by Data collection and Preprocessing,choosing a appropriate neural network architecture ,Train the neural network using the collected and preprocessed data,Assess the performance of the trained model using evaluation metrics,Depending on the performance of the model, you might need to fine-tune hyperparameters or adjust the architecture to achieve better results,Once you're satisfied with the model's performance, you can deploy it to production where it can be used to make predictions on new, unseen data

For the problem statement we have dealt with , we have developed a neural network with three hidden layers. First hidden layer consists of 4 neurons ,second hidden layer with 8 neurons , third layer with 5 neurons .
The input and output layer contain 1 neuron . The Activation Function used is 'relu'.

## Neural Network Model

![image](https://github.com/poojaanbu0/basic-nn-model/assets/119390329/bcf34274-a843-4091-8b77-f71d634d5317)

## DESIGN STEPS

### STEP 1:

Loading the dataset

### STEP 2:

Split the dataset into training and testing

### STEP 3:

Create MinMaxScalar objects ,fit the model and transform the data.

### STEP 4:

Build the Neural Network Model and compile the model.

### STEP 5:

Train the model with the training data.

### STEP 6:

Plot the performance plot

### STEP 7:

Evaluate the model with the testing data.

## PROGRAM
### Name: POOJA A
### Register Number:212222240072
```
import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import MinMaxScaler

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

from google.colab import auth
import gspread
from google.auth import default

auth.authenticate_user()
creds, _ = default()
gc = gspread.authorize(creds)

worksheet = gc.open('datasheet').sheet1

rows = worksheet.get_all_values()

df = pd.DataFrame(rows[1:], columns=rows[0])
df = df.astype({'Input':'int'})
df = df.astype({'Output':'int'})
df.head()

X = df[['Input']].values
y = df[['Output']].values

X

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.33,random_state = 33)

Scaler = MinMaxScaler()

Scaler.fit(X_train)

X_train1 = Scaler.transform(X_train)

marks_data = Sequential([Dense(4,activation='relu'),Dense(8,activation='relu'),Dense(5,activation='relu'),Dense(1)])

marks_data.compile(optimizer = 'rmsprop' , loss = 'mae')

marks_data.fit(X_train1 , y_train,epochs = 500)

loss_df = pd.DataFrame(marks_data.history.history)

loss_df.plot()

X_test1 = Scaler.transform(X_test)

marks_data.evaluate(X_test1,y_test)

X_n1 = [[30]]

X_n1_1 = Scaler.transform(X_n1)

marks_data.predict(X_n1_1)
```
## Dataset Information

![image](https://github.com/poojaanbu0/basic-nn-model/assets/119390329/efcec8ef-2db6-4b3e-b258-9ea6a71ebc11)

## OUTPUT
### Training Loss Vs Iteration Plot

![image](https://github.com/poojaanbu0/basic-nn-model/assets/119390329/71022b79-c6de-48cb-b118-6febfb0bcfad)

### Test Data Root Mean absolute Error

![image](https://github.com/poojaanbu0/basic-nn-model/assets/119390329/b389a4ec-5ae0-49f7-8abe-ae553dff4d7a)

### New Sample Data Prediction

![image](https://github.com/poojaanbu0/basic-nn-model/assets/119390329/e8e06d7a-b239-4be8-a406-8e7ce0a64765)



### RESULT
Thus a neural network regression model is developed for the created dataset.
