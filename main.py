# https://towardsdatascience.com/predicting-the-music-mood-of-a-song-with-deep-learning-c3ac2b45229e
# Author: Cristobal Veas
# Date: August 15, 2020
# "Predicting the Music Mood of a Song with Deep Learning."
# This includes only the training part. The socket server is written by teammate Max Hansen

# Our data is trained with 1746 songs (contained in data.csv)

import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
import tensorflow as tf
import socket
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score

from numpy.random import seed
seed(1)
tf.random.set_seed(5)

tf.compat.v1.disable_eager_execution()
tf.compat.v1.disable_v2_behavior()

df = pd.read_csv("data.csv")

col_features = df.columns[0:10]
X = MinMaxScaler().fit_transform(df[col_features])
X2 = np.array(df[col_features])
Y = df['mood']

# Encode the categories
encoder = LabelEncoder()
encoder.fit(Y)
encoded_y = encoder.transform(Y)


# Convert to  dummy
dummy_y = np_utils.to_categorical(encoded_y)
X_train, X_test, Y_train, Y_test = train_test_split(X, encoded_y, test_size=0.15, random_state=15)
target = pd.DataFrame({'mood': df['mood'].tolist(), 'encode': encoded_y}).drop_duplicates().sort_values(['encode'], ascending=True)


def base_model():
    # Create the model
    model = Sequential()
    # Add 1 layer with 8 nodes,input of 4 dim with relu function
    model.add(Dense(8, input_dim=10, activation='relu'))
    # Add 1 layer with output 3 and softmax function
    model.add(Dense(4, activation='softmax'))
    # Compile the model using sigmoid loss function and adam optim
    model.compile(loss='categorical_crossentropy', optimizer='adamax', metrics=['accuracy'])

    return model


# Configure the model
estimator = KerasClassifier(build_fn=base_model, epochs=400, batch_size=100, verbose=0)
estimator.fit(X_train, Y_train)
y_preds = estimator.predict(X_test)

print("Accuracy Score", accuracy_score(Y_test, y_preds))


def predict_mood(song_features):
    # Join the model and the scaler in a Pipeline
    pip = Pipeline([('minmaxscaler', MinMaxScaler()), ('keras', KerasClassifier(build_fn=base_model, epochs=400, batch_size=100, verbose=0))])

    pip.fit(X2, encoded_y)    # Fit the Pipeline
    preds = song_features     # Obtain the features of the song
    preds_features = np.array(preds).reshape(-1,1).T    # Pre-process the features
    r = pip.predict(preds_features) # Predict
    mood = np.array(target['mood'][target['encode']==int(r)])

    return mood[0]


### Server Stuff
# Written by team member Max Hansen. 
# Used references from official documentation

# Declare IPV4 TCP socket
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

# Bind to socket (0.0.0.0 binds to the server, arbitrary port number 4040)
s.bind(("0.0.0.0", 4040))

# Allow up-to 3 simultaneous connections (not necessary but for safety)
s.listen(3)

whitelist = ["69.174.157.38", "24.1.251.2", "73.103.92.140"]    
​
# Infinite loop of server tasks
while True:
    # Accept connection
    clientSocket, address = s.accept()
    print(f"Connection from {address} established\n")

    if address[0] not in whitelist and address:
        clientSocket.close()
        print(f"{address} is not in the whitelist!\nClosing connection\n")
    else:
        # Wait for message from client arbitrary 1024 byte size
        msg = clientSocket.recv(1024)
        try:
            # Decode the message
            decoded = msg.decode(encoding="UTF-8")
​
            # Format string into list (of strings)
            res = decoded.strip('][').split(', ')
            failed = False
            # Change the list to appropriate data types
            song = []
            for i in res:
                if i == "null":
                    failed = True
                    break
                else:
                    song.append(float(i))
​
            if not failed:
                print(f"Song Data Received: {song}")
​
                # Predict with parameters in song
                result = predict_mood(song)
​
                # Send result of prediction
​
                clientSocket.send(f"{result}".encode(encoding='UTF-8'))
​
                if result == 0:
                    print("Sad\n")
                elif result == 1:
                    print("Happy\n")
                elif result == 2:
                    print("Chill\n")
                elif result == 3:
                    print("Energetic\n")
        except e:
            print(e)