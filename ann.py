from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.optimizers import SGD, RMSprop
from keras.utils import np_utils
from keras.regularizers import l2
from keras.callbacks import ModelCheckpoint
from keras import regularizers
from IPython.display import clear_output
import time
import os
import csv
import matplotlib.pyplot as plt
import numpy as np
import sklearn 
import tensorflow
from sklearn.model_selection import train_test_split

## Declaration of model hyperparameters

np.set_printoptions(threshold=np.inf)

#record programme start time
time_start = time.time()

#system parameters
batch_size = 1			#size of each batch
nb_epoch = 2000				#number of time training through the set of data

## Data-preprocessing

# Lazy doing the file processing since the data is small


input_data = np.array([[0.904377769,0.774029158,-0.981089637,1.619231654,-0.841681234,-0.800252014,0.985609803,-1.001273189,0.919859801,0.692497167,-0.922513733,1.861571657],
[1.171580291,-1.850939292,-0.718658253,-0.186966367,-1.011163944,3.021359646,0.485354858,-0.214168123,1.242560108,-1.443680534,-0.757777397,-0.356980628],
[-0.698837367,-0.975949808,-1.191034745,0.816476978,0.531128642,0.898242056,1.524345897,-0.758537839,-0.760617269,-0.935066796,-1.036055084,0.71986054],
[1.438782814,-0.975949808,-1.427222991,0.459697122,-1.168540735,0.898242056,2.370036278,-0.607831922,1.577437785,-0.935066796,-1.144400905,0.292391596],
[0.637175246,-0.100960325,0.987145746,-0.766733633,-0.658639921,-0.163316738,-0.959826095,0.394188912,0.609336864,-0.223007562,0.930770045,-0.801748224],
[-0.431634844,1.649018642,0.908416331,-0.588343705,0.24785041,-1.224875532,-0.924454533,0.166890678,-0.510981183,1.811447391,0.829267118,-0.678735579],
[-0.698837367,-0.975949808,1.144604577,-0.632941187,0.531128642,0.898242056,-1.026053701,0.219557342,-0.760617269,-0.935066796,1.140618793,-0.710641984],
[0.637175246,-0.100960325,1.0921183,-0.610642446,-0.658639921,-0.163316738,-1.00461311,0.192906741,0.609336864,-0.223007562,1.069655448,-0.694784885],
[-0.966039889,-0.100960325,1.013388884,-0.722136151,0.844225626,-0.163316738,-0.971269835,0.332993233,-0.998075986,-0.223007562,0.965111235,-0.772148307],
[-2.034849979,0.774029158,1.93189873,-1.569488308,2.51407625,-0.800252014,-1.287478457,2.54215721,-1.826137151,0.692497167,2.32672041,-1.203076981],
[-1.500444934,0.774029158,-0.351254315,-0.878227338,1.580924431,-0.800252014,-0.014900089,0.562261548,-1.436461308,0.692497167,-0.48456852,-0.872384392],
[1.171580291,-0.975949808,-0.797387668,1.663829136,-1.011163944,0.898242056,0.619751707,-1.011988532,1.242560108,-0.935066796,-0.809859423,1.932303928],
[-0.966039889,1.649018642,-0.902360222,-0.833629856,0.844225626,-1.224875532,0.818858153,0.492269656,-0.998075986,1.811447391,-0.875753957,-0.844706547],
[-0.164432322,-0.100960325,0.672228085,1.039464388,-0.009675245,-0.163316738,-0.807987196,-0.837130495,-0.249167726,-0.223007562,0.538444125,1.012015573],
[0.637175246,-0.975949808,-0.272524899,-0.186966367,-0.658639921,0.898242056,-0.101150941,-0.214168123,0.609336864,-0.935066796,-0.419561028,-0.356980628],
[-0.431634844,-0.100960325,-0.797387668,-0.298460072,0.24785041,-0.163316738,0.619751707,-0.122504494,-0.510981183,-0.223007562,-0.809859423,-0.452603739],
[0.904377769,-0.975949808,-1.164791606,0.682684532,-0.841681234,0.898242056,1.448109658,-0.706031257,0.919859801,-0.935066796,-1.022749457,0.553793468],
[0.637175246,-0.100960325,1.24957713,0.036021043,-0.658639921,-0.163316738,-1.067160491,-0.372942623,0.609336864,-0.223007562,1.285586768,-0.151318861],
[-0.698837367,1.649018642,0.672228085,-1.480293344,0.531128642,-1.224875532,-0.807987196,2.12992087,-0.760617269,1.811447391,0.538444125,-1.170786161],
[-1.767647457,-0.975949808,1.144604577,-1.524890826,2.018339345,0.898242056,-1.026053701,2.326223889,-1.637387914,-0.935066796,1.140618793,-1.187315985],
[0.637175246,-0.100960325,1.0921183,-0.610642446,-0.658639921,-0.163316738,-1.00461311,0.192906741,0.609336864,-0.223007562,1.069655448,-0.694784885],
[-0.966039889,-0.100960325,-0.298768038,-0.276161331,0.844225626,-0.163316738,-0.073069267,-0.141585494,-0.998075986,-0.223007562,-0.441483632,-0.433863531],
[0.904377769,0.774029158,-0.692415114,1.329348021,-0.841681234,-0.800252014,0.443079791,-0.925477886,0.919859801,0.692497167,-0.739909841,1.420552101],
[0.369972724,0.774029158,-0.613685699,1.128659352,-0.460345157,-0.800252014,0.323110009,-0.8658339,0.310991297,0.692497167,-0.684786528,1.134259389],
[1.171580291,-0.975949808,-0.797387668,1.217854316,-1.011163944,0.898242056,0.619751707,-0.893145624,1.242560108,-0.935066796,-0.809859423,1.259578522],
[0.102770201,1.649018642,-0.902360222,1.173256834,-0.244807374,-1.224875532,0.818858153,-0.879657318,0.024823101,1.811447391,-0.875753957,1.196534541]])

output_data = np.array([[-0.7445281],
[-0.5925438],
[-0.7850572],
[-0.5976099],
[0.15217912],
[1.26673042],
[0.6537272],
[0.15724526],
[1.180606],
[2.18876831],
[0.60306578],
[-1.4335234],
[0.08125313],
[-0.2379138],
[-0.5925438],
[-0.278443],
[-0.0909957],
[0.15217912],
[1.59096352],
[2.00132105],
[0.15724526],
[-0.3341705],
[-1.2156793],
[-0.9167769],
[-1.4335234],
[-0.9319753]])

#shuffling data set
in_data, out_data = sklearn.utils.shuffle(input_data, output_data, random_state=1)
train_data = [in_data, out_data]
np.set_printoptions(threshold=10)

#passing of data and labeling to variable X and y
(X,y) = (train_data[0],train_data[1])

#splitting the data set into training and testng sets
X_train,X_test,Y_train,Y_test = train_test_split(X,y, test_size=0.2, random_state=4)

#con. neural network architecture
opt = SGD(0.001, clipnorm=1.)
model = Sequential()
model.add(Dense(4))
model.add(Activation('tanh'))
model.add(Dropout(0.1))
model.add(Dense(4))
model.add(Activation('tanh'))
model.add(Dropout(0.1))
model.add(Dense(4))
model.add(Dropout(0.1))
model.add(Dense(1))
model.compile(metrics=['mse'],loss='mean_squared_error',optimizer='adam')
#END of con. neural network architecture

## Training of the model

#training

#fitted_model=model.fit(X_train,	Y_train,	batch_size=batch_size,	nb_epoch=nb_epoch,	verbose=1,	validation_data=(X_test,Y_test))
#fitted_model=model.fit(X_train,	Y_train,	batch_size=batch_size,	nb_epoch=nb_epoch,	verbose=1,	validation_split=0.2)
hist = model.fit(X_train, Y_train, batch_size=batch_size, epochs=nb_epoch,
                 verbose=1, validation_data=(X_test,Y_test))

#hist = model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch,
                # verbose=1, validation_split=0.2)	#to change the validation_split value

#evaluation
score = model.evaluate(X_test,Y_test, verbose=2)
print("Test score:", score[0])
print("Test mse:", score[1])

'''
edit log
error message:
ValueError: Error when checking target: expected activation_4 to have shape (None, 2) but got array with shape (2, 1)
'''

train_loss = hist.history['loss']
val_loss = hist.history['val_loss']
train_mse = hist.history['mse']
val_mse = hist.history['val_mse']
xc = range(nb_epoch)

plt.figure(1, figsize=(7, 5))
plt.plot(xc, train_loss)
plt.plot(xc, val_loss)
plt.xlabel('num of Epochs')
plt.ylabel('loss')
plt.title('train_loss vs val_loss')
plt.grid(True)
plt.legend(['train', 'val'], loc=4)
print (plt.style.available)  # use bmh, classic,ggplot for big pictures
plt.style.use(['classic'])

plt.figure(2, figsize=(7, 5))
plt.plot(xc, train_mse)
plt.plot(xc, val_mse)
plt.xlabel('num of Epochs')
plt.ylabel('mse')
plt.title('train_mse vs val_mse')
plt.grid(True)
plt.legend(['train', 'val'], loc=4)
# print plt.style.available # use bmh, classic,ggplot for big pictures
plt.style.use(['classic'])
plt.show()

y_preds = model.predict(X_test)
y_train = model.predict(X_train)
print(y_preds)
print(y_train)
fig = plt.figure(figsize=(10, 4))
ax = fig.add_subplot(111)
ax.plot(Y_test, y_preds, 'k',label='prediction')
ax.plot(Y_train, y_train, 'r.', markersize=14,markeredgecolor='k',markeredgewidth=0.5, label='train')
#ax.plot(x_grid,data_function(x_grid,is_noise=False), 'g--', label='data function',alpha=0.5)
ax.set_xlim(-3, 3)
ax.set_ylim(-3, 3)
ax.legend()
ax.set_ylabel('y')
ax.set_xlabel('x')
fig.show()

for element in y_preds:
  print(element)

for element in y_train:
  print(element)

print("Target:\t\n")
for element in Y_test:
  print(element)
for element in Y_train:
  print(element)

model.summary()
