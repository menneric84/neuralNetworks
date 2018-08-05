import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense
import time
from time import strftime
from keras.callbacks import TensorBoard
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
from keras.models import Sequential
from keras.layers import Dense
import pickle
# Lab 2 - Load the data\
# find the right path for batch ai vs local
basepath = '.'
outpath = outpath = os.path.join (basepath, "out")
if not os.path.exists(outpath):
    os.makedirs(outpath)
dataset = pd.read_csv('WA_Fn-UseC_-Sales-Win-Loss.csv')
seed = 7
np.random.seed(7)
# Lab 3 - Data Processing and selecting the correlated data
responsetimeencoder = LabelEncoder()
dataset['Opportunity Result'] = responsetimeencoder.fit_transform(dataset['Opportunity Result'])

suppliesgroupencoder = LabelEncoder()
dataset['Supplies Group'] = suppliesgroupencoder.fit_transform(dataset['Supplies Group'])

suppliessubgroupencoder = LabelEncoder()
dataset['Supplies Subgroup'] = suppliessubgroupencoder.fit_transform(dataset['Supplies Subgroup'])

regionencoder = LabelEncoder()
dataset['Region'] = regionencoder.fit_transform(dataset['Region'])

competitortypeencoder = LabelEncoder()
dataset['Competitor Type'] = competitortypeencoder.fit_transform(dataset['Competitor Type'])



#routetomarketencoder = LabelEncoder()
#dataset['Route To Market'] = routetomarketencoder.fit_transform(dataset['Route To Market'])

#correlations = dataset.corr()['Opportunity Result'].sort_values()
# Lab 4 - Encoding and scaling the data
#Throw out unneeded columns 
dataset = dataset.drop(columns=['Client Size By Employee Count',
'Client Size By Revenue',
'Elapsed Days In Sales Stage',
'Opportunity Number',
'Opportunity Amount USD',
'Competitor Type',
'Supplies Group',
'Supplies Subgroup',
'Region'])
    
dataset = pd.concat([pd.get_dummies(dataset['Route To Market'], prefix='Route To Market', drop_first=True),dataset], axis=1)
dataset = dataset.drop(columns=['Route To Market'])

#Create the input data set (X) and the outcome (y)
X = dataset.drop(columns=['Opportunity Result']).iloc[:, 0:dataset.shape[1] - 1].values
y = dataset.iloc[:, dataset.columns.get_loc('Opportunity Result')].values

sc = StandardScaler()
X = sc.fit_transform(X)
# Lab 5 - Create the ANN
model = Sequential()
model.add(Dense(units = 8, activation = 'relu', input_dim=X.shape[1], name= 'Input_Layer'))
model.add(Dense(units = 8, activation = 'relu', name= 'Hidden_Layer_1'))
model.add(Dense(units = 1, activation = 'sigmoid', name= 'Output_Layer'))
model.compile(optimizer= 'nadam', loss = 'binary_crossentropy', metrics=['accuracy'])
# Lab 6 Tensorboard and fitting the model to the data
embedding_layer_names = set(layer.name
                            for layer in model.layers)
tensorboard = TensorBoard(log_dir=os.path.join (os.path.join (basepath, "logs"), format(strftime("%Y %d %m %H %M %S", time.localtime()))), write_graph=True, write_grads=True, write_images=True, embeddings_metadata=None, embeddings_layer_names=embedding_layer_names)
#Fit the ANN to the training set
model.fit(X, y, validation_split = .20, batch_size = 64, epochs = 25, callbacks = [tensorboard])

# summary to console
print (model.summary())

# Lab 7 Improve your model
def build_classifier(optimizer):
   model = Sequential()
   model.add(Dense(units = 24, activation = 'relu', input_dim=X.shape[1], name= 'Input_Layer'))
   model.add(Dense(units = 24, activation = 'relu', name= 'Hidden_Layer_1'))
   model.add(Dense(1, activation = 'sigmoid', name= 'Output_Layer'))
   model.compile(optimizer= optimizer, loss = 'binary_crossentropy', metrics=['accuracy'])
   return model
classifier = KerasClassifier(build_fn = build_classifier)
parameters = {'batch_size': [64],
              'epochs': [25, 10, 15],
              'optimizer': ['nadam']}
grid_search = GridSearchCV(estimator = classifier,
   param_grid = parameters,
   scoring = 'accuracy',
   verbose = 5)
grid_search = grid_search.fit(X, y)
best_parameters = grid_search.best_params_
best_accuracy = grid_search.best_score_
# Lab 8 Save your model
model_json = model.to_json()
with open(os.path.join (outpath, "model.json"), "w") as json_file:
    json_file.write(model_json)

with open(os.path.join (outpath, "standardscalar.pickle"), 'wb') as handle:
    pickle.dump(sc, handle, protocol=pickle.HIGHEST_PROTOCOL)
model.save_weights(os.path.join (outpath, "model.h5"))
new_prediction = model.predict(sc.transform(np.array([[0,1,0,0,9,119,97,0,.81,.644,.245,6]])))
if (new_prediction > 0.5):
    print ("Won")
else:
    print ("Loss")


