import tensorflow as tf
import numpy as np
import pandas as pd
import seaborn as sns
from datetime import datetime
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import regularizers
from sklearn import preprocessing
from matplotlib import pyplot as plt
from matplotlib import dates as md
from tensorflow.keras import layers, losses
from tensorflow.keras import regularizers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1DTranspose, Conv1D, Conv2D, Flatten, Dropout, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator



column_names = ['fare_amount','pickup_datetime', 'pickup_longitude', 'pickup_latitude',
                'dropoff_longitude', 'dropoff_latitude', 'passenger_count']
KEY = 'id'
TARGET = 'trip_duration'
TRAIN_URL = "C:\\Users\\Minh\\Downloads\\nyc-taxi-trip-duration\\train.csv"
FINAL_URL = "C:\\Users\\Minh\\Downloads\\nyc-taxi-trip-duration\\test.csv"
RESULT_URL = "C:\\Users\\Minh\\Downloads\\nyc-taxi-trip-duration\\result.csv"
NROWS = 22513
MAXNROWS = 625135
DATETIME = 'pickup_datetime'
DROPOFFDATETIME = 'dropoff_datetime'

"""
PREPARING THE DATASET

"""

def process(file_url):
    dataset = pd.read_csv(file_url, nrows=NROWS)    

    dataset = dataset.dropna()
    dataset = dataset[dataset['dropoff_latitude'] != 0]
    dataset = dataset[dataset['dropoff_longitude'] != 0]
    dataset = dataset[dataset['pickup_latitude'] != 0]
    dataset = dataset[dataset['pickup_longitude'] != 0]
    
    dataset['diff_longtitude'] = dataset.apply(lambda x: 
                    abs(x['pickup_longitude'] - x['dropoff_longitude']), axis=1)
    dataset['diff_latitude'] = dataset.apply(lambda x: 
                    abs(x['pickup_latitude'] - x['dropoff_latitude']), axis=1)
    
    dataset[DATETIME] = [datetime.strptime(x, "%Y-%m-%d %H:%M:%S") 
                                   for x in dataset[DATETIME]]
    dataset['hour'] = dataset[DATETIME].apply(lambda x: x.hour)
    dataset['minute'] = dataset[DATETIME].apply(lambda x: x.minute)
    # dataset['day'] = dataset[DATETIME].apply(lambda x: x.dayofyear)
    dataset['weekday'] = dataset[DATETIME].apply(lambda x: x.weekday())
    dataset['week'] = dataset[DATETIME].apply(lambda x: x.week)
   
    dataset.pop(DATETIME)
    dataset.pop('store_and_fwd_flag')
    return dataset


dataset = process(TRAIN_URL)
dataset.pop(DROPOFFDATETIME)
dataset.pop(KEY)

dataset = dataset.sample(frac=1).reset_index(drop=True)
dataset[TARGET] = dataset[TARGET]

train_dataset = dataset.sample(frac=0.9,random_state=0)
test_dataset = dataset.drop(train_dataset.index)

train_labels = train_dataset.pop(TARGET)
test_labels = test_dataset.pop(TARGET)

final = process(FINAL_URL)
final_key = final.pop(KEY)


train_stats = train_dataset.describe()
train_stats = train_stats.transpose()
print(train_stats.head())

def norm(x):
    return (x - train_stats['mean']) / train_stats['std']
final = norm(final)
test_dataset = norm(test_dataset)
train_dataset = norm(train_dataset)

print(train_dataset.head())
print(final.head())
print(train_labels.head())

"""
TWO DENSELY PRELU ACTIVATED CONNECTED LAYERS WITH DROPOUTS, REGULARIZATIONS
AND LAYER NORMALIZATIONS

"""
from tensorflow.keras.losses import MSLE

def build_model():
    model = keras.Sequential([
        layers.Dense(50, 
                     input_shape = [len(train_dataset.keys())],
                     ),
        layers.PReLU(alpha_initializer=tf.initializers.constant(0.25)),
        layers.LayerNormalization(),
        layers.Dropout(rate = 0.5),
        layers.Dense(50),
        layers.PReLU(alpha_initializer=tf.initializers.constant(0.25)),
        layers.LayerNormalization(),
        layers.Dropout(rate = 0.5),
        layers.Dense(1)
        ])

    model.compile(loss = MSLE,
                optimizer = tf.keras.optimizers.Adam(0.0005),
                )
    return model

model = build_model()
model.summary()


"""
FITTING AND PLOTTING THE LOSS VALUE GRAPH

"""



history = model.fit(
    train_dataset, train_labels,
    epochs = 1000, batch_size = 128, validation_split = 0.2, verbose = 1,
    callbacks = [keras.callbacks.EarlyStopping(monitor = 'val_loss', patience = 20)],
)

plt.plot(history.history["loss"], label="Training Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.ylim([0, 40])
plt.legend()
plt.show()

"""
PLOTTING THE PREDICTION GRAPH

"""
train_predictions = model.predict(train_dataset).flatten()
plt.axes(aspect = 'equal')
plt.scatter(train_labels, train_predictions, s=1, color="b", label="Training")
plt.xlabel('True Values')
plt.ylabel('Predictions')
lims = [0, 5000]
plt.xlim(lims)
plt.ylim(lims)
plt.legend()
plt.grid(True)
plt.plot(lims, lims)



test_predictions = model.predict(test_dataset).flatten()
plt.axes(aspect = 'equal')
plt.scatter(test_labels, test_predictions, s=1, color="r", label="Test")
plt.xlabel('True Values')
plt.ylabel('Predictions')
lims = [0, 5000]
plt.xlim(lims)
plt.ylim(lims)
plt.legend()
plt.grid(True)
plt.plot(lims, lims)

"""
PRINTING THE RESULT

"""

result = pd.DataFrame()
result[KEY] = final_key
result[TARGET] = model.predict(final).flatten()
result[TARGET] = result[TARGET]
result[TARGET] = result[TARGET].apply(lambda x: 1 if (x<1) else x)
pd.DataFrame(result).to_csv(RESULT_URL, index = False)





