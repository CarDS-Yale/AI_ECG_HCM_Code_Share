import pandas as pd
import numpy as np
import tensorflow as tf


from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard

from tensorflow.keras.models import Model
from tensorflow.keras.layers import BatchNormalization, GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.applications import EfficientNetB3
import tensorflow.keras.backend as K
from keras.callbacks import CSVLogger
from utils import *



def loss_fn(y_true,y_pred):
    y_true = tf.cast(y_true, tf.float32)

    weights = [["PUT YOUR WEIGHTS HERE"]]
    weights = np.array(weights)
    return K.mean((weights[:,1]**(1-y_true))*(weights[:,0]**(y_true))*K.binary_crossentropy(y_true, y_pred), axis=-1)


def build_transfer_model():
    cnn_model = EfficientNetB3(input_shape=(300, 300, 3), include_top = False, weights = 'imagenet')
    # OPTIONALLY LOAD WEIGHTS FROM PRETRAINED MODEL HERE
    cnn_model.trainable = False
    x = GlobalAveragePooling2D(name='avg_pool')(cnn_model.output)

    x = BatchNormalization()(x)
    x = Dropout(0.2, name="top_dropout1")(x)
    x = Dense(64, activation = 'relu')(x)
    x = Dropout(0.2, name="top_dropout2")(x)
    x = Dense(32, activation = 'relu')(x)
    x = Dropout(0.2, name="top_dropout3")(x)
    x = Dense(1,activation = 'sigmoid')(x)

    model = Model(cnn_model.input, x)
    opt = tf.keras.optimizers.Adam(learning_rate=0.001, clipnorm=1.0)
    model.compile(optimizer = opt,loss = loss_fn, metrics=[tf.keras.metrics.BinaryAccuracy(),tf.keras.metrics.AUC(curve='ROC'), tf.keras.metrics.AUC(curve='PR')])
    return model

def unfreeze_model(model):
    for layer in model.layers:
        if not isinstance(layer, BatchNormalization):
            layer.trainable = True
    
    opt = tf.keras.optimizers.Adam(learning_rate=0.00001, clipnorm=1.0)
    model.compile(
        optimizer=opt,loss=loss_fn, metrics=[tf.keras.metrics.BinaryAccuracy(),tf.keras.metrics.AUC(curve='ROC'), tf.keras.metrics.AUC(curve='PR')]
    )



gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


strategy = tf.distribute.MultiWorkerMirroredStrategy()
with strategy.scope():
    model_cnn = build_transfer_model()

model_cnn.summary()
# ok, let's train then!
saved_model_file = 'PATH TO SAVE'
checkpoint = ModelCheckpoint(saved_model_file, monitor='val_loss', save_best_only=False, verbose=1)


#Add data file here
train_df = pd.read_csv('TRAIN FILE HERE')
validate_df = pd.read_csv('VALIDATE FILE HERE')

train_sequence = DataSequenceTrain(df=train_df, batch_size = 64, mode = "image")
validation_sequence = DataSequenceTest(df=validate_df, batch_size = 64, mode ="image")

csv_logger = CSVLogger('PATH TO LOG HERE', append=True, separator=';')

# Train model
model_cnn.fit(train_sequence, validation_data = validation_sequence, epochs = 50, 
    verbose = 1, callbacks=[checkpoint, csv_logger], max_queue_size = 100 ,
    use_multiprocessing=True, workers=40,shuffle=True)


