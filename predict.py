import pandas as pd
import numpy as np
import tensorflow as tf


from tensorflow.keras.models import load_model
import tensorflow.keras.backend as K
from utils import *

test_df = pd.read_csv('PATH_TO_TEST_FILE')

def loss_fn(y_true,y_pred):
    y_true = tf.cast(y_true, tf.float32)

    weights = [['PUT CUSTOM WEIGHTS HERE']]
    weights = np.array(weights)
    return K.mean((weights[:,1]**(1-y_true))*(weights[:,0]**(y_true))*K.binary_crossentropy(y_true, y_pred), axis=-1)



test_sequence = DataSequenceTest(df=test_df, batch_size = 32, mode ="image")


# Open a strategy scope.
strategy = tf.distribute.MirroredStrategy()

with strategy.scope():
    model_cnn = load_model('MODEL_PATH', custom_objects={"loss_fn":loss_fn})


yhat = model_cnn.predict(test_sequence, max_queue_size =30,
        use_multiprocessing=True, workers=10,verbose=1)

test_df['Pred_CNN'] = yhat
test_df.to_csv('OUTPUT_PATH')
