import s3fs
from s3fs.core import S3FileSystem
import numpy as np
from keras.layers import LSTM, Dense, Dropout
from keras.models import Sequential
from keras.layers import Dense
import projectAdminInfo as PAI
import tempfile

def build_train():

    s3 = S3FileSystem()

    # Build Files names from Lake
    fileNames = PAI.getFileNames()

    # Build Training Files
    for fn in fileNames:
        # Load the X/Y Training and Testing Files
        X_TRAIN_1 = np.load(s3.open('{}/{}'.format(PAI.train_test, 'X_train_' + fn)), allow_pickle=True)
        X_TEST_1 = np.load(s3.open('{}/{}'.format(PAI.train_test, 'X_test_' + fn)), allow_pickle=True)
        Y_TRAIN_1 = np.load(s3.open('{}/{}'.format(PAI.train_test, 'Y_train_' + fn)), allow_pickle=True)
        Y_TEST_1 = np.load(s3.open('{}/{}'.format(PAI.train_test, 'Y_test_' + fn)), allow_pickle=True)

        X_TRAIN_2 = np.array(X_TRAIN_1)
        X_TEST_2 = np.array(X_TEST_1)
        X_TRAIN_1 = X_TRAIN_2.reshape(X_TRAIN_1.shape[0], 1, X_TRAIN_1.shape[1])
        X_TEST_1 = X_TEST_2.reshape(X_TEST_1.shape[0], 1, X_TEST_1.shape[1])

        # Building the LSTM Model
        lstm_TEMP = Sequential()
        lstm_TEMP.add(LSTM(32, input_shape=(1, X_TRAIN_2.shape[1]), activation='linear', return_sequences=False))
        lstm_TEMP.add(Dense(1))
        lstm_TEMP.compile(loss='mean_squared_error', optimizer='rmsprop', metrics=["accuracy"])

        # Model Training
        lstm_TEMP.fit(x=X_TRAIN_1, y=Y_TRAIN_1, epochs=100, batch_size=64, verbose=0, shuffle=False,
                      validation_data=(X_TEST_1, Y_TEST_1))

        # Save model temporarily
        tempH5FileName = fn[:5] + '.h5'
        with tempfile.TemporaryDirectory() as tempdir:
            lstm_TEMP.save(f"{tempdir}/{tempH5FileName}")
            # Push saved temporary model to S3
            s3.put(f"{tempdir}/{tempH5FileName}", f"{PAI.model}/{tempH5FileName}")
