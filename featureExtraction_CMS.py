import s3fs
from s3fs.core import S3FileSystem
import pickle
import numpy as np
from sklearn.model_selection import train_test_split
import projectAdminInfo as PAI


def feature_extract():

    s3 = S3FileSystem()

    # Build Files names from Lake
    fileNames = PAI.getFileNames()

    warehouseData = []

    # Get data from S3 bucket as a pickle file and store them in a list
    for fn in fileNames:
        tempDF = np.load(s3.open('{}/{}'.format(PAI.dataWareHouseLocation, fn)), allow_pickle=True)
        warehouseData.append(tempDF)

    features = ['2008', '2009', '2010', '2011', '2012', '2013', '2014', '2015', '2016', '2017']

    # Feature Extraction
    for fe in range(0, len(warehouseData)):
        train, test = train_test_split(warehouseData[fe], test_size=0.25, random_state=0, shuffle=False)
        X_train = train[features]
        X_test = test[features]
        y_train = train['2018']
        y_test = test['2018']

        # Push extracted features to data warehouse
        with s3.open('{}/{}'.format(PAI.train_test, 'X_train_' + fileNames[fe]), 'wb') as f:
            f.write(pickle.dumps(X_train))
        with s3.open('{}/{}'.format(PAI.train_test, 'X_test_' + fileNames[fe]), 'wb') as f:
            f.write(pickle.dumps(X_test))
        with s3.open('{}/{}'.format(PAI.train_test, 'Y_train_' + fileNames[fe]), 'wb') as f:
            f.write(pickle.dumps(y_train))
        with s3.open('{}/{}'.format(PAI.train_test, 'Y_test_' + fileNames[fe]), 'wb') as f:
            f.write(pickle.dumps(y_test))




