import s3fs
from s3fs.core import S3FileSystem
import numpy as np
import pickle
import projectAdminInfo as PAI
from keras.models import load_model


def predict():

    s3 = S3FileSystem()

    predictYears = ['2019', '2020', '2021', '2022', '2023', '2024', '2025']
    features = ['2017', '2016', '2015', '2014', '2013', '2012', '2011', '2010', '2009', '2008']

    fileNames = PAI.getFileNames()

    for fn in fileNames:
        tempData_full = np.load(s3.open('{}/{}'.format(PAI.dataWareHouseLocation, fn)), allow_pickle=True)
        focusColumns = ['2009', '2010', '2011', '2012', '2013', '2014', '2015', '2016', '2017', '2018']
        tempData_full = tempData_full[focusColumns]
        for tempYear in predictYears:
            data = tempData_full[focusColumns]
            data.columns = features
            model = load_model('temp/MILearning/' + fn[:5] + '.h5')
            data_temp = data
            data_array = np.array(data_temp)
            data_reshape = data_array.reshape(data_array.shape[0], 1, data_array.shape[1])
            prediction = model.predict(data_reshape)
            data[tempYear] = prediction
            data = data.rename(columns={'2018':tempYear})
            data = data[tempYear]

            #Join both sets
            tempData_full = tempData_full.join(data)

            #update focus column
            del focusColumns[9]
            focusColumns.insert(0, tempYear)

        with s3.open('{}/{}'.format(PAI.prediction, fn), 'wb') as f:
            f.write(pickle.dumps(tempData_full))