import numpy as np
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.layers import LSTM, Dense, Dropout
from keras.models import Sequential
from keras.layers import Dense
from keras.models import load_model


def ingest_data():
    # The IDs for the available years.
    cmsIDs = {
        2018: '60ccbf1c-d3f5-4354-86a3-465711d81c5a',
        2017: '51231049-d7bc-41d2-90aa-07d711c375b2',
        2016: '547d7d2c-6667-46ca-9973-6c6e93f6a467',
        2015: 'f7675b78-2006-422c-96cf-dc22e3d22b90',
        2014: '40dab184-4531-4865-8d0b-3ba32d4ac3e9',
        2013: '3659b970-d9c1-4a8d-8052-9180598fd27c',
        2012: 'd88974a6-3214-4e09-8ae7-b56dc479125e',
        2011: 'dfad5b34-3ad8-45e5-833f-e53d9dbd7584',
        2010: 'c78b10cd-4a19-4468-abf7-0266d1c0dbb9',
        2009: 'b12072e9-e736-4bac-a794-ab3f842813c4',
        2008: 'ab51ced1-1984-4d66-b60a-47857fc48023',
        2007: '4b079921-8e18-463f-91cc-beb538004498'
    }

    # Two parts of the URL to Data.CMS.gov JSON API.
    dataCmsAptFront = 'https://data.cms.gov/data-api/v1/dataset/'
    dataCmsAptBack = '/data?filter[Bene_Age_Lvl]=All&filter[Bene_Geo_Lvl]=State&filter[Bene_Demo_Lvl]=All&filter[' \
                     'Bene_Demo_Desc]=All&offset=0&size=5000'

    # Add the first year.
    data = pd.read_json(dataCmsAptFront + str(cmsIDs[list(cmsIDs)[0]]) + dataCmsAptBack)
    data['Year'] = list(cmsIDs)[0]
    i = 1

    # Loop through the IDs to gather the data, and add a year identifier column.
    for cmsID in cmsIDs:
        tempData = pd.read_json(dataCmsAptFront + str(cmsIDs[cmsID]) + dataCmsAptBack)
        tempData['Year'] = cmsID
        data = pd.concat([data, tempData], ignore_index=True)

    # Adds information to the datalake

    with open('temp/datalake/data.pkl', 'wb') as f:
        pickle.dump(data, f)


# Collect

def transform_data():
    # Get data from the datawarehouse
    raw_data = np.load('temp/datalake/data.pkl', allow_pickle=True)
    raw_data = raw_data[raw_data['Bene_Cond'] != 'Autism Spectrum Disorders']
    raw_data = raw_data[raw_data['Bene_Cond'] != 'HIV/AIDS']

    # Get the list of states
    stateList = raw_data.Bene_Geo_Desc.unique()

    # Get the list of chronic conditions
    chronicConditions = raw_data.Bene_Cond.unique()

    # Counter used to keep all files names unique.
    counter = 1

    for cc in chronicConditions:
        data = raw_data[raw_data['Bene_Cond'] == cc]

        # Drop rows with N/A
        data = data.dropna(axis=0, how='any')

        data = data[data['Prvlnc'] != '']
        data = data[data['Tot_Mdcr_Stdzd_Pymt_PC'] != '']
        data = data[data['Tot_Mdcr_Pymt_PC'] != '']
        data = data[data['Hosp_Readmsn_Rate'] != '']
        data = data[data['ER_Visits_Per_1000_Benes'] != '']

        # Drop duplicate rows
        data = data.drop_duplicates()

        # Drop Unknown State
        data = data[data['Bene_Geo_Desc'] != 'Unknown']

        # Drop incomplete datasets
        data = data[data['Bene_Geo_Desc'] != 'Virgin Islands']
        data = data[data['Bene_Geo_Desc'] != 'Puerto Rico']
        data = data[data['Bene_Geo_Desc'] != 'District of Columbia']

        # Assign Proper DataTypes
        data['ER_Visits_Per_1000_Benes'] = data['ER_Visits_Per_1000_Benes'].astype(float)
        data['Bene_Geo_Desc'] = data['Bene_Geo_Desc'].astype(str)

        # Transpose the Year column
        newData = data[['Bene_Geo_Desc', 'ER_Visits_Per_1000_Benes']][data['Year'] == 2018]
        newData = newData.rename(columns={'ER_Visits_Per_1000_Benes': '2018'})
        tempData = data[['Bene_Geo_Desc', 'ER_Visits_Per_1000_Benes']][data['Year'] == 2017]
        tempData = tempData.rename(columns={'ER_Visits_Per_1000_Benes': str(2017)})
        newData = newData.set_index('Bene_Geo_Desc').join(tempData.set_index('Bene_Geo_Desc'))

        for tempDate in range(2016, 2007, -1):
            tempData = data[['Bene_Geo_Desc', 'ER_Visits_Per_1000_Benes']][data['Year'] == tempDate]
            tempData = tempData.rename(columns={'ER_Visits_Per_1000_Benes': str(tempDate)})
            newData = newData.join(tempData.set_index('Bene_Geo_Desc'))

        with open('temp/datawarehouse/' + str(counter) + "_" + cc[:3] + '.pkl', 'wb') as f:
            pickle.dump(newData, f)
        counter += 1

# Transform

def feature_extract():
    lake_data = np.load('temp/datalake/data.pkl', allow_pickle=True)
    lake_data = lake_data[lake_data['Bene_Cond'] != 'Autism Spectrum Disorders']
    lake_data = lake_data[lake_data['Bene_Cond'] != 'HIV/AIDS']

    # Build Files names from Lake
    fileNames = []
    chronicConditions = lake_data.Bene_Cond.unique()

    counter = 1
    for cc in chronicConditions:
        fileNames.append(str(counter) + "_" + cc[:3] + '.pkl')
        counter += 1

    warehouseData = []

    # Get data from S3 bucket as a pickle file and store them in a list
    for fn in fileNames:
        tempDF = np.load('temp/datawarehouse/' + fn, allow_pickle=True)
        warehouseData.append(tempDF)

    features = ['2008', '2009', '2010', '2011', '2012', '2013', '2014', '2015', '2016', '2017']

    for fe in range(0, len(warehouseData)):
        train, test = train_test_split(warehouseData[fe], test_size=0.25, random_state=0, shuffle=False)
        X_train = train[features]
        X_test = test[features]
        y_train = train['2018']
        y_test = test['2018']

        # Push extracted features to data warehouse
        with open('temp/test_train/X_train_' + fileNames[fe], 'wb') as f:
            f.write(pickle.dumps(X_train))
        with open('temp/test_train/X_test_' + fileNames[fe], 'wb') as f:
            f.write(pickle.dumps(X_test))
        with open('temp/test_train/Y_train_' + fileNames[fe], 'wb') as f:
            f.write(pickle.dumps(y_train))
        with open('temp/test_train/Y_test_' + fileNames[fe], 'wb') as f:
            f.write(pickle.dumps(y_test))

# Feature Extraction

def build_train():
    lake_data = np.load('temp/datalake/data.pkl', allow_pickle=True)
    lake_data = lake_data[lake_data['Bene_Cond'] != 'Autism Spectrum Disorders']
    lake_data = lake_data[lake_data['Bene_Cond'] != 'HIV/AIDS']

    # Build Files names from Lake
    fileNames = []
    chronicConditions = lake_data.Bene_Cond.unique()

    counter = 1
    for cc in chronicConditions:
        fileNames.append(str(counter) + "_" + cc[:3] + '.pkl')
        counter += 1

    # Build Training Files
    for fn in fileNames:
        # Load the X/Y Training and Testing Files
        X_TRAIN_1 = np.load('temp/test_train/' + 'X_train_' + fn, allow_pickle=True)
        X_TEST_1 = np.load('temp/test_train/' + 'X_test_' + fn, allow_pickle=True)
        Y_TRAIN_1 = np.load('temp/test_train/' + 'Y_train_' + fn, allow_pickle=True)
        Y_TEST_1 = np.load('temp/test_train/' + 'Y_test_' + fn, allow_pickle=True)

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
        lstm_TEMP.save('temp/MILearning/' + fn[:5] + '.h5')

# Build Train

def predict():
    predictYears = ['2019', '2020', '2021', '2022', '2023', '2024', '2025']
    features = ['2017', '2016', '2015', '2014', '2013', '2012', '2011', '2010', '2009', '2008']

    lake_data = np.load('temp/datalake/data.pkl', allow_pickle=True)
    lake_data = lake_data[lake_data['Bene_Cond'] != 'Autism Spectrum Disorders']
    lake_data = lake_data[lake_data['Bene_Cond'] != 'HIV/AIDS']

    # Build Files names from Lake
    fileNames = []
    chronicConditions = lake_data.Bene_Cond.unique()

    counter = 1
    for cc in chronicConditions:
        fileNames.append(str(counter) + "_" + cc[:3] + '.pkl')
        counter += 1

    for fn in fileNames:
        tempData_full = np.load('temp/datawarehouse/' + fn, allow_pickle=True)
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
            data = data.rename(columns={'2018': tempYear})
            data = data[tempYear]

            # Join both sets
            tempData_full = tempData_full.join(data)

            # update focus column
            del focusColumns[9]
            focusColumns.insert(0, tempYear)

        with open('temp/predicted/' + fn, 'wb') as f:
            pickle.dump(tempData_full, f)

# Predict

def load_data():
    codeLookup = {}

    # Build large dataframe
    lake_data = np.load('temp/datalake/data.pkl', allow_pickle=True)
    lake_data = lake_data[lake_data['Bene_Cond'] != 'Autism Spectrum Disorders']
    lake_data = lake_data[lake_data['Bene_Cond'] != 'HIV/AIDS']

    # Build Files names from Lake
    fileNames = []
    chronicConditions = lake_data.Bene_Cond.unique()

    counter = 1
    for cc in chronicConditions:
        fileNames.append(str(counter) + "_" + cc[:3] + '.pkl')
        codeLookup[str(counter) + "_" + cc[:3] + '.pkl'] = cc
        counter += 1

    dataAll = pd.DataFrame()

    for fn in fileNames:
        tempDF = np.load('temp/predicted/' + fn, allow_pickle=True)
        tempDF['Chronic Condition'] = codeLookup[fn]
        for year in range(2009, 2026):
            if year > 2018:
                predict = 'Prediction'
            else:
                predict = 'CMS Data'
            tempDFYear = tempDF[['Chronic Condition', str(year)]]
            tempDFYear = tempDFYear.rename(columns={str(year): 'ER_Visits_Per_1000_Benes'})
            tempDFYear['Year'] = year
            tempDFYear['Data Type'] = predict
            dataAll = dataAll._append(tempDFYear)

    with open('temp/load/' + 'data.pkl', 'wb') as f:
        pickle.dump(dataAll, f)


def convert():
    object = np.load('temp/load/' + 'data.pkl', allow_pickle=True)
    df = pd.DataFrame(object)
    df.to_csv('temp/DataVisualization/vis-data.csv', index=False)

# Perform

ingest_data()
transform_data()
feature_extract()
build_train()
predict()
load_data()
convert()
