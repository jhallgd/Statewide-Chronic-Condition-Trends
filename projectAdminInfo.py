import s3fs
from s3fs.core import S3FileSystem
import numpy as np

#Enter Bucket Name here
bucketName = "ece5984-bucket-jhallgd"

#Enter your S3 directory of your data Lake without the last "/"
dataLakeLocation = "s3://ece5984-bucket-jhallgd/CCProject/DataLake"

#Enter your S3 directory of your data warehouse without the last "/"
dataWareHouseLocation = "s3://ece5984-bucket-jhallgd/CCProject/DataWarehouse"

#Enter your S3 directory of your train test folder here. Without the last "/"
train_test = "s3://ece5984-bucket-jhallgd/CCProject/TestTrain"

#Enter your S3 directory of your training model folder here. Without the last "/"
model = "s3://ece5984-bucket-jhallgd/CCProject/Models"

#Enter the model Directory folder
modelDirectory = "CCProject/Models/"

#Enter your S3 directory of your prediction folder here. Without the last "/"
prediction = "s3://ece5984-bucket-jhallgd/CCProject/Predictions"

#Database information
visualization = "s3://ece5984-bucket-jhallgd/CCProject/DataVisualization"

# Shared Functions
def getFileNames():
    s3 = S3FileSystem()
    lake_data = np.load(s3.open('{}/{}'.format(dataLakeLocation, 'data.pkl')), allow_pickle=True)

    # Remove Incomplete Data Sets
    lake_data = lake_data[lake_data['Bene_Cond'] != 'Autism Spectrum Disorders']
    lake_data = lake_data[lake_data['Bene_Cond'] != 'HIV/AIDS']

    # Build Files names from Lake
    fileNames = []
    chronicConditions = lake_data.Bene_Cond.unique()

    counter = 1
    for cc in chronicConditions:
        fileNames.append(str(counter) + "_" + cc[:3] + '.pkl')
        counter += 1
    return fileNames


def getFileNameDictionary():
    s3 = S3FileSystem()
    lake_data = np.load(s3.open('{}/{}'.format(dataLakeLocation, 'data.pkl')), allow_pickle=True)

    # Remove Incomplete Data Sets
    lake_data = lake_data[lake_data['Bene_Cond'] != 'Autism Spectrum Disorders']
    lake_data = lake_data[lake_data['Bene_Cond'] != 'HIV/AIDS']

    # Build Files names from Lake
    codeLookup = {}
    chronicConditions = lake_data.Bene_Cond.unique()

    counter = 1
    for cc in chronicConditions:
        codeLookup[str(counter) + "_" + cc[:3] + '.pkl'] = cc
        counter += 1
    return codeLookup