import s3fs
from s3fs.core import S3FileSystem
import numpy as np

#Enter Bucket Name here
bucketName = ""

#Enter your S3 directory of your data Lake. Without the last "/"
dataLakeLocation = ""

#Enter your S3 directory of your data warehouse. Without the last "/"
dataWareHouseLocation = ""

#Enter your S3 directory of your train test folder here. Without the last "/"
train_test = ""

#Enter your S3 directory of your training model folder here. Without the last "/"
model = ""

#Enter model directory of your training model folder here. Without the last "/"
modelDirectory = ""

#Enter your S3 directory of your prediction folder here. Without the last "/"
prediction = ""

#Enter your S3 directory of your prediction folder here. Without the last "/"
visualization = ""



#Shared Functions
def getFileNames():
    s3 = S3FileSystem()
    lake_data = np.load(s3.open('{}/{}'.format(dataLakeLocation, 'data.pkl')), allow_pickle=True)

    #Remove Incomplete Data Sets
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

    #Remove Incomplete Data Sets
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