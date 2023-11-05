import s3fs
from s3fs.core import S3FileSystem
import numpy as np


#Enter your S3 directory of your data Lake. Without the last "/"
dataLakeLocation = ""

#Enter your S3 directory of your data warehouse. Without the last "/"
dataWareHouseLocation = ""

#Enter your S3 directory of your train test folder here. Without the last "/"
train_test = ""

#Enter your S3 directory of your training model folder here. Without the last "/"
model = ""

#Enter your S3 directory of your prediction folder here. Without the last "/"
prediction = ""

#Shared Functions
def getFileNames():
    s3 = S3FileSystem()
    lake_data = np.load(s3.open('{}/{}'.format(dataLakeLocation, 'data.pkl')), allow_pickle=True)

    # Build Files names from Lake
    fileNames = []
    chronicConditions = lake_data.Bene_Cond.unique()

    counter = 1
    for cc in chronicConditions:
        fileNames.append(str(counter) + "_" + cc[:3] + '.pkl')
        counter += 1
    return fileNames