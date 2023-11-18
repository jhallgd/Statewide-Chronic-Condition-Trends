from sqlalchemy import create_engine
import numpy as np
import pandas as pd
from s3fs.core import S3FileSystem
import projectAdminInfo as PAI
import tempfile


def load_data():
    s3 = S3FileSystem()

    #Build large dataframe
    fileNames = PAI.getFileNames()

    dataAll = pd.DataFrame()
    codeLookup = PAI.getFileNameDictionary()

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

    with open('temp/load/' + 'vis-data.csv', 'wb') as f:
        pickle.dump(dataAll, f)

    with s3.open('{}/{}'.format(PAI.visualization, 'vis-data.csv'), 'wb') as f:
        f.write(pickle.dumps(newData))

    #Save the file to the s3 Bucket
    tempFileName = 'vis-data.csv'
    with tempfile.TemporaryDirectory() as tempdir:
        dataAll.to_csv(f"{tempdir}/{tempFileName}")
        # Push saved temporary model to S3
        s3.put(f"{tempdir}/{tempFileName}", f"{PAI.visualization}/{tempFileName}")