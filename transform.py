import s3fs
from s3fs.core import S3FileSystem
import numpy as np
import pickle
import projectAdminInfo as PAI

def transform_data():

    s3 = S3FileSystem()

    # S3 bucket directory (data lake)
    # Get data from S3 bucket as a pickle file
    raw_data = np.load(s3.open('{}/{}'.format(PAI.dataLakeLocation, 'data.pkl')), allow_pickle=True)

    # Get the list of states
    stateList = raw_data.Bene_Geo_Desc.unique()

    # Get the list of chronic conditions
    chronicConditions = raw_data.Bene_Cond.unique()

    # Split by state
    for state in stateList:
        stateData = raw_data[raw_data['Bene_Geo_Desc'] == state]

        # Split by chronic condition and save.

        # Counter used to keep all files names unique.
        counter = 1

        for cc in chronicConditions:
            data = stateData[stateData['Bene_Cond'] == cc]
            # Drop rows with N/A
            data = data.dropna(axis=0, how='any')

            # Drop duplicate rows
            data = data.drop_duplicates()

            # Drop Unknown State
            data = data[data['Bene_Geo_Desc'] != 'Unknown']

            # Push cleaned data to S3 bucket warehouse
            with s3.open('{}/{}'.format(PAI.dataWareHouseLocation, state + "_" +str(counter) + "_" + cc[:3] + '.pkl'), 'wb') as f:
                f.write(pickle.dumps(data))
            counter += 1



