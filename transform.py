import s3fs
from s3fs.core import S3FileSystem
import numpy as np
import pickle
import projectAdminInfo as PAI

def transform_data():

    s3 = S3FileSystem()

    # S3 bucket directory (data lake)
    # Get data from S3 bucket as a pickle file

    #Remove the incomplete CC datasets
    raw_data = np.load(s3.open('{}/{}'.format(PAI.dataLakeLocation, 'data.pkl')), allow_pickle=True)
    raw_data = raw_data[raw_data['Bene_Cond'] != 'Autism Spectrum Disorders']
    raw_data = raw_data[raw_data['Bene_Cond'] != 'HIV/AIDS']

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

            # Push cleaned data to S3 bucket warehouse
        with s3.open('{}/{}'.format(PAI.dataWareHouseLocation, str(counter) + "_" + cc[:3] + '.pkl'), 'wb') as f:
            f.write(pickle.dumps(newData))
        counter += 1



