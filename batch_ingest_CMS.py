import s3fs
from s3fs.core import S3FileSystem
import numpy as np
import pickle
import pandas as pd
import projectAdminInfo as PAI


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

    # The URL to Data.CMS.gov's JSON API.
    dataCmsAptFront = 'https://data.cms.gov/data-api/v1/dataset/'
    dataCmsAptBack = '/data?keyword=State&offset=0&size=500000'

    # Create Blank DataFrame to store all information.
    columnHeaders = ['',
                     'Bene_Geo_Lvl',
                     'Bene_Geo_Desc',
                     'Bene_Geo_Cd',
                     'Bene_Age_Lvl',
                     'Bene_Demo_Lvl',
                     'Bene_Demo_Desc',
                     'Bene_Cond',
                     'Prvlnc',
                     'Tot_Mdcr_Stdzd_Pymt_PC',
                     'Tot_Mdcr_Pymt_PC',
                     'Hosp_Readmsn_Rate	ER_Visits_Per_1000_Benes',
                     'Year'
]
    data = pd.DataFrame()
    i = 0
    while i < len(columnHeaders):
        data.insert(i, columnHeaders[i], [""])
        i+=1
    data = data.drop([0])

    # Loop through the IDs to gather the data, and add a year identifier column.
    for cmsID in cmsIDs:
        tempData = pd.read_json(dataCmsAptFront + str(cmsIDs[cmsID]) + dataCmsAptBack)
        tempData['Year'] = cmsID
        tempData.columns = columnHeaders
        data = pd.concat([data, tempData], ignore_index=True)

    # Adds information to the datalake
    s3 = S3FileSystem()
    #S3 bucket directory
    DIR = PAI.dataLakeLocation
    #Push data to S3 bucket as a pickle file
    with s3.open('{}/{}'.format(DIR, 'data.pkl'), 'wb') as f:
        f.write(pickle.dumps(data))

ingest_data()