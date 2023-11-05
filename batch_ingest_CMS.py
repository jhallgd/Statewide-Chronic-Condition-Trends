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

    # Two parts of the URL to Data.CMS.gov JSON API.
    dataCmsAptFront = 'https://data.cms.gov/data-api/v1/dataset/'
    dataCmsAptBack = '/data?filter[Bene_Age_Lvl]=All&filter[Bene_Geo_Lvl]=State&filter[Bene_Demo_Lvl]=All&filter[Bene_Demo_Desc]=All&offset=0&size=5000'

    #Add the first year.
    data = pd.read_json(dataCmsAptFront + str(cmsIDs[list(cmsIDs)[0]]) + dataCmsAptBack)
    data['Year'] = list(cmsIDs)[0]
    i = 1

    # Loop through the IDs to gather the data, and add a year identifier column.
    for cmsID in cmsIDs:
        tempData = pd.read_json(dataCmsAptFront + str(cmsIDs[cmsID]) + dataCmsAptBack)
        tempData['Year'] = cmsID
        data = pd.concat([data, tempData], ignore_index=True)

    # Adds information to the datalake
    s3 = S3FileSystem()

    #Push data to S3 bucket as a pickle file
    with s3.open('{}/{}'.format(PAI.dataLakeLocation, 'data.pkl'), 'wb') as f:
        f.write(pickle.dumps(data))