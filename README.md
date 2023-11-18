<h1>Statewide Chronic Condition Trends</h2>

<h2>Project’s Function</h2>
<p>The lifecycle of the Statewide Chronic Condition Trend data pipeline is to collect, clean, store, analyze, predict, and visualize health care data throughout the United States. The main goal is to provide future trends of chronic conditions based off historic data and visualize the information in an interactive dashboard. As more data is available, the pipeline can add new information then use the information to train its algorithm to make better predictions with future data. Once completed this pipeline could be used to help determine future trends across the country and help populations with growing levels of chronic conditions. </p>
<p>Once the pipeline is complete, the goal is to have an integrated data pipeline process. Once that runs all process back-to-back without the need of break data flow. The system will make data driven decisions correctly and provide high value trends and fill the whole of unavailable data sets. I also hope to gain a better understanding of the data pipeline process and gain knowledge to build similar or more advanced systems in the future. </p>

<h2>Dataset</h2>
<p>This pipeline utilizes the Specific Chronic Conditions dataset provided by the Centers for Medicare & Medicaid Services. The dataset stores information about 21 chronic conditions among Original Medicare beneficiaries. The information is stored by state and by age group, for the purposes of this project only state information has been used.</p> 
<p>Chronic Conditions:</p><ul>
<li>Alcohol Abuse Drug Abuse/ Substance Abuse</li>
<li>Alzheimer’s Disease and Related Dementia</li>
<li>Arthritis (Osteoarthritis and Rheumatoid) </li>
<li>Asthma </li>
<li>Atrial Fibrillation </li>
<li>Autism Spectrum Disorders </li>
<li>Cancer (Breast, Colorectal, Lung, and Prostate)</li>
<li>Chronic Kidney Disease </li>
<li>Chronic Obstructive Pulmonary Disease </li>
<li>Depression </li>
<li>Diabetes  </li>
<li>Drug Abuse/ Substance Abuse</li>
<li>Heart Failure</li>
<li>Hepatitis (Chronic Viral B & C)</li>
<li>HIV/AIDS</li>
<li>Hyperlipidemia (High cholesterol)</li>
<li>Hypertension (High blood pressure)</li>
<li>Ischemic Heart Disease</li>
<li>Osteoporosis</li>
<li>Schizophrenia and Other Psychotic Disorders</li>
<li>Stroke</li></ul>
<p>To view the full CMS data dictionary please <a href="https://data.cms.gov/resources/specific-chronic-conditions-data-dictionary" target="_blank">click here</a>. The key attributes that I will be focusing it this project are: </p><ul>
<li>Beneficiary Geographic Code (Bene_Geo_Cd) the State where the beneficiary resides.</li>
<li>Beneficiary Chronic Condition (Bene_Cond) the Chronic condition identifier. </li>
<li>Emergency Room Visits per 1,000 Beneficiaries (ER_Visits_Per_1000_Benes) Emergency department visits are presented as the number of visits per 1,000 beneficiaries.</li></ul>


<h2>Pipeline / Architecture</h2>
<p>The pipeline that I have decided to follow for this project is the “Data Pipeline 1 Batch – ML – Visualization” method. It meets the needs for my goal of building a system that collects, cleans, analyzes, and visualizes data. </p>
<p>The pipeline begins with the source data provided by Data.CMS.gov. The data will be imported from their database in the form of a JSON batch file. Multiple imports will have to be made for the previous years. The data will be imported by a command of a python document. Once the information is imported, the raw data will be stored in an Amazon S3 folder, serving as the data lake, in a large batch document in a pickle file format.<br>
After the raw data has been stored, a Python script using the Panda’s library will clean and transform the data. The script will be searching for rows and columns that will not be used in the later stages of the pipeline. The script will also be cleaning the data of any outliers or missing data points. The data set will also be rearange into a better format for machine learning. The files will be grouped by chronic condition and store all the previous years. Once all the information has been cleaned and transfromed the new data files will be stored in a separate Amazon S3 storage folder representing the Data Warehouse. </p>
<p>The pipeline then uses the new data to train a model in determining the level of chronic conditions in data that has not yet been provided. The predicitons will be trained using historical data to predict the upcomming year. Once the model has been trained appropriately, then it will apply itself to the dataset, the trend datasets that will be stored in new files inside Amazon’s S3 bucket. 
After all the trend data has been generated, the system will build a file for visualization software to point to. A dashboard will display the historic data and future trend information. Interactive tools will be available to the user to drill down into state levels. </p>

<p><b>Pipeline Tools</b></p>
<ul>
  <li><a href="https://aws.amazon.com/ec2/" target="_blank">Amazon Elastic Compute Cloud</a></li>
  <li><a href="https://aws.amazon.com/pm/serv-s3/" target="_blank">Amazon S3 Cloud Storage</a></li>
  <li><a href="https://www.docker.com/" target="_blank">Docker</a></li>
  <li><a href="https://airflow.apache.org/" target="_blank">Apache Airflow</a></li>
  <li><a href="https://www.python.org/" target="_blank">Python</a></li>
  <li><a href="https://pandas.pydata.org/" target="_blank">Pandas</a></li>
  <li><a href="https://numpy.org/" target="_blank">NumPy</a></li>
  <li><a href="https://keras.io/" target="_blank">Keras</a></li>
  <li><a href="https://www.tableau.com/" target="_blank">Tableau</a></li>
</ul>

<h2>Data Quality Assessment</h2>
<p>I chose this data set because I believe the Centers for Medicare & Medicaid Services can provide accurate, complete, valid, consistent information regarding Medicare beneficiaries. Using this information at the core of my pipeline will ensure that I start with high quality data, that will be used to make data driven forecasting. </p>

<h2>Data Transformation Models</h2>
<p>To make the predictions a transformation model was trained using the previous year’s information. It was split to create training and testing data sets. Once the models were trained they were applied to the data sets to create the year that are not current available. </p>
<p>To execute the code, fill in the appropriate S3 bucket information in the “projectAdminInfo.py” file. The other files can be run as is. </p>

<h2>Infographic</h2>
<img src="https://jahgd.com//va/DataFlowChart.png" width="900px" height="auto" alt="Statewide Chronic Condition Trends Data Pipeline">

<h2>Code</h2>
<p>To download the code please follow this <a href="https://github.com/jhallgd/Statewide-Chronic-Condition-Trends.git" target="_blank">GitHub Link</a></p>

<h2>Investigation</h2> 
<p>I believe that this pipeline is a great first step into predicting Chronic Conditions throughout the country. I am happy with the results of this pilot project. I believe that it gives a slight indication of the future trends of chronic conditions. I hope that with more time and information it can grow and give value to communities.
 To grow this project will require more information than possible from other sources. Information could include population statistics, environmental factors, quality of life, and other health informatics. Using this information, with more server resources, could help provide a more accurate reading of future data. Finding create ways of what factors could affect the chronic conditions could provide innovative results. </p>
<p>Large events, like COVID, will have a big impact on how the pipeline can perform with accuracy. As of 2023, the information is not available through CMS. I would be interested to see how that would affect the training models. </p> 
<p>As the next step, I would begin by adding population data to the pipeline to help train and build the learning algorithms and help determine the future trends. 
</p>



