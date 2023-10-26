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
<li>Prevalence (Prvlnc) which are estimates of beneficiaries with that particular condition divided by the total number of beneficiaries. </li>
<li>Hospital Readmission Rate (Hosp_Readmsn_Rate) A 30-day readmission is an admission to an acute care hospital for any cause within 30 days of discharge from an acute care hospital.</li>
<li>Emergency Room Visits per 1,000 Beneficiaries (ER_Visits_Per_1000_Benes) Emergency department visits are presented as the number of visits per 1,000 beneficiaries.</li></ul>


<h2>Pipeline / Architecture</h2>
<p>The pipeline that I have decided to follow for this project is the “Data Pipeline 1 Batch – ML – Visualization” method. It meets the needs for my goal of building a system that collects, cleans, analyzes, and visualizes data. </p>
<p>The pipeline begins with the source data provided by Data.CMS.gov. The data will be imported from their database in the form of a JSON batch file. Multiple imports will have to be made for the previous years. The data will be imported by a command of a python document. Once the information is imported, the raw data will be stored in an Amazon S3 folder, serving as the data lake, in a large batch document in a pickle file format.
After the raw data has been stored, a Python script using the Panda’s library will clean and transform the data. The script will be searching for rows and columns that will not be used in the later stages of the pipeline. The script will also be cleaning the data of any outliers or missing data points. Once all the information has been cleaned the new data files will be stored in a separate Amazon S3 storage folder representing the Data Warehouse. </p>
<p>After the data has been transformed and stored, the pipeline uses the data to train a model in determining the level of chronic conditions in data that has not yet been provided. Once the model has been trained appropriately, then it will apply itself to the dataset, trend datasets that will be stored in new files inside Amazon’s S3 bucket. 
After all the trend data has been generated, the system will build a file for visualization software to point to. A dashboard will be created to display the historic data and future trend information. Interactive tools will be available to the user to drill down into state levels. </p>

<h2>Data Quality Assessment</h2>

<h2>Data Transformation Models</h2>

<h2>Infographic</h2>
<img src="https://jahgd.com//va/DataFlowChart.png" width="900px" height="auto" alt="Statewide Chronic Condition Trends Data Pipeline">

<h2>Code</h2>
<p>To download the code please follow this <a href="https://github.com/jhallgd/Statewide-Chronic-Condition-Trends.git" target="_blank">GitHub Link</a></p>

<h2>Investigation</h2> 



