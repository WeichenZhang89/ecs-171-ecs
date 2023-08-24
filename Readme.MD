# Online News Popularity Predictor
We use OnlineNewsPopularity.csv as our dataset to create a machine learning model to predict popularity of online news articles.

## Goals
The number of shares an online news article gets is of great interest to news sites and advertisers alike since it serves as a good estimator of how many people an article will reach. Our data set includes various attributes that can potentially influence the number of shares, such as the number of links, pictures, and videos the article has; which day of the week the article was released, what category the article belongs to (business, entertainment, tech, etc), the number of positive or negative words, among many others. Applying the right machine learning model, we will be able to accurately predict how popular an article will be, allowing authors, news outlets (and even you!) to make relevant changes to increase engagement from readers before release.

## Project Overview
<!-- Things need to be added here -->

## Dataset
* __Filename:__ [OnlineNewsPopularity.csv](OnlineNewsPopularity.csv)
* __Source:__ [Kaggle](https://www.kaggle.com/datasets/thehapyone/uci-online-news-popularity-data-set)
* __Dataset Detail:__ [Link](OnlineNewsPopularity.names)
* __Features:__ 
    * <!-- Things need to be added here -->
    * <!-- Things need to be added here -->
    * <!-- Things need to be added here -->
    * <!-- Things need to be added here -->

## Data preprocessing

### Data Imputation
We found that our data did not have any null values. This can be examined from printing the results of .isnull().sum() (sums up the count of how many null values there are per column) and the heatmap generated in the notebook. While we see no need to impute values for now, it may be useful to impute/remove some values that are deemed to be outliers. As we train and evaluate our model, we may revisit data imputation to handle outliers in order to increase the accuracy of our model.

### Data Normalization
From our data distributions in our notebook, we see that only a few of our features (global_subjectivity, avg_positive_polarity) seem to be normally distributed. We also see from .describe() on our data set that our values are not scaled. Since only a few of our features are normally distributed, we will have to normalize our data from 0 to 1 instead of standardizing.

### Data Encoding
Rather than predicting the exact number of shares an article will get, we plan to split the number of shares into different classes (Not Interesting, Slightly Interesting, Interestring, Very Interestring, for example). In order to do this, we will have to split the number of shares into different classes and label encode or one-hot encode them.

## How to use this project
1. __Clone the repo:__ `git clone https://github.com/WeichenZhang89/ecs-171-ecs`
2. __Install libraries:__ `pip install pandas tensorflow matplotlib seaborn scikit-learn (on Terminal)`
3. __Install libraries:__ `!pip install pandas tensorflow matplotlib seaborn scikit-learn (on colab)`
4. __Run model:__ `python ML_model.ipynb`

   You might encounter "ImportError : No Moduled Name "tensorflow" when running "import tensorflow as tf". 

   To solve this, you can uninstall tensorflow and then reinstall it by pip uninstall tensorflow + pip install tensorflow in terminal.

   If you encounter "ERROR: Could not install packages due to an OSError: [Errno 2] No such file or directory" 

   and "HINT: This error might have occurred since this system does not have Windows Long Path support enabled"

   the most easy way to solve this is going into the registry editor with this directory "Computer\HKEY_LOCAL_MACHINE\SYSTEM\CurrentControlSet\Control\FileSystem", 

   and thening edit "LongPathsEnabled" to a value of 1. Then it should be done.
5. <!-- Things need to be added here -->

## Result
<!-- Things need to be added here -->

## Contributions
<!-- Things need to be added here -->

## Link to Jupyter Notebook
[Jupyter Notebook](ML_model.ipynb)
