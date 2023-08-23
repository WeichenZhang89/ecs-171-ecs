## Online News Popularity Predictor
We use OnlineNewsPopularity.csv as our dataset to create a machine learning model to predict popularity of online news articles.

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
<!-- Things need to be added here -->
The different features of our data are not scaled, as we can see from running .describe() on our data. We can also see from our scatterplots that most of our features are not normally distributed, so we will be normalizing our data to scale it from 0 to 1 instead of standardizing. 

In terms of null values, we found that our data set did not have any null values. This can be examined from printing the results of .isnull().sum() (sums up the count of how many null values there are per column) and the heatmap generated in the notebook.

## How to use this project
1. __Clone the repo:__ `git clone https://github.com/WeichenZhang89/ecs-171-ecs`
2. __Install libraries:__ `pip install pandas tensorflow matplotlib seaborn scikit-learn (on Terminal)`
3. __Install libraries:__ `!pip install pandas tensorflow matplotlib seaborn scikit-learn (on colab)`
4. __Run model:__ `python ML_model.ipynb`
5. <!-- Things need to be added here -->

## Result
<!-- Things need to be added here -->

## Contributions
<!-- Things need to be added here -->

## Link to Jupyter Notebook
[Jupyter Notebook](ML_model.ipynb)
