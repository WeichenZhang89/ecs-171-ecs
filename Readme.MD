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

One aspect that needs further consideration when building our model is the mix of continuous and binary values in our feature. For example, the feature 'num_imgs' is a continuous independent variablke specifying how many images there are in an article, but the feature 'data_chanel_is_bus' is a binary variable that can only be 0 or 1. Since we are planning to scale all of our data from 0 to 1, we will need to take this into consideration when building/evaluating our model.

As for the model, we plan to build a neural network that takes into consideration the mix of continuous and logistic dependent variables as features.

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
