# Online News Popularity Predictor
[Link to data set](https://www.kaggle.com/datasets/thehapyone/uci-online-news-popularity-data-set)

## Dataset
* __Filename:__ [OnlineNewsPopularity.csv](OnlineNewsPopularity.csv)
* __Source:__ [Kaggle](https://www.kaggle.com/datasets/thehapyone/uci-online-news-popularity-data-set)
* __Dataset Detail:__ [Link](OnlineNewsPopularity.names)

## Introduction
The number of shares an online news article gets is of great interest to news sites and advertisers alike since it serves as a good estimator of how many people an article will reach. Our data set includes various attributes that can potentially influence the number of shares, such as the number of links, pictures, and videos the article has; which day of the week the article was released, what category the article belongs to (business, entertainment, tech, etc), the number of positive or negative words, among many others. Applying the right machine learning model, we will be able to predict how popular an article will be. With accurate predictions, authors and news outlets will be able to make relevant changes to increase engagement from readers before release.

### Previous work


## Data Preprocessing

### Data Imputation
We found that our data did not have any null values. This can be examined from printing the results of .isnull().sum() (sums up the count of how many null values there are per column) and the heatmap generated in the notebook. 

Although we did not have any null values, we saw from our scatterplots that many features had outliers that could be problematic. We used an isolation forest model to remove these outliers.
More information on isolation forest [here][https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.IsolationForest.html#sklearn.ensemble.IsolationForest]

### Dropping Columns
Since the 'url' column and the 'timedelta' column have no effect on the number of shares, we dropped these columns when developing our model.

### Data Normalization
From our data distributions in our notebook, we see that only a few of our features (global_subjectivity, avg_positive_polarity) seem to be normally distributed. We also see from .describe() on our data set that our values are not scaled. Since only a few of our features are normally distributed, we rescaled our data using MinMaxScaler() so that our values are between 0 and 1.

### Data Encoding
Rather than predicting the exact number of shares an article will get, we split the number of shares into different classes as such:

    * Great Article: Shares >= 90%
    * Good Article: 70% <= Shares < 90%
    * Normal Article: 30% <= Shares < 70%
    * Bad article: Shares < 30%

For example, a share number that is above the 90th percentile out of our data would be classified as a "Great Article".
We used one-hot encoding to encode the continuous share values into binary values.

## Our Model:
Since simple regression models have already been done by other people on this data set, we set out to develop an ensemble model that takes different predictions from multiple machine learning models before making its decision. 

The models we used for building the ensemble model were:

    * A simple linear regression model
    * A neural network model
    * A model utilizing Random Forest
    * A SVM model

Our model then takes account of the predictions made by different models, and takes the mode (or mean) as the prediction value.

The model complexity of our ensemble model can be defined by the complexity of the models the ensemble is made of. For instance, an ensemble model that considers predictions from only linear regression models would be considered simple, while an ensemble consisting of neural network models and a random forest model more complex.

Note: During testing, we found that the SVM model did not perform well (around 40% accuracy). We excluded the SVM model from our list of models because it was inaccurate and also took a very long time to train.

## Our Models & Test for Overfitting
We tested three ensemble models. Ordered in increasing complexity, they were:

   * Model 1: Ensemble of 3 linear regression models (With ridge cost function)
   * Model 2: Ensemble of 1 linear regression model + 2 neural network models
   * model 3: Ensemble of 2 neural network models + 1 Random Forest model

The train/test accuracy for our models were as follows:

Model 1:
   * Training Accuracy: 0.4529
   * Testing Accuracy: 0.4414

Model 2:
   * Training Accuracy: 0.4740
   * Testing Accuracy: 0.4505

Model 3:
   * Training Accuracy: 0.8530
   * Testing Accuracy: 0.4561

While accuracy improves slightly as model complexity increases, we see that the difference between train/test accuracy increases greatly when going from Model 2 to Model 3. Based on these observations, we can conclude that Model 3 is overfitting and that the ideal complexity for our model would be somewhere inbetween Model 2 and Model 3.

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

## Result
<!-- Things need to be added here -->

## Contributions
<!-- Things need to be added here -->

## Link to Jupyter Notebook
[Jupyter Notebook](https://github.com/WeichenZhang89/ecs-171-ecs/blob/main/Data%20Preprocessing%20%26%20ML%20model.ipynb)
