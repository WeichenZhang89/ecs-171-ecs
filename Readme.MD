# Online News Popularity Predictor
## Dataset
* __Filename:__ [OnlineNewsPopularity.csv](OnlineNewsPopularity.csv)
* __Source:__ [Kaggle](https://www.kaggle.com/datasets/thehapyone/uci-online-news-popularity-data-set)
* __Dataset Detail:__ [Link](OnlineNewsPopularity.names)


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

   and then edit "LongPathsEnabled" to a value of 1. Then it should be done.


## Introduction
The number of shares an online news article gets is of great interest to news sites and advertisers alike since it serves as a good estimator of how many people an article will reach. Our data set includes various attributes that can potentially influence the number of shares, such as the number of links, pictures, and videos the article has; which day of the week the article was released, what category the article belongs to (business, entertainment, tech, etc), the number of positive or negative words, among many others. Applying the right machine learning model, we will be able to predict how popular an article will be. With accurate predictions, authors and news outlets will be able to make relevant changes to increase engagement from readers before release. 

### Previous work

Most of the work on Kaggle employed a regression algorithm to fit the number of shares (target) using the various features of the data set. We saw few that used an SVM to predict the target, and one that used the a Random Forest. To differentiate ourselves from the work that has already been done, we combined the many methods we learned this quarter (linear regression, neural nets, SVM, Random Forest) to create a new ensemble model. 

## Methods
### Data Exploration
[Link to data exploration notebook](https://github.com/WeichenZhang89/ecs-171-ecs/blob/main/Data%20Exploration.ipynb)

#### 1. Reading the CSV file
- The data was loaded from the OnlineNewsPopularity.csv file into a Pandas DataFrame
```
data = pd.read_csv('OnlineNewsPopularity.csv')
data.columns = [col.strip() for col in data.columns]
```

#### 2. Observations and Features
- Determined the total number of observations (rows) and features (columns).
```
# Total rows
print(f"Total number of observations: {data.shape[0]}\n")
# Total columns
print(f"Total number of columns: {data.shape[1]}\n")
```

#### 3.Descriptive Statistics
- Displayed the descriptive statistics for each column.
```
pd.set_option('display.max_columns', None)
data.describe()
```

#### 4.Checking Missing Values
- Checked the dataset for missing values
```
null_data_count = data.isnull().sum().sum()
print (f"There are total of {null_data_count} missing data")
```

#### 5. Scatter Plots for Features vs. Shares
- Created scatter plots for each feature against the 'shares' column.
```
fig = plt.figure(figsize=(50, 50))

# Remove all data can not be plotted
plot_columns = []
for column in columns:
    if column not in ['url', 'timedelta', 'shares']:
        plot_columns.append(column)

plt_cols = 6
plt_rows = len(plot_columns) // plt_cols + 1

fig, axes = plt.subplots(nrows=plt_rows, ncols=plt_cols, figsize=(50, 5*plt_rows))

for i, col in enumerate(plot_columns):
    ax = axes[i // plt_cols, i % plt_cols]
    ax.scatter(data[col], data['shares'])
    ax.set_xlabel(col)
    ax.set_ylabel('shares')
    ax.set_title(f'{col} vs shares')

plt.tight_layout()
plt.show()
```

#### 6. Distribution of Continuous Features
- Visualized the distribution of continuous features using histograms.
```
# List of categorical columns provided by the dataset
categorical_cols = [
    'data_channel_is_lifestyle', 'data_channel_is_entertainment',
    'data_channel_is_bus', 'data_channel_is_socmed', 'data_channel_is_tech',
    'data_channel_is_world', 'weekday_is_monday', 'weekday_is_tuesday',
    'weekday_is_wednesday', 'weekday_is_thursday', 'weekday_is_friday',
    'weekday_is_saturday', 'weekday_is_sunday', 'is_weekend', 'url',
    'timedelta'
]
extra_drop = [
    "n_unique_tokens", "n_non_stop_words","n_non_stop_unique_tokens"
]
# Remove discrete columns from being plotted
continuous_data = data.drop(columns=categorical_cols+extra_drop)

fig, axes = plt.subplots(nrows=len(continuous_data.columns) // 2 + len(continuous_data.columns) % 2, ncols=2, figsize=(15, 80))

for ax, column in zip(axes.flatten(), continuous_data.columns):
    continuous_data[column].hist(ax=ax, bins=50, grid=False)
    ax.set_title(column)
    ax.set_xlabel('Values')
    ax.set_ylabel('Frequency')
    ax.set_yscale('log')
# Hide any unused subplots
for i in range(len(continuous_data.columns), len(axes.flatten())):
    axes.flatten()[i].axis('off')

plt.tight_layout()
plt.show()
```

#### 7. Distribution of One-Hot Encoded Categorical Features
- Plotted histograms for one-hot encoded categorical features.
```
#Related one-hot encoded columns histograms

data_channel_cols = [col for col in data.columns if col.strip().startswith('data_channel_is_')]
weekday_cols = [col for col in data.columns if col.strip().startswith('weekday_is_')]

fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(15, 12))

# Plotting for data_channel columns
data[data_channel_cols].sum().plot(kind='bar', ax=axes[0], color='blue')
axes[0].set_title('Data Channel Distribution')
axes[0].set_xlabel('Data Channels')
axes[0].set_ylabel('Frequency')
axes[0].tick_params(axis='x', rotation=45)  # Tilting x axis labels
axes[0].set_yscale('log')

# Plotting for weekday columns
data[weekday_cols].sum().plot(kind='bar', ax=axes[1], color='green')
axes[1].set_title('Weekday Distribution')
axes[1].set_xlabel('Weekdays')
axes[1].set_ylabel('Frequency')
axes[1].tick_params(axis='x', rotation=45)  # Tilting x axis labels
axes[1].set_yscale('log')

plt.tight_layout()
plt.show()
```



### Data Preprocessing
[Link to preprocessing & first model notebook](https://github.com/WeichenZhang89/ecs-171-ecs/blob/main/Data%20Preprocessing%20%26%20ML%20model.ipynb)


#### Checking for null values
We run
```
null_data_count = data.isnull().sum().sum()
print (f"There are total of {null_data_count} missing data")
```
To check if there are any null values in our data and see that there are none.

#### Removing outliers
We use an isolation forest model to remove outliers with the following code:
```
IF_model = IsolationForest(contamination=0.1) # Remove 10% data
outliers = IF_model.fit_predict(data.drop(columns=['url', 'timedelta', 'shares'])) # Delete columns than don't need in IF and train IF model
data['outliers'] = outliers

# Remove outliers
data_outliers = data[data['outliers'] == -1]
new_data = data[data['outliers'] != -1].drop(columns = ['outliers'])
```

#### Data Normalization
We normalize our data using MinMaxScaler:
```
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

scaler = MinMaxScaler()
scaled_X = scaler.fit_transform(x)
```

#### Data Encoding
We change our number of shares (continuous value) to categorical variables. The following function is defined for transforming our continuous values.
```
def article_classifier(shares, top_percent):
    if shares >= top_percent[0.9]:
        return 'Great'
    elif shares >= top_percent[0.7]:
        return 'Good'
    elif shares >= top_percent[0.3]:
        return 'Normal'
    else:
        return 'Bad'
```

We then apply this function to our data with:
```
top_percent = new_data['shares'].quantile([0.3, 0.7, 0.9])

quality = []
for index, row in new_data.iterrows():
    quality.append(article_classifier(row['shares'], top_percent))

new_data['quality'] = quality
```

### Original model
#### Preprocessing and Visualizing the Input Data
We create an input matrix x by dropping specific columns from new_data, and then encodes the 'quality' column using one-hot encoding to produce a target matrix y:
```
x = new_data.drop(columns=['quality', 'url', 'timedelta', 'shares'], axis=1)
y = OneHotEncoder().fit_transform(new_data[['quality']]).toarray()
```

We scale the input matrix x using the MinMaxScaler and then splits the scaled data into training and testing sets. It subsequently prints the range of the scaled training data and displays a histogram for each column of the scaled training data:
```
scaler = MinMaxScaler()
scaled_X = scaler.fit_transform(x)

xTrain, xTest, yTrain, yTest = train_test_split(scaled_X, y, test_size=0.2, random_state=10)

print("range of the xTrain:")
print(xTrain.max() - xTrain.min())

scale_xTrain = pd.DataFrame(xTrain)
scale_xTest = pd.DataFrame(xTest)
for i in scale_xTrain.columns:
    plt.hist(xTrain[i])
plt.title("Histogram of scale_xTrain")
plt.show()
```

#### Setting Up the Nerual Networks and Random Forest Components
We build 2 neural network models with  'MirroredStrategy' for potential distributed training across multiple GPUs:
```
def build_nn_model(input_shape, num_classes):
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(128, activation='relu', input_shape=input_shape),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model

mirror_strategy = tf.distribute.MirroredStrategy()
with mirror_strategy.scope():
    nn_model1 = build_nn_model(scale_xTrain.shape[1:], yTrain.shape[1])
    nn_model2 = build_nn_model(scale_xTrain.shape[1:], yTrain.shape[1])

    fit_speed = tf.data.Dataset.from_tensor_slices((scale_xTrain, yTrain))
    fit_speed = fit_speed.repeat().batch(100)

    nn_model1.fit(scale_xTrain, yTrain, epochs=10)
    nn_model2.fit(scale_xTrain, yTrain, epochs=10)
```

We initialize a Random Forest classifier with 100 trees and then trains it on the scaled training data, converting the one-hot encoded target labels back to integer labels using argmax:
```
RF_model = RandomForestClassifier(n_estimators=100)
RF_model.fit(scale_xTrain, yTrain.argmax(axis=1))
```

We then run the predictions of class probabilities for both the training and test datasets using three models (two neural networks and a Random Forest), then convert these probabilities to class labels, and print out the shapes of the predictions and the predicted labels.
```
nn1_train_pred = nn_model1.predict(scale_xTrain)
nn2_train_pred = nn_model2.predict(scale_xTrain)
RF_train_pred = RF_model.predict_proba(scale_xTrain)

nn1_test_pred = nn_model1.predict(scale_xTest)
nn2_test_pred = nn_model2.predict(scale_xTest)
RF_test_pred = RF_model.predict_proba(scale_xTest)

print("Shape:")
print("nn1_train_pred:", nn1_train_pred.shape)
print("nn2_train_pred:", nn2_train_pred.shape)
print("RF_train_pred:", RF_train_pred.shape)

nn1_train_labels = nn1_train_pred.argmax(axis=1)
nn2_train_labels = nn2_train_pred.argmax(axis=1)
RF_train_labels = RF_train_pred.argmax(axis=1)

nn1_test_labels = nn1_test_pred.argmax(axis=1)
nn2_test_labels = nn2_test_pred.argmax(axis=1)
RF_test_labels = RF_test_pred.argmax(axis=1)

print("Labels:")
print("nn1_train_pred:", nn1_train_labels)
print("nn2_train_pred:", nn2_train_labels)
print("RF_train_pred:", RF_train_labels)
print("nn1_test_pred:", nn1_test_labels)
print("nn2_test_pred:", nn2_test_labels)
print("RF_test_pred:", RF_test_labels)
print()
```

#### Ensemble Models
To optimize performance, we constructed three separate ensemble models with different complexities.

##### The Simple Ensemble Model
For the simple ensemble model, We initialize three Ridge regression models with different regularization strengths (alpha values of 0.01, 0.3, and 0.5), then fit each of these models to the scaled training data, and construct the ensemble model.
```
linmodel_1 = Ridge(alpha=0.01)
linmodel_2 = Ridge(alpha=0.3)
linmodel_3 = Ridge(alpha=0.5)

linmodel_1.fit(scale_xTrain, yTrain)
linmodel_2.fit(scale_xTrain, yTrain)
linmodel_3.fit(scale_xTrain, yTrain)

linmodel1_train_pred = linmodel_1.predict(scale_xTrain)
linmodel2_train_pred = linmodel_2.predict(scale_xTrain)
linmodel3_train_pred = linmodel_3.predict(scale_xTrain)

linmodel1_test_pred = linmodel_1.predict(scale_xTest)
linmodel2_test_pred = linmodel_2.predict(scale_xTest)
linmodel3_test_pred = linmodel_3.predict(scale_xTest)

ensemble_simple_train_pred = (linmodel1_train_pred + linmodel2_train_pred + linmodel3_train_pred) / 3
ensemble_simple_test_pred = (linmodel1_test_pred + linmodel2_test_pred + linmodel3_test_pred) / 3

ensemble_simple_train_labels = ensemble_simple_train_pred.argmax(axis=1)
ensemble_simple_test_labels = ensemble_simple_test_pred.argmax(axis=1)
```

We then do the evaluation with classification report and confusion matrix for it:
```
print("Train:")
print(classification_report(yTrain.argmax(axis=1), ensemble_simple_train_labels, zero_division=0))
print("Test:")
print(classification_report(yTest.argmax(axis=1), ensemble_simple_test_labels, zero_division=0))

yTrain_labels = yTrain.argmax(axis=1)
cf1 = confusion_matrix(yTrain_labels, ensemble_simple_train_labels)
sns.heatmap(cf1, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix for Training Set')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

yTest_labels = yTest.argmax(axis=1)
cf1 = confusion_matrix(yTest_labels, ensemble_simple_test_labels)
sns.heatmap(cf1, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix for Testing Set')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()
```

And we calculate the accuracy for each component to check for overfitting:
```
lin1_train_labels = linmodel1_train_pred.argmax(axis=1)
lin2_train_labels = linmodel2_train_pred.argmax(axis=1)
lin3_train_labels = linmodel3_train_pred.argmax(axis=1)

lin1_test_labels = linmodel1_test_pred.argmax(axis=1)
lin2_test_labels = linmodel2_test_pred.argmax(axis=1)
lin3_test_labels = linmodel3_test_pred.argmax(axis=1)

lin1_train_accuracy = accuracy_score(yTrain.argmax(axis=1), lin1_train_labels)
lin2_train_accuracy = accuracy_score(yTrain.argmax(axis=1), lin2_train_labels)
lin3_train_accuracy = accuracy_score(yTrain.argmax(axis=1), lin3_train_labels)

lin1_test_accuracy = accuracy_score(yTest.argmax(axis=1), lin1_test_labels)
lin2_test_accuracy = accuracy_score(yTest.argmax(axis=1), lin2_test_labels)
lin3_test_accuracy = accuracy_score(yTest.argmax(axis=1), lin3_test_labels)

ensemble_simple_train_accuracy = accuracy_score(yTrain.argmax(axis=1), ensemble_simple_train_labels)
ensemble_simple_test_accuracy = accuracy_score(yTest.argmax(axis=1), ensemble_simple_test_labels)

print("Model Train Accuracies:")
print(f"LR 1: {lin1_train_accuracy:.4f}")
print(f"LR 2: {lin2_train_accuracy:.4f}")
print(f"LR 3: {lin3_train_accuracy:.4f}")
print()

print("Model Test Accuracies:")
print(f"LR 1: {lin1_test_accuracy:.4f}")
print(f"LR 2: {lin2_test_accuracy:.4f}")
print(f"LR 3: {lin3_test_accuracy:.4f}")
print()

print("Ensemble Model:")
print(f"Training Accuracy: {ensemble_simple_train_accuracy:.4f}")
print(f"Testing Accuracy: {ensemble_simple_test_accuracy:.4f}")

if ensemble_simple_test_accuracy > ensemble_simple_train_accuracy:
    print("There could be overfitting.")
else:
    print("There should be no overfitting.")
```

##### The Medium Ensemble Model
For the medium ensemble model, we use 2 constructed nerual networks and 1 Ridge model to construct it:
```
ridge_train_prob = np.exp(linmodel1_train_pred) / np.sum(np.exp(linmodel1_train_pred), axis=1, keepdims=True)
ridge_test_prob = np.exp(linmodel1_test_pred) / np.sum(np.exp(linmodel1_test_pred), axis=1, keepdims=True)

ensemble_medium_train_pred = (nn1_train_pred + nn2_train_pred + linmodel1_train_pred) / 3
ensemble_medium_test_pred = (nn1_test_pred + nn2_test_pred + linmodel1_test_pred) / 3

ensemble_medium_train_labels = ensemble_medium_train_pred.argmax(axis=1)
ensemble_medium_test_labels = ensemble_medium_test_pred.argmax(axis=1)
```

Similarly, we do the evaluation for it and check for overfitting:
```
print("Train:")
print(classification_report(yTrain.argmax(axis=1), ensemble_medium_train_labels, zero_division=0))
print("Test:")
print(classification_report(yTest.argmax(axis=1), ensemble_medium_test_labels, zero_division=0))

yTrain_labels = yTrain.argmax(axis=1)
ensemble_medium_train_labels = ensemble_simple_train_pred.argmax(axis=1)
cf1 = confusion_matrix(yTrain_labels, ensemble_medium_train_labels)
sns.heatmap(cf1, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix for Training Set')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

yTest_labels = yTest.argmax(axis=1)
ensemble_medium_test_labels = ensemble_simple_test_pred.argmax(axis=1)
cf1 = confusion_matrix(yTest_labels, ensemble_medium_test_labels)
sns.heatmap(cf1, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix for Testing Set')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

lin1_train_labels = linmodel1_train_pred.argmax(axis=1)
lin2_train_labels = linmodel2_train_pred.argmax(axis=1)

lin1_test_labels = linmodel1_test_pred.argmax(axis=1)
lin2_test_labels = linmodel2_test_pred.argmax(axis=1)

lin1_train_accuracy = accuracy_score(yTrain.argmax(axis=1), lin1_train_labels)
lin2_train_accuracy = accuracy_score(yTrain.argmax(axis=1), lin2_train_labels)
nn1_train_accuracy = accuracy_score(yTrain.argmax(axis=1), nn1_train_labels)

lin1_test_accuracy = accuracy_score(yTest.argmax(axis=1), lin1_test_labels)
lin2_test_accuracy = accuracy_score(yTest.argmax(axis=1), lin2_test_labels)
nn1_test_accuracy = accuracy_score(yTest.argmax(axis=1), nn1_test_labels)

ensemble_medium_train_accuracy = accuracy_score(yTrain.argmax(axis=1), ensemble_medium_train_labels)
ensemble_medium_test_accuracy = accuracy_score(yTest.argmax(axis=1), ensemble_medium_test_labels)

print("Model Train Accuracies:")
print(f"LR 1: {lin1_train_accuracy:.4f}")
print(f"LR 2: {lin2_train_accuracy:.4f}")
print(f"NN : {nn1_train_accuracy:.4f}")
print()

print("Model Test Accuracies:")
print(f"LR 1: {lin1_test_accuracy:.4f}")
print(f"LR 2: {lin2_test_accuracy:.4f}")
print(f"NN : {nn1_test_accuracy:.4f}")
print()

print("Ensemble Model:")
print(f"Training Accuracy: {ensemble_medium_train_accuracy:.4f}")
print(f"Testing Accuracy: {ensemble_medium_test_accuracy:.4f}")

if ensemble_simple_test_accuracy > ensemble_simple_train_accuracy:
    print("There could be overfitting.")
else:
    print("There should be no overfitting.")
```

##### The Complex Ensemble Model
For the complex ensemble model, we apply 2 nerual networks with 1 random forest and do the evaluation:
```
ensemble_train_pred = (nn1_train_pred + nn2_train_pred + RF_train_pred) / 3
ensemble_test_pred = (nn1_test_pred + nn2_test_pred + RF_test_pred) / 3

ensemble_train_labels = ensemble_train_pred.argmax(axis=1)
ensemble_test_labels = ensemble_test_pred.argmax(axis=1)

print("ensemble_train_labels:",ensemble_train_labels)
print("ensemble_train_labels:",ensemble_train_labels)

print("Train:")
print(classification_report(yTrain.argmax(axis=1), ensemble_train_labels))
print("Test:")
print(classification_report(yTest.argmax(axis=1), ensemble_test_labels))

yTrain_labels = yTrain.argmax(axis=1)
cf1 = confusion_matrix(yTrain_labels, ensemble_train_labels)
sns.heatmap(cf1, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix for Training Set')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

yTest_labels = yTest.argmax(axis=1)
cf1 = confusion_matrix(yTest_labels, ensemble_test_labels)
sns.heatmap(cf1, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix for Testing Set')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

nn1_train_accuracy = accuracy_score(yTrain.argmax(axis=1), nn1_train_labels)
nn2_train_accuracy = accuracy_score(yTrain.argmax(axis=1), nn2_train_labels)
RF_train_accuracy = accuracy_score(yTrain.argmax(axis=1), RF_train_labels)

nn1_test_accuracy = accuracy_score(yTest.argmax(axis=1), nn1_test_labels)
nn2_test_accuracy = accuracy_score(yTest.argmax(axis=1), nn2_test_labels)
RF_test_accuracy = accuracy_score(yTest.argmax(axis=1), RF_test_labels)

ensemble_train_accuracy = accuracy_score(yTrain.argmax(axis=1), ensemble_train_labels)
ensemble_test_accuracy = accuracy_score(yTest.argmax(axis=1), ensemble_test_labels)

print("Model Train Accuracies:")
print(f"Neural Network 1: {nn1_train_accuracy:.4f}")
print(f"Neural Network 2: {nn2_train_accuracy:.4f}")
print(f"Random Forest: {RF_train_accuracy:.4f}")
print()

print("Model Test Accuracies:")
print(f"Neural Network 1: {nn1_test_accuracy:.4f}")
print(f"Neural Network 2: {nn2_test_accuracy:.4f}")
print(f"Random Forest: {RF_test_accuracy:.4f}")
print()

print("Ensemble Model:")
print(f"Training Accuracy: {ensemble_train_accuracy:.4f}")
print(f"Testing Accuracy: {ensemble_test_accuracy:.4f}")

if ensemble_test_accuracy > ensemble_train_accuracy:
    print("There could be overfitting.")
else:
    print("There should be no overfitting.")
```

#### Applying Stratified KFold
Then we apply Stratified KFold to each ensemble model:
```
# Initialize empty lists to store trained models
nn1_models, nn2_models, rf_models, ridge1_models, ridge2_models, ridge3_models = [], [], [], [], [], []

# Define StratifiedKFold instance
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=10)

mirror_strategy = tf.distribute.MirroredStrategy()

for train_index, test_index in skf.split(x, y.argmax(axis=1)):
    xTrain_fold, xTest_fold = x.iloc[train_index].values, x.iloc[test_index].values
    yTrain_fold, yTest_fold = y[train_index], y[test_index]

    # Scale data after splitting
    scaler = MinMaxScaler()
    xTrain_fold = scaler.fit_transform(xTrain_fold)
    xTest_fold = scaler.transform(xTest_fold)  # Important: transform only using the train data scaling parameters

    # Neural Network Training within the mirror_strategy scope
    with mirror_strategy.scope():
        nn_model1 = build_nn_model(xTrain_fold.shape[1:], yTrain_fold.shape[1])
        nn_model2 = build_nn_model(xTrain_fold.shape[1:], yTrain_fold.shape[1])

        nn_model1.fit(xTrain_fold, yTrain_fold, epochs=10, verbose=0)
        nn_model2.fit(xTrain_fold, yTrain_fold, epochs=10, verbose=0)

    # Random Forest Training
    RF_model = RandomForestClassifier(n_estimators=100)
    RF_model.fit(xTrain_fold, yTrain_fold.argmax(axis=1))

    # Ridge Regression Training
    linmodel_1 = Ridge(alpha=0.01)
    linmodel_2 = Ridge(alpha=0.3)
    linmodel_3 = Ridge(alpha=0.5)
    linmodel_1.fit(xTrain_fold, yTrain_fold)
    linmodel_2.fit(xTrain_fold, yTrain_fold)
    linmodel_3.fit(xTrain_fold, yTrain_fold)

    # Store trained models
    nn1_models.append(nn_model1)
    nn2_models.append(nn_model2)
    rf_models.append(RF_model)
    ridge1_models.append(linmodel_1)
    ridge2_models.append(linmodel_2)
    ridge3_models.append(linmodel_3)
```

#### Evaluating the Models with Stratified KFold
We then do the evaluations for them again:
```
def complex_ensemble_accuracy(nn1_models, nn2_models, rf_models):

    train_accuracies = []
    test_accuracies = []

    for i in range(5):
        # Train metrics
        nn1_train_pred = nn1_models[i].predict(scale_xTrain)
        nn2_train_pred = nn2_models[i].predict(scale_xTrain)
        RF_train_pred = rf_models[i].predict_proba(scale_xTrain)

        ensemble_train_pred = (nn1_train_pred + nn2_train_pred + RF_train_pred) / 3
        ensemble_train_labels = ensemble_train_pred.argmax(axis=1)

        train_acc = accuracy_score(yTrain.argmax(axis=1), ensemble_train_labels)
        train_accuracies.append(train_acc)

        # Test metrics
        nn1_test_pred = nn1_models[i].predict(scale_xTest)
        nn2_test_pred = nn2_models[i].predict(scale_xTest)
        RF_test_pred = rf_models[i].predict_proba(scale_xTest)

        ensemble_test_pred = (nn1_test_pred + nn2_test_pred + RF_test_pred) / 3
        ensemble_test_labels = ensemble_test_pred.argmax(axis=1)

        test_acc = accuracy_score(yTest.argmax(axis=1), ensemble_test_labels)
        test_accuracies.append(test_acc)

    return train_accuracies, test_accuracies

# Compute average train and test accuracies for the complex ensemble
complex_ensemble_train_acc, complex_ensemble_test_acc = complex_ensemble_accuracy(nn1_models, nn2_models, rf_models)

#Avearage of train and test accuracies
avg_train_acc = np.mean(complex_ensemble_train_acc)
avg_test_acc = np.mean(complex_ensemble_test_acc)

print("Complex Ensemble Model:")
print(f"Training Accuracy: {avg_train_acc:.4f}")
print(f"Testing Accuracy: {avg_test_acc:.4f}")

#Plot train against test acc across the folds
plt.plot(complex_ensemble_train_acc, label='Training Accuracy')
plt.plot(complex_ensemble_test_acc, label='Test Accuracy')
plt.title('Training and Test Accuracy for Complex Ensemble with 2NN + 1RF')
plt.xlabel('Folds')
plt.xticks(list(range(5)), [f"{i+1}" for i in range(5)])
plt.ylabel('Accuracy')
plt.legend()
plt.show()
```

```
def medium_ensemble_accuracy(nn1_models, nn2_models, ridge_models):
    train_accuracies = []
    test_accuracies = []

    for i in range(5):
        # Train metrics
        nn1_train_pred = nn1_models[i].predict(scale_xTrain)
        nn2_train_pred = nn2_models[i].predict(scale_xTrain)
        ridge_train_pred = ridge_models[i].predict(scale_xTrain)

        # Convert ridge predictions to probabilities using softmax
        ridge_train_prob = np.exp(ridge_train_pred) / np.sum(np.exp(ridge_train_pred), axis=1, keepdims=True)

        ensemble_train_pred = (nn1_train_pred + nn2_train_pred + ridge_train_prob) / 3
        ensemble_train_labels = ensemble_train_pred.argmax(axis=1)

        train_acc = accuracy_score(yTrain.argmax(axis=1), ensemble_train_labels)
        train_accuracies.append(train_acc)

        # Test metrics
        nn1_test_pred = nn1_models[i].predict(scale_xTest)
        nn2_test_pred = nn2_models[i].predict(scale_xTest)
        ridge_test_pred = ridge_models[i].predict(scale_xTest)

        # Convert ridge predictions to probabilities using softmax
        ridge_test_prob = np.exp(ridge_test_pred) / np.sum(np.exp(ridge_test_pred), axis=1, keepdims=True)

        ensemble_test_pred = (nn1_test_pred + nn2_test_pred + ridge_test_prob) / 3
        ensemble_test_labels = ensemble_test_pred.argmax(axis=1)

        test_acc = accuracy_score(yTest.argmax(axis=1), ensemble_test_labels)
        test_accuracies.append(test_acc)

    return train_accuracies, test_accuracies

# Compute average train and test accuracies for the medium ensemble with Ridge 1 (alpha = 0.01)
medium_ensemble_train_acc_1, medium_ensemble_test_acc_1 = medium_ensemble_accuracy(nn1_models, nn2_models, ridge1_models)
# Average of train and test accuracies
avg_train_acc = np.mean(medium_ensemble_train_acc_1)
avg_test_acc = np.mean(medium_ensemble_test_acc_1)

print("Medium Ensemble Model with Ridge1:")
print(f"Training Accuracy: {avg_train_acc:.4f}")
print(f"Testing Accuracy: {avg_test_acc:.4f}")
print("\n")

plt.plot(medium_ensemble_train_acc_1, label='Training Accuracy')
plt.plot(medium_ensemble_test_acc_1, label='Test Accuracy')
plt.title('Training and Test Accuracy for Medium Ensemble with 2NN + Ridge 1')
plt.xlabel('Folds')
plt.xticks(list(range(5)), [f"{i+1}" for i in range(5)])
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Compute average train and test accuracies for the medium ensemble with Ridge 1 (alpha = 0.3)
medium_ensemble_train_acc_2, medium_ensemble_test_acc_2 = medium_ensemble_accuracy(nn1_models, nn2_models, ridge2_models)
# Average of train and test accuracies
avg_train_acc2 = np.mean(medium_ensemble_train_acc_2)
avg_test_acc2 = np.mean(medium_ensemble_test_acc_2)

print("Medium Ensemble Model with Ridge2:")
print(f"Training Accuracy: {avg_train_acc2:.4f}")
print(f"Testing Accuracy: {avg_test_acc2:.4f}")

plt.plot(medium_ensemble_train_acc_2, label='Training Accuracy')
plt.plot(medium_ensemble_test_acc_2, label='Test Accuracy')
plt.title('Training and Test Accuracy for Medium Ensemble with 2NN +Ridge 2')
plt.xlabel('Folds')
plt.xticks(list(range(5)), [f"{i+1}" for i in range(5)])
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Compute average train and test accuracies for the medium ensemble with Ridge 1 (alpha = 0.5)
medium_ensemble_train_acc_3, medium_ensemble_test_acc_3 = medium_ensemble_accuracy(nn1_models, nn2_models, ridge3_models)
# Average of train and test accuracies
avg_train_acc3 = np.mean(medium_ensemble_train_acc_3)
avg_test_acc3 = np.mean(medium_ensemble_test_acc_3)
print("Medium Ensemble Model with Ridge3:")
print(f"Training Accuracy: {avg_train_acc3:.4f}")
print(f"Testing Accuracy: {avg_test_acc3:.4f}")

plt.plot(medium_ensemble_train_acc_3, label='Training Accuracy')
plt.plot(medium_ensemble_test_acc_3, label='Test Accuracy')
plt.title('Training and Test Accuracy for Medium Ensemble with 2NN + Ridge 3')
plt.xlabel('Folds')
plt.xticks(list(range(5)), [f"{i+1}" for i in range(5)])
plt.ylabel('Accuracy')
plt.legend()
plt.show()
```

```
def easy_ensemble_accuracy(ridge1_models, ridge2_models, ridge3_models):

    train_accuracies = []
    test_accuracies = []

    for i in range(5):
        # Train metrics
        ridge1_train_pred = ridge1_models[i].predict(scale_xTrain)
        ridge2_train_pred = ridge2_models[i].predict(scale_xTrain)
        ridge3_train_pred = ridge3_models[i].predict(scale_xTrain)

        ensemble_train_pred = (ridge1_train_pred + ridge2_train_pred + ridge3_train_pred) / 3
        ensemble_train_labels = ensemble_train_pred.argmax(axis=1)

        train_acc = accuracy_score(yTrain.argmax(axis=1), ensemble_train_labels)
        train_accuracies.append(train_acc)

        # Test metrics
        ridge1_test_pred = ridge1_models[i].predict(scale_xTest)
        ridge2_test_pred = ridge2_models[i].predict(scale_xTest)
        ridge3_test_pred = ridge3_models[i].predict(scale_xTest)

        ensemble_test_pred = (ridge1_test_pred + ridge2_test_pred + ridge3_test_pred) / 3
        ensemble_test_labels = ensemble_test_pred.argmax(axis=1)

        test_acc = accuracy_score(yTest.argmax(axis=1), ensemble_test_labels)
        test_accuracies.append(test_acc)



    return train_accuracies, test_accuracies

# Compute average train and test accuracies for the easy ensemble
easy_ensemble_train_acc, easy_ensemble_test_acc = easy_ensemble_accuracy(ridge1_models, ridge2_models, ridge3_models)

# Average of train and test accuracies
avg_train_acc = np.mean(easy_ensemble_train_acc)
avg_test_acc = np.mean(easy_ensemble_test_acc)

print("Easy Ensemble Model:")
print(f"Training Accuracy: {avg_train_acc:.4f}")
print(f"Testing Accuracy: {avg_test_acc:.4f}")

# Plot train acc against test acc
plt.plot(easy_ensemble_train_acc, label='Training Accuracy')
plt.plot(easy_ensemble_test_acc, label='Test Accuracy')
plt.title('Training and Test Accuracy for Easy Ensemble')
plt.xlabel('Folds')
plt.xticks(list(range(5)), [f"{i+1}" for i in range(5)])
plt.ylabel('Accuracy')
plt.legend()
plt.show()
```


### Improved model

### set X and y
#### select all columns except 'quality', 'url', 'timedelta', 'shares'
#### target column is 'quality'
```
x = new_data.drop(columns=['quality', 'url', 'timedelta', 'shares'], axis=1)
y = new_data['quality']

```
#### apply label encoding to y
```
le = LabelEncoder()
y = le.fit_transform(y)

```
#### scale the x, and then split train/test for x/y
```
scaler = MinMaxScaler()
scaled_X = scaler.fit_transform(x)
xTrain, xTest, yTrain, yTest = train_test_split(scaled_X, y, test_size=0.2, random_state=42)

```

#### convert data type of xTrain, xTest to dataframe
```
scale_xTrain = pd.DataFrame(xTrain)
scale_xTest = pd.DataFrame(xTest)

```

### Build the NN1 model
#### build the model with 4 layers, input dimension correspond to scale_xTrain, 2 types of activation functions used are 'relu' and 'softmax'
```
nn1 = Sequential([
    Dense(512, activation='relu', input_dim=scale_xTrain.shape[1]),
    Dense(128, activation='relu'),
    Dense(64, activation='relu'),
    Dense(4, activation='softmax'),
])


```

#### compile and train the model

```
nn1.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

history = nn1.fit(scale_xTrain, yTrain, epochs=100, batch_size=32)

```
### train accuracy graph
#### extract 'accuracy' value and use that as y value
#### set the name for x-axis = 'Epochs', y-axis = 'Accuracy'

### train loss graph
#### extract 'loss' value and use that as y value
#### set the name for x-axis = 'Epochs', y-axis = 'Loss'

```
nn1.save('nn1.h5')

plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()


```

### Evaluate NN1 model
#### load the model we saved in the previous step
#### do the prediction of y based on the scale_xTest
#### print the classification report
```
import seaborn as sns
nn1 = load_model('nn1.h5')
y_predict = np.argmax(nn1.predict(scale_xTest), axis=1)

print(classification_report(yTest, y_predict))

```

#### build the confusion matrix and display that as the heatmap form
#### number are shown on the heatmap
#### xlabel = 'Predicted', ylabel = 'True'
```
cf = confusion_matrix(yTest, y_predict)
sns.heatmap(cf, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

```

#### print the testing and training accuracy
```
nn1_test_accuracy = accuracy_score(yTest, y_predict)
print(f"Test Accuracy: {nn1_test_accuracy * 100:.5f}%")

y_predict = np.argmax(nn1.predict(scale_xTrain), axis=1)
nn1_train_accuracy = accuracy_score(yTrain, y_predict)
print(f"Train Accuracy: {nn1_test_accuracy * 100:.5f}%")

```


### Build the NN2 model
#### build the model with 5 layers, input dimension correspond to scale_xTrain, 2 types of activation functions used are 'tanh' and 'softmax'
```
nn2 = Sequential([
    Dense(512, activation='tanh', input_dim=scale_xTrain.shape[1]),
    Dense(256, activation='tanh'),
    Dense(128, activation='tanh'),
    Dense(64, activation='tanh'),
    Dense(4, activation='softmax'),
])

```
#### compile and train the model
```
nn2.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

history = nn2.fit(scale_xTrain, yTrain, epochs=100, batch_size=32)
nn2.save('nn2.h5')


```
### train accuracy graph
#### extract 'accuracy' value and use that as y value
#### set the name for x-axis = 'Epochs', y-axis = 'Accuracy'

### train loss graph
#### extract 'loss' value and use that as y value
#### set the name for x-axis = 'Epochs', y-axis = 'Loss'


```
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
```


### Evaluate NN2 model
#### load the model we saved in the previous step
#### do the prediction of y based on the scale_xTest
#### print the classification report

```
nn2 = load_model('nn2.h5')
y_predict = np.argmax(nn2.predict(scale_xTest), axis=1)

print(classification_report(yTest, y_predict))


```

#### build the confusion matrix and display that as the heatmap form
#### number are shown on the heatmap
#### xlabel = 'Predicted', ylabel = 'True'


```
cf = confusion_matrix(yTest, y_predict)
sns.heatmap(cf, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()
```

#### print the testing and training accuracy
```
nn2_test_accuracy = accuracy_score(yTest, y_predict)
print(f"Test Accuracy: {nn2_test_accuracy * 100:.5f}%")

y_predict = np.argmax(nn2.predict(scale_xTrain), axis=1)
nn2_train_accuracy = accuracy_score(yTrain, y_predict)
print(f"Train Accuracy: {nn2_train_accuracy * 100:.5f}%")


```


### build the Random Forest Model
#### build the Random Forest model and train the model
```
RF_model = RandomForestClassifier(n_estimators=1000)
RF_model.fit(scale_xTrain, yTrain)
dump(RF_model, 'RF.joblib')

```

### Evaluate the Random Forest Model
#### do the prediction based on the model we train
```
RF_model = load('RF.joblib')
y_pred = RF_model.predict(scale_xTest)

```

#### print the classification report
```
print("Classification Report:")
print(classification_report(yTest, y_pred))

```

#### print the test accuracy
```
rf_test_accuracy = accuracy_score(yTest, y_pred)
print(f"Test Accuracy: {rf_test_accuracy * 100:.5f}%")
```

#### print the train accuracy
```
y_pred = RF_model.predict(scale_xTrain)
rf_train_accuracy = accuracy_score(yTrain, y_pred)
print(f"Train Accuracy: {rf_train_accuracy * 100:.5f}%")
```


### Ensemble Model
#### Get prediction base on NN1, NN2, Random Forest Model #### that we build
```
from sklearn.metrics import accuracy_score, classification_report
import numpy as np

# Get predictions for each model on the training and test set
nn1_preds_train = nn1.predict(scale_xTrain)
nn2_preds_train = nn2.predict(scale_xTrain)
rf_preds_train = RF_model.predict_proba(scale_xTrain)

nn1_preds_test = nn1.predict(scale_xTest)
nn2_preds_test = nn2.predict(scale_xTest)
rf_preds_test = RF_model.predict_proba(scale_xTest)

```

#### use the classification report and turn on the output_dict option,
#### so we can access the precision score for 4 times due to the article_classifier
#### have four types.

```
# Classification report on test set to get the precisions for weighting
report_nn1_test = classification_report(yTest, np.argmax(nn1.predict(scale_xTest), axis=1), output_dict=True)
report_nn2_test = classification_report(yTest, np.argmax(nn2.predict(scale_xTest), axis=1), output_dict=True)
report_rf_test = classification_report(yTest, RF_model.predict(scale_xTest), output_dict=True)

# Get precisions for each class on the test set
precisions_nn1_test = np.array([report_nn1_test[str(i)]['precision'] for i in range(4)])
precisions_nn2_test = np.array([report_nn2_test[str(i)]['precision'] for i in range(4)])
precisions_rf_test = np.array([report_rf_test[str(i)]['precision'] for i in range(4)])


```
#### weight the prediction, reshape in order to make the 
#### multiplication success

```
# Use test set precisions to weight each model's predictions on both the training and test sets
nn1_preds_train *= precisions_nn1_test.reshape(1, -1)
nn2_preds_train *= precisions_nn2_test.reshape(1, -1)
rf_preds_train *= precisions_rf_test.reshape(1, -1)

nn1_preds_test *= precisions_nn1_test.reshape(1, -1)
nn2_preds_test *= precisions_nn2_test.reshape(1, -1)
rf_preds_test *= precisions_rf_test.reshape(1, -1)


```


#### average the prediction result, and find the most confirmed 
#### probablity on each row

```
# Compute the ensemble predictions for both the training and test sets
ensemble_preds_train = (nn1_preds_train + nn2_preds_train + rf_preds_train) / 3
ensemble_preds_test = (nn1_preds_test + nn2_preds_test + rf_preds_test) / 3

# Get results for both the training and test sets
ensemble_preds_train = np.argmax(ensemble_preds_train, axis=1)
ensemble_preds_test = np.argmax(ensemble_preds_test, axis=1)
```

#### find the training and testing accuracy

```
# Calculate the ensemble model accuracy on both the training and testing sets
ensemble_train_accuracy = accuracy_score(yTrain, ensemble_preds_train)
ensemble_test_accuracy = accuracy_score(yTest, ensemble_preds_test)

# Print the ensemble model accuracies
print(f"Ensemble training accuracy: {ensemble_train_accuracy:.5f}")
print(f"Ensemble test accuracy: {ensemble_test_accuracy:.5f}")
```

#### print the classification report for the emsemble model training and testing set

```
print("Classification Report for Training Set:")
print(classification_report(yTrain, ensemble_preds_train))

print("Classification Report for Test Set:")
print(classification_report(yTest, ensemble_preds_test))
```

#### construct the confusion matrix base on the emsemble model training and testing set

```
confusion_matrix_train = confusion_matrix(yTrain, ensemble_preds_train)
confusion_matrix_test = confusion_matrix(yTest, ensemble_preds_test)
```


#### Base on the confusion matrix, we plot the figure of heatmap
```
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
sns.heatmap(confusion_matrix_train, annot=True, fmt="d", cmap='Blues')
plt.title('Confusion Matrix for Training Set')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')

plt.subplot(1, 2, 2)
sns.heatmap(confusion_matrix_test, annot=True, fmt="d", cmap='Blues')
plt.title('Confusion Matrix for Test Set')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')

plt.tight_layout()
plt.show()
```

## Results
TODO: Summarize what happened to our data after we normalized it (put the result of .describe() here).

TODO: Accuracy & Overfitting graph for model 1.

TODO: Accuracy & Overfitting graph for model 2.

Note: Just reporting the results will be good enough. I think the analysis is supposed to be done in the Discussion section.


## Discussion
### Discussion for FIRST MODEL
TODO: Repeat the Methods & Results section, but explain the reasoning behind each section. 

TODO: Explore results of our first model, do overfitting analysis

### Discussion for SECOND MODEL

TODO: Explain how second model differs.

TODO: Explore second model results, do overfitting analysis.

TODO: Summarize final results of model 2 (accuracy, etc).


Note: The following was preserved from our original milestone submissions. I think we only have to tweak & add on what we have for the discussion section.

### Data Exploration
#### Descriptive Statistics
- The descriptive statistics provided a summary of the central tendency, dispersion, and shape of the distribution of the dataset. It's a very useful method for a quick statistical summary, however it doesn't provide a full picture of data distribution. For a more comprehensive understanding, please refer to the histograms below.

<details>
  <summary>Click to expand the `.describe()` table</summary>

|       |   timedelta |   n_tokens_title |   n_tokens_content |   n_unique_tokens |   n_non_stop_words |   n_non_stop_unique_tokens |   num_hrefs |   num_self_hrefs |    num_imgs |   num_videos |   average_token_length |   num_keywords |   data_channel_is_lifestyle |   data_channel_is_entertainment |   data_channel_is_bus |   data_channel_is_socmed |   data_channel_is_tech |   data_channel_is_world |   kw_min_min |   kw_max_min |   kw_avg_min |   kw_min_max |   kw_max_max |   kw_avg_max |   kw_min_avg |   kw_max_avg |   kw_avg_avg |   self_reference_min_shares |   self_reference_max_shares |   self_reference_avg_sharess |   weekday_is_monday |   weekday_is_tuesday |   weekday_is_wednesday |   weekday_is_thursday |   weekday_is_friday |   weekday_is_saturday |   weekday_is_sunday |   is_weekend |        LDA_00 |        LDA_01 |        LDA_02 |        LDA_03 |        LDA_04 |   global_subjectivity |   global_sentiment_polarity |   global_rate_positive_words |   global_rate_negative_words |   rate_positive_words |   rate_negative_words |   avg_positive_polarity |   min_positive_polarity |   max_positive_polarity |   avg_negative_polarity |   min_negative_polarity |   max_negative_polarity |   title_subjectivity |   title_sentiment_polarity |   abs_title_subjectivity |   abs_title_sentiment_polarity |    shares |
|:------|------------:|-----------------:|-------------------:|------------------:|-------------------:|---------------------------:|------------:|-----------------:|------------:|-------------:|-----------------------:|---------------:|----------------------------:|--------------------------------:|----------------------:|-------------------------:|-----------------------:|------------------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|----------------------------:|----------------------------:|-----------------------------:|--------------------:|---------------------:|-----------------------:|----------------------:|--------------------:|----------------------:|--------------------:|-------------:|--------------:|--------------:|--------------:|--------------:|--------------:|----------------------:|----------------------------:|-----------------------------:|-----------------------------:|----------------------:|----------------------:|------------------------:|------------------------:|------------------------:|------------------------:|------------------------:|------------------------:|---------------------:|---------------------------:|-------------------------:|-------------------------------:|----------:|
| count |   39644     |      39644       |          39644     |      39644        |       39644        |               39644        |  39644      |      39644       | 39644       |  39644       |           39644        |    39644       |               39644         |                    39644        |          39644        |            39644         |           39644        |            39644        |   39644      |     39644    |    39644     |      39644   |        39644 |        39644 |     39644    |     39644    |     39644    |                    39644    |                     39644   |                    39644     |        39644        |         39644        |           39644        |          39644        |        39644        |         39644         |       39644         | 39644        | 39644         | 39644         | 39644         | 39644         | 39644         |          39644        |               39644         |                39644         |               39644          |          39644        |          39644        |            39644        |           39644         |            39644        |            39644        |            39644        |            39644        |         39644        |              39644         |             39644        |                   39644        |  39644    |
| mean  |     354.53  |         10.3987  |            546.515 |          0.548216 |           0.996469 |                   0.689175 |     10.8837 |          3.29364 |     4.54414 |      1.24987 |               4.54824  |        7.22377 |                   0.0529462 |                        0.178009 |              0.157855 |                0.0585965 |               0.185299 |                0.212567 |      26.1068 |      1153.95 |      312.367 |      13612.4 |       752324 |       259282 |      1117.15 |      5657.21 |      3135.86 |                     3998.76 |                     10329.2 |                     6401.7   |            0.16802  |             0.186409 |               0.187544 |              0.183306 |            0.143805 |             0.0618757 |           0.0690395 |     0.130915 |     0.184599  |     0.141256  |     0.216321  |     0.22377   |     0.234029  |              0.44337  |                   0.119309  |                    0.0396248 |                   0.0166121  |              0.68215  |              0.287934 |                0.353825 |               0.0954455 |                0.756728 |               -0.259524 |               -0.521944 |               -0.1075   |             0.282353 |                  0.0714254 |                 0.341843 |                       0.156064 |   3395.38 |
| std   |     214.164 |          2.11404 |            471.108 |          3.52071  |           5.23123  |                   3.26482  |     11.332  |          3.85514 |     8.30943 |      4.10786 |               0.844406 |        1.90913 |                   0.223929  |                        0.382525 |              0.36461  |                0.234871  |               0.388545 |                0.409129 |      69.6332 |      3857.99 |      620.784 |      57986   |       214502 |       135102 |      1137.46 |      6098.87 |      1318.15 |                    19738.7  |                     41027.6 |                    24211.3   |            0.373889 |             0.389441 |               0.390353 |              0.386922 |            0.350896 |             0.240933  |           0.253524  |     0.337312 |     0.262975  |     0.219707  |     0.282145  |     0.295191  |     0.289183  |              0.116685 |                   0.0969307 |                    0.0174287 |                   0.0108278  |              0.190206 |              0.156156 |                0.104542 |               0.0713149 |                0.247786 |                0.127726 |                0.29029  |                0.095373 |             0.324247 |                  0.26545   |                 0.188791 |                       0.226294 |  11627    |
| min   |       8     |          2       |              0     |          0        |           0        |                   0        |      0      |          0       |     0       |      0       |               0        |        1       |                   0         |                        0        |              0        |                0         |               0        |                0        |      -1      |         0    |       -1     |          0   |            0 |            0 |        -1    |         0    |         0    |                        0    |                         0   |                        0     |            0        |             0        |               0        |              0        |            0        |             0         |           0         |     0        |     0         |     0         |     0         |     0         |     0         |              0        |                  -0.39375   |                    0         |                   0          |              0        |              0        |                0        |               0         |                0        |               -1        |               -1        |               -1        |             0        |                 -1         |                 0        |                       0        |      1    |
| 25%   |     164     |          9       |            246     |          0.47087  |           1        |                   0.625739 |      4      |          1       |     1       |      0       |               4.4784   |        6       |                   0         |                        0        |              0        |                0         |               0        |                0        |      -1      |       445    |      141.75  |          0   |       843300 |       172847 |         0    |      3562.1  |      2382.45 |                      639    |                      1100   |                      981.188 |            0        |             0        |               0        |              0        |            0        |             0         |           0         |     0        |     0.0250506 |     0.0250125 |     0.0285715 |     0.0285715 |     0.0285737 |              0.396167 |                   0.0577574 |                    0.0283843 |                   0.00961538 |              0.6      |              0.185185 |                0.306244 |               0.05      |                0.6      |               -0.328383 |               -0.7      |               -0.125    |             0        |                  0         |                 0.166667 |                       0        |    946    |
| 50%   |     339     |         10       |            409     |          0.539226 |           1        |                   0.690476 |      8      |          3       |     1       |      0       |               4.66408  |        7       |                   0         |                        0        |              0        |                0         |               0        |                0        |      -1      |       660    |      235.5   |       1400   |       843300 |       244572 |      1023.64 |      4355.69 |      2870.07 |                     1200    |                      2800   |                     2200     |            0        |             0        |               0        |              0        |            0        |             0         |           0         |     0        |     0.0333874 |     0.033345  |     0.0400039 |     0.0400007 |     0.0407274 |              0.453457 |                   0.119117  |                    0.0390228 |                   0.0153374  |              0.710526 |              0.28     |                0.358755 |               0.1       |                0.8      |               -0.253333 |               -0.5      |               -0.1      |             0.15     |                  0         |                 0.5      |                       0        |   1400    |
| 75%   |     542     |         12       |            716     |          0.608696 |           1        |                   0.75463  |     14      |          4       |     4       |      1       |               4.85484  |        9       |                   0         |                        0        |              0        |                0         |               0        |                0        |       4      |      1000    |      357     |       7900   |       843300 |       330980 |      2056.78 |      6019.95 |      3600.23 |                     2600    |                      8000   |                     5200     |            0        |             0        |               0        |              0        |            0        |             0         |           0         |     0        |     0.240958  |     0.150831  |     0.334218  |     0.375763  |     0.399986  |              0.508333 |                   0.177832  |                    0.0502793 |                   0.0217391  |              0.8      |              0.384615 |                0.411428 |               0.1       |                1        |               -0.186905 |               -0.3      |               -0.05     |             0.5      |                  0.15      |                 0.5      |                       0.25     |   2800    |
| max   |     731     |         23       |           8474     |        701        |        1042        |                 650        |    304      |        116       |   128       |     91       |               8.04153  |       10       |                   1         |                        1        |              1        |                1         |               1        |                1        |     377      |    298400    |    42827.9   |     843300   |       843300 |       843300 |      3613.04 |    298400    |     43567.7  |                   843300    |                    843300   |                   843300     |            1        |             1        |               1        |              1        |            1        |             1         |           1         |     1        |     0.926994  |     0.925947  |     0.919999  |     0.926534  |     0.927191  |              1        |                   0.727841  |                    0.155488  |                   0.184932   |              1        |              1        |                1        |               1         |                1        |                0        |                0        |                0        |             1        |                  1         |                 0.5      |                       1        | 843300    |

</details>

#### Scatter Plots for Features vs. Shares
- Scatter plots provide visual representation of the relationship between each feature and the target variable ('shares'). This helped in identifying patterns, outliers, or any potential relationships between the variables.

<details>
  <summary>Click to view the Scatter Plots</summary>

![Scatter plots](Scatter_Plot.png)

</details>

#### Distribution of Continuous Features
- Histograms offer a visual representation of data distribution across different continuous features. Using a logarithmic scale on the y-axis helped in visualizing features with wide-ranging frequencies.

<details>
  <summary>Click to view the Histogram Plots</summary>

![Histogram plots](Histograms.png)

</details>

#### Distribution of One-Hot Encoded Categorical Features
- The histograms for one-hot encoded categorical features provide insights into the distribution across different categories, like which data channel or which day of the week has more articles.
- We separated the visualizations for continous and categorical features for three reasons:
   - For continuous features, histograms reflect the distribution, central tendency, and spread of the data.
   - For one-hot encoded categorical features, histograms display the count or frequency of observations in each distinct category, indicating the prevalence of each category in the dataset.
   - Preprocessing techniques vary based on feature type (continuous vs categorical), affecting model performance, data integrity, and interpretability.

<details>
  <summary>Click to view the Histograms (categorical columns)</summary>

![Histogram categorical](histogram_plot.png)

</details>



### Data Preprocessing
#### Data Imputation
From the result of 

```
null_data_count = data.isnull().sum().sum()
print (f"There are total of {null_data_count} missing data")
```

We found that our data did not have any null values. Although we did not have any null values, we saw from our scatterplots that many features had outliers that could be problematic. To address these outliers, we used an isolation forest model to remove the outliers. More information on the isolation forest model can be found [here](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.IsolationForest.html#sklearn.ensemble.IsolationForest).

#### Dropping features
The ‘url’ column contains the url for each article in our observations. The ‘timedelta’ column indicates days between the article publication and the dataset acquisition. Since these two features will have no effect on how many shares an article will get, we excluded these variables when training our model.

#### Data normalization
During the data exploration phase, we saw from our scatterplots that only a few of our features (global_subjectivity and avg_positive_polarity) for example, seemed to be normally distributed. Since only a few of our features were normally distributed, we rescaled our data using MinMaxScaler() so that our feature values were between 0 and 1.

#### Data encoding
Rather than having our model be a regression model that predicts a continuous value, we decided to make it a classifier instead. This made more sense to us because predicting the exact number of shares an article will get is not as important or as useful as knowing how “interesting” an article would be. For instance, it is not important that an article will be predicted to get exactly 200 shares but rather that it would belong to a class of being an interesting article. 

In order to accomplish this, we split the number of shares (a continuous value) into different classes as such:

* Great Article: Shares >= 90%
* Good Article: 70% <= Shares < 90%
* Normal Article: 30% <= Shares < 70%
* Bad article: Shares < 30%

Where positive words such as “Great” and “Good” denote that it will get many shares and negative words denote that it will get few. For example, a share number that is above the 90th percentile out of our data would be classified as a "Great Article". We used one-hot encoding mentioned in the methods sections to encode the continuous values into binary values.

### Developing our original model
Since simple regression models have already been done by other people on this data set, we set out to develop an ensemble model that takes different predictions from multiple machine learning models before making its decision. 

The models we used for building the ensemble model were:

* A simple linear regression model
* A neural network model
* A model utilizing Random Forest
* A SVM model

Our model then takes account of the predictions made by different models, and takes the mode (or mean) as the prediction value.

The model complexity of our ensemble model can be defined by the complexity of the models the ensemble is made of. For instance, an ensemble model that considers predictions from only linear regression models would be considered simple, while an ensemble consisting of neural network models and a random forest model more complex.

Note: During testing, we found that the SVM model did not perform well (around 40% accuracy). We excluded the SVM model from our list of models because it was inaccurate and also took a very long time to train.

#### Test for overfitting (original model)
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

### Developing our improved model
TODO: Write the reasoning behind the improved model

## Conclusion
TODO: What do we think of our model? What can we do better?

TODO: Could be good to include Kenny's vision for using the model along with an article scraper for future work.

## Contributions
<!-- Things need to be added here -->

## Link to Jupyter Notebook
[Jupyter Notebook](https://github.com/WeichenZhang89/ecs-171-ecs/blob/main/Data%20Preprocessing%20%26%20ML%20model.ipynb)
