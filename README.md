## Importing the necessary packages


```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use("ggplot")
from sklearn.model_selection import train_test_split as split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.model_selection import GridSearchCV
```

## Importing CSV Data


```python
# Load dataset
diabetes = pd.read_csv("diabetes.csv")

# Check the first 6 rows
diabetes.head(6)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Pregnancies</th>
      <th>Glucose</th>
      <th>BloodPressure</th>
      <th>SkinThickness</th>
      <th>Insulin</th>
      <th>BMI</th>
      <th>DiabetesPedigreeFunction</th>
      <th>Age</th>
      <th>Outcome</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>6</td>
      <td>148</td>
      <td>72</td>
      <td>35</td>
      <td>0</td>
      <td>33.6</td>
      <td>0.627</td>
      <td>50</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>85</td>
      <td>66</td>
      <td>29</td>
      <td>0</td>
      <td>26.6</td>
      <td>0.351</td>
      <td>31</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>8</td>
      <td>183</td>
      <td>64</td>
      <td>0</td>
      <td>0</td>
      <td>23.3</td>
      <td>0.672</td>
      <td>32</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>89</td>
      <td>66</td>
      <td>23</td>
      <td>94</td>
      <td>28.1</td>
      <td>0.167</td>
      <td>21</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>137</td>
      <td>40</td>
      <td>35</td>
      <td>168</td>
      <td>43.1</td>
      <td>2.288</td>
      <td>33</td>
      <td>1</td>
    </tr>
    <tr>
      <th>5</th>
      <td>5</td>
      <td>116</td>
      <td>74</td>
      <td>0</td>
      <td>0</td>
      <td>25.6</td>
      <td>0.201</td>
      <td>30</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Checking the dimension of the data(rows and columns)
diabetes.shape
```




    (768, 9)



As observed above, there have 768 observations and 9 variables. The first 8 columns represent the features and the last column represent the target.

## Creating features/Input/independent and target/dependent

### Features


```python
X = diabetes.drop(["Outcome"], axis=1)
X.head(3)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Pregnancies</th>
      <th>Glucose</th>
      <th>BloodPressure</th>
      <th>SkinThickness</th>
      <th>Insulin</th>
      <th>BMI</th>
      <th>DiabetesPedigreeFunction</th>
      <th>Age</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>6</td>
      <td>148</td>
      <td>72</td>
      <td>35</td>
      <td>0</td>
      <td>33.6</td>
      <td>0.627</td>
      <td>50</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>85</td>
      <td>66</td>
      <td>29</td>
      <td>0</td>
      <td>26.6</td>
      <td>0.351</td>
      <td>31</td>
    </tr>
    <tr>
      <th>2</th>
      <td>8</td>
      <td>183</td>
      <td>64</td>
      <td>0</td>
      <td>0</td>
      <td>23.3</td>
      <td>0.672</td>
      <td>32</td>
    </tr>
  </tbody>
</table>
</div>



### Creating Array of features


```python
X = X.values
```

### Target variable


```python
y = diabetes["Outcome"]
y.head(3)
```




    0    1
    1    0
    2    1
    Name: Outcome, dtype: int64



### Creating array target variable


```python
y = y.values
```

## Split the data randomly into training and test set
* It is a best practice to perform our split in such a way that out split reflects theproportions of labels in the data. 
* If positive cases occurs in 10% of the observations the we want 10% of the label in training and test set to represent positive cases.This is achieved by setting `stratify = y`. 
* In other words, we want labels to be split in train and test set as they are in the original dataset.

Also we create a test set of size of about 20% of the dataset.


```python
X_train, X_test, y_train, y_test = split(X, 
                                         y, 
                                         test_size=0.2, 
                                         random_state=21, 
                                         stratify=y)
```

## create a classifier using k-Nearest Neighbors algorithm
Accuracies for different values of k


```python
neighbors = np.arange(1,20)
# Setup arrays to store training and test accuracies
train_accuracies = {}
test_accuracies = {}

for neighbor in neighbors:
    #Setup a knn classifier with k neighbors
    knn = KNeighborsClassifier(n_neighbors=neighbor)
    #Fit the model
    knn.fit(X_train, y_train)
    #Compute accuracy on the training set
    train_accuracies[neighbor] = knn.score(X_train, y_train)
    #Compute accuracy on the test set
    test_accuracies[neighbor] = knn.score(X_test, y_test)
    
```


```python
train = pd.DataFrame(list(train_accuracies.items()), columns=['n_neighbors', 'Training Accuracy'])
test = pd.DataFrame(list(test_accuracies.items()), columns=['n_neighbors', 'Testing Accuracy'])
Accuracy = train.merge(test, on="n_neighbors")
Accuracy
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>n_neighbors</th>
      <th>Training Accuracy</th>
      <th>Testing Accuracy</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>1.000000</td>
      <td>0.655844</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>0.835505</td>
      <td>0.720779</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>0.858306</td>
      <td>0.681818</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>0.799674</td>
      <td>0.694805</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>0.796417</td>
      <td>0.733766</td>
    </tr>
    <tr>
      <th>5</th>
      <td>6</td>
      <td>0.780130</td>
      <td>0.720779</td>
    </tr>
    <tr>
      <th>6</th>
      <td>7</td>
      <td>0.794788</td>
      <td>0.714286</td>
    </tr>
    <tr>
      <th>7</th>
      <td>8</td>
      <td>0.778502</td>
      <td>0.707792</td>
    </tr>
    <tr>
      <th>8</th>
      <td>9</td>
      <td>0.798046</td>
      <td>0.727273</td>
    </tr>
    <tr>
      <th>9</th>
      <td>10</td>
      <td>0.783388</td>
      <td>0.733766</td>
    </tr>
    <tr>
      <th>10</th>
      <td>11</td>
      <td>0.788274</td>
      <td>0.740260</td>
    </tr>
    <tr>
      <th>11</th>
      <td>12</td>
      <td>0.780130</td>
      <td>0.746753</td>
    </tr>
    <tr>
      <th>12</th>
      <td>13</td>
      <td>0.773616</td>
      <td>0.740260</td>
    </tr>
    <tr>
      <th>13</th>
      <td>14</td>
      <td>0.780130</td>
      <td>0.733766</td>
    </tr>
    <tr>
      <th>14</th>
      <td>15</td>
      <td>0.778502</td>
      <td>0.746753</td>
    </tr>
    <tr>
      <th>15</th>
      <td>16</td>
      <td>0.789902</td>
      <td>0.733766</td>
    </tr>
    <tr>
      <th>16</th>
      <td>17</td>
      <td>0.786645</td>
      <td>0.740260</td>
    </tr>
    <tr>
      <th>17</th>
      <td>18</td>
      <td>0.793160</td>
      <td>0.733766</td>
    </tr>
    <tr>
      <th>18</th>
      <td>19</td>
      <td>0.786645</td>
      <td>0.720779</td>
    </tr>
  </tbody>
</table>
</div>




```python
plt.figure(figsize=(8,6))
plt.title("KNN: Varying number of neighbors")
plt.plot(neighbors, train_accuracies.values(), label = "Training Accuracy")
plt.plot(neighbors, test_accuracies.values(), label = "Testing Accuracy")
plt.legend()
plt.xlabel("Number of Neighbors")
plt.ylabel("Accuracy")
plt.show()
```


    
![png](output_20_0.png)
    


We can observe above that we get maximum testing accuracy for k=9. So lets create a KNeighborsClassifier with number of neighbors as 9.


```python
#Setup a knn classifier with k neighbors
knn = KNeighborsClassifier(n_neighbors=12)
knn.fit(X_train, y_train)
knn.score(X_test, y_test)
```




    0.7467532467532467



## Confusion Matrix

A confusion matrix is a table that is often used to describe the performance of a classification model (or "classifier") on a set of test data for which the true values are known. Scikit-learn provides facility to calculate confusion matrix using the confusion_matrix method.


```python
y_pred = knn.predict(X_test)
```


```python
confusion_matrix(y_test,y_pred)
```




    array([[91,  9],
           [30, 24]], dtype=int64)



Considering confusion matrix above:

True negative = 91

False positive = 9

True postive = 24

Fasle negative = 30


```python
pd.crosstab(y_test, y_pred, rownames=["True"], colnames=["Predicted"], margins=True)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>Predicted</th>
      <th>0</th>
      <th>1</th>
      <th>All</th>
    </tr>
    <tr>
      <th>True</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>91</td>
      <td>9</td>
      <td>100</td>
    </tr>
    <tr>
      <th>1</th>
      <td>30</td>
      <td>24</td>
      <td>54</td>
    </tr>
    <tr>
      <th>All</th>
      <td>121</td>
      <td>33</td>
      <td>154</td>
    </tr>
  </tbody>
</table>
</div>



## Classification Report


```python
print(classification_report(y_test, y_pred))
```

                  precision    recall  f1-score   support
    
               0       0.75      0.91      0.82       100
               1       0.73      0.44      0.55        54
    
        accuracy                           0.75       154
       macro avg       0.74      0.68      0.69       154
    weighted avg       0.74      0.75      0.73       154
    
    

## ROC (Reciever Operating Charecteristic) curve

It is a plot of the true positive rate against the false positive rate for the different possible cutpoints of a diagnostic test.

An ROC curve demonstrates several things:

1) It shows the tradeoff between sensitivity and specificity (any increase in sensitivity will be accompanied by a decrease in specificity).

2) The closer the curve follows the left-hand border and then the top border of the ROC space, the more accurate the test.

3) The closer the curve comes to the 45-degree diagonal of the ROC space, the less accurate the test.

4) The area under the curve is a measure of test accuracy.


```python
y_pred_prob = knn.predict_proba(X_test)[:,1]

```


```python
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
```


```python
plt.figure(figsize=(8,6))
plt.plot([0,1],[0,1], 'k--')
plt.plot(fpr, tpr, label = 'Knn')
plt.xlabel('fpr')
plt.ylabel('tpr')
plt.title("KNN: ROC curve(n_neighbors=12)")
plt.show()
```


    
![png](output_33_0.png)
    


## Area under ROC curve


```python
roc_auc_score(y_test, y_pred_prob)
```




    0.7707407407407407



## Cross Validation

Now before getting into the details of Hyperparamter tuning, let us understand the concept of Cross validation.

The trained model's performance is dependent on way the data is split. It might not representative of the model’s ability to generalize.

The solution is cross validation.

Cross-validation is a technique to evaluate predictive models by partitioning the original sample into a training set to train the model, and a test set to evaluate it.

In k-fold cross-validation, the original sample is randomly partitioned into k equal size subsamples. Of the k subsamples, a single subsample is retained as the validation data for testing the model, and the remaining k-1 subsamples are used as training data. The cross-validation process is then repeated k times (the folds), with each of the k subsamples used exactly once as the validation data. The k results from the folds can then be averaged (or otherwise combined) to produce a single estimation. The advantage of this method is that all observations are used for both training and validation, and each observation is used for validation exactly once.

Hyperparameter tuning

The value of k (i.e 7) we selected above was selected by observing the curve of accuracy vs number of neighbors. This is a primitive way of hyperparameter tuning.

There is a better way of doing it which involves:

1) Trying a bunch of different hyperparameter values

2) Fitting all of them separately

3) Checking how well each performs

4) Choosing the best performing one

5) Using cross-validation every time

Scikit-learn provides a simple way of achieving this using GridSearchCV i.e Grid Search cross-validation.


```python
# In case of classifier like knn the parameter to be tuned is n_neighbors
paramgrid = {'n_neighbors' :np.arange(1,50)}
knn = KNeighborsClassifier()
knn_cv = GridSearchCV(knn, paramgrid, cv= 5)
knn_cv.fit(X, y)
```




<style>#sk-container-id-1 {color: black;background-color: white;}#sk-container-id-1 pre{padding: 0;}#sk-container-id-1 div.sk-toggleable {background-color: white;}#sk-container-id-1 label.sk-toggleable__label {cursor: pointer;display: block;width: 100%;margin-bottom: 0;padding: 0.3em;box-sizing: border-box;text-align: center;}#sk-container-id-1 label.sk-toggleable__label-arrow:before {content: "▸";float: left;margin-right: 0.25em;color: #696969;}#sk-container-id-1 label.sk-toggleable__label-arrow:hover:before {color: black;}#sk-container-id-1 div.sk-estimator:hover label.sk-toggleable__label-arrow:before {color: black;}#sk-container-id-1 div.sk-toggleable__content {max-height: 0;max-width: 0;overflow: hidden;text-align: left;background-color: #f0f8ff;}#sk-container-id-1 div.sk-toggleable__content pre {margin: 0.2em;color: black;border-radius: 0.25em;background-color: #f0f8ff;}#sk-container-id-1 input.sk-toggleable__control:checked~div.sk-toggleable__content {max-height: 200px;max-width: 100%;overflow: auto;}#sk-container-id-1 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {content: "▾";}#sk-container-id-1 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-1 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-1 input.sk-hidden--visually {border: 0;clip: rect(1px 1px 1px 1px);clip: rect(1px, 1px, 1px, 1px);height: 1px;margin: -1px;overflow: hidden;padding: 0;position: absolute;width: 1px;}#sk-container-id-1 div.sk-estimator {font-family: monospace;background-color: #f0f8ff;border: 1px dotted black;border-radius: 0.25em;box-sizing: border-box;margin-bottom: 0.5em;}#sk-container-id-1 div.sk-estimator:hover {background-color: #d4ebff;}#sk-container-id-1 div.sk-parallel-item::after {content: "";width: 100%;border-bottom: 1px solid gray;flex-grow: 1;}#sk-container-id-1 div.sk-label:hover label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-1 div.sk-serial::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: 0;}#sk-container-id-1 div.sk-serial {display: flex;flex-direction: column;align-items: center;background-color: white;padding-right: 0.2em;padding-left: 0.2em;position: relative;}#sk-container-id-1 div.sk-item {position: relative;z-index: 1;}#sk-container-id-1 div.sk-parallel {display: flex;align-items: stretch;justify-content: center;background-color: white;position: relative;}#sk-container-id-1 div.sk-item::before, #sk-container-id-1 div.sk-parallel-item::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: -1;}#sk-container-id-1 div.sk-parallel-item {display: flex;flex-direction: column;z-index: 1;position: relative;background-color: white;}#sk-container-id-1 div.sk-parallel-item:first-child::after {align-self: flex-end;width: 50%;}#sk-container-id-1 div.sk-parallel-item:last-child::after {align-self: flex-start;width: 50%;}#sk-container-id-1 div.sk-parallel-item:only-child::after {width: 0;}#sk-container-id-1 div.sk-dashed-wrapped {border: 1px dashed gray;margin: 0 0.4em 0.5em 0.4em;box-sizing: border-box;padding-bottom: 0.4em;background-color: white;}#sk-container-id-1 div.sk-label label {font-family: monospace;font-weight: bold;display: inline-block;line-height: 1.2em;}#sk-container-id-1 div.sk-label-container {text-align: center;}#sk-container-id-1 div.sk-container {/* jupyter's `normalize.less` sets `[hidden] { display: none; }` but bootstrap.min.css set `[hidden] { display: none !important; }` so we also need the `!important` here to be able to override the default hidden behavior on the sphinx rendered scikit-learn.org. See: https://github.com/scikit-learn/scikit-learn/issues/21755 */display: inline-block !important;position: relative;}#sk-container-id-1 div.sk-text-repr-fallback {display: none;}</style><div id="sk-container-id-1" class="sk-top-container"><div class="sk-text-repr-fallback"><pre>GridSearchCV(cv=5, estimator=KNeighborsClassifier(),
             param_grid={&#x27;n_neighbors&#x27;: array([ 1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17,
       18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34,
       35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49])})</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class="sk-container" hidden><div class="sk-item sk-dashed-wrapped"><div class="sk-label-container"><div class="sk-label sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-1" type="checkbox" ><label for="sk-estimator-id-1" class="sk-toggleable__label sk-toggleable__label-arrow">GridSearchCV</label><div class="sk-toggleable__content"><pre>GridSearchCV(cv=5, estimator=KNeighborsClassifier(),
             param_grid={&#x27;n_neighbors&#x27;: array([ 1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17,
       18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34,
       35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49])})</pre></div></div></div><div class="sk-parallel"><div class="sk-parallel-item"><div class="sk-item"><div class="sk-label-container"><div class="sk-label sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-2" type="checkbox" ><label for="sk-estimator-id-2" class="sk-toggleable__label sk-toggleable__label-arrow">estimator: KNeighborsClassifier</label><div class="sk-toggleable__content"><pre>KNeighborsClassifier()</pre></div></div></div><div class="sk-serial"><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-3" type="checkbox" ><label for="sk-estimator-id-3" class="sk-toggleable__label sk-toggleable__label-arrow">KNeighborsClassifier</label><div class="sk-toggleable__content"><pre>KNeighborsClassifier()</pre></div></div></div></div></div></div></div></div></div></div>




```python
knn_cv.best_score_
```




    0.7578558696205755




```python
knn_cv.best_params_
```




    {'n_neighbors': 14}



Thus a knn classifier with number of neighbors as 14 achieves the best score/accuracy of 0.7578 i.e about 76%