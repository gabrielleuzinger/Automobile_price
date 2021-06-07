# Analysing the 'Automobile Data Set' from the UC Irvine Machine Learning repository

This notebook is used to analyze the 'Automobile Data Set' from the UC Irvine Machine Learning repository. The dataset is available [here](https://archive.ics.uci.edu/ml/datasets/Automobile). **The objective is to create a model to predict car prices base on its attributes**.

This data set consists of three types of entities: (a) the specification of an auto in terms of various characteristics, (b) its assigned insurance risk rating, (c) its normalized losses in use as compared to other cars. Sources:

1) 1985 Model Import Car and Truck Specifications, 1985 Ward's Automotive Yearbook.
2) Personal Auto Manuals, Insurance Services Office, 160 Water Street, New York, NY 10038
3) Insurance Collision Report, Insurance Institute for Highway Safety, Watergate 600, Washington, DC 20037

The notebook is divided as follows:
    
1. Data exploration
2. Train ML model
3. Evaluate the ML model
4. Conclusion

In this notebook we test different regression models. Moreover, **only the numerical variables will be considered in the models**. Therefore, all the categorical variables will be ignored.

----------

## 1. Data exploration

In this section, we explore the characteristics of the dataset, including its dimensions and characteristics of its variables.

The dataset contains only 25 columns and 205 lines. The attributes for each column are upload from the website.


```python
import pandas as pd
import numpy as np
```

----------

### Getting the data


```python
attributes = ['symboling','normalized_losses','make','fuel_type','aspiration','num_of_doors','body_style',
           'drive_wheels','engine_location','wheel_base','length','width','height','curb_weight','engine_type',
           'num_of_cylinders','engine_size','fuel_system','bore','stroke','compression_ratio','horsepower',
           'peak_rpm','city_mpg','highway_mpg','price']
df_data = pd.read_csv('/Users/leuzinger/Dropbox/Data Science/Awari/Regressions/Automobile Data Set/imports-85(1).data',names=attributes)
df_data.reset_index(inplace=False)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
df_data.head()
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
      <th>symboling</th>
      <th>normalized_losses</th>
      <th>make</th>
      <th>fuel_type</th>
      <th>aspiration</th>
      <th>num_of_doors</th>
      <th>body_style</th>
      <th>drive_wheels</th>
      <th>engine_location</th>
      <th>wheel_base</th>
      <th>length</th>
      <th>width</th>
      <th>height</th>
      <th>curb_weight</th>
      <th>engine_type</th>
      <th>num_of_cylinders</th>
      <th>engine_size</th>
      <th>fuel_system</th>
      <th>bore</th>
      <th>stroke</th>
      <th>compression_ratio</th>
      <th>horsepower</th>
      <th>peak_rpm</th>
      <th>city_mpg</th>
      <th>highway_mpg</th>
      <th>price</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>3</td>
      <td>?</td>
      <td>alfa-romero</td>
      <td>gas</td>
      <td>std</td>
      <td>two</td>
      <td>convertible</td>
      <td>rwd</td>
      <td>front</td>
      <td>88.6</td>
      <td>168.8</td>
      <td>64.1</td>
      <td>48.8</td>
      <td>2548</td>
      <td>dohc</td>
      <td>four</td>
      <td>130</td>
      <td>mpfi</td>
      <td>3.47</td>
      <td>2.68</td>
      <td>9.0</td>
      <td>111</td>
      <td>5000</td>
      <td>21</td>
      <td>27</td>
      <td>13495</td>
    </tr>
    <tr>
      <th>1</th>
      <td>3</td>
      <td>?</td>
      <td>alfa-romero</td>
      <td>gas</td>
      <td>std</td>
      <td>two</td>
      <td>convertible</td>
      <td>rwd</td>
      <td>front</td>
      <td>88.6</td>
      <td>168.8</td>
      <td>64.1</td>
      <td>48.8</td>
      <td>2548</td>
      <td>dohc</td>
      <td>four</td>
      <td>130</td>
      <td>mpfi</td>
      <td>3.47</td>
      <td>2.68</td>
      <td>9.0</td>
      <td>111</td>
      <td>5000</td>
      <td>21</td>
      <td>27</td>
      <td>16500</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>?</td>
      <td>alfa-romero</td>
      <td>gas</td>
      <td>std</td>
      <td>two</td>
      <td>hatchback</td>
      <td>rwd</td>
      <td>front</td>
      <td>94.5</td>
      <td>171.2</td>
      <td>65.5</td>
      <td>52.4</td>
      <td>2823</td>
      <td>ohcv</td>
      <td>six</td>
      <td>152</td>
      <td>mpfi</td>
      <td>2.68</td>
      <td>3.47</td>
      <td>9.0</td>
      <td>154</td>
      <td>5000</td>
      <td>19</td>
      <td>26</td>
      <td>16500</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2</td>
      <td>164</td>
      <td>audi</td>
      <td>gas</td>
      <td>std</td>
      <td>four</td>
      <td>sedan</td>
      <td>fwd</td>
      <td>front</td>
      <td>99.8</td>
      <td>176.6</td>
      <td>66.2</td>
      <td>54.3</td>
      <td>2337</td>
      <td>ohc</td>
      <td>four</td>
      <td>109</td>
      <td>mpfi</td>
      <td>3.19</td>
      <td>3.40</td>
      <td>10.0</td>
      <td>102</td>
      <td>5500</td>
      <td>24</td>
      <td>30</td>
      <td>13950</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2</td>
      <td>164</td>
      <td>audi</td>
      <td>gas</td>
      <td>std</td>
      <td>four</td>
      <td>sedan</td>
      <td>4wd</td>
      <td>front</td>
      <td>99.4</td>
      <td>176.6</td>
      <td>66.4</td>
      <td>54.3</td>
      <td>2824</td>
      <td>ohc</td>
      <td>five</td>
      <td>136</td>
      <td>mpfi</td>
      <td>3.19</td>
      <td>3.40</td>
      <td>8.0</td>
      <td>115</td>
      <td>5500</td>
      <td>18</td>
      <td>22</td>
      <td>17450</td>
    </tr>
  </tbody>
</table>
</div>




```python
df_data.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 205 entries, 0 to 204
    Data columns (total 26 columns):
     #   Column             Non-Null Count  Dtype  
    ---  ------             --------------  -----  
     0   symboling          205 non-null    int64  
     1   normalized_losses  205 non-null    object 
     2   make               205 non-null    object 
     3   fuel_type          205 non-null    object 
     4   aspiration         205 non-null    object 
     5   num_of_doors       205 non-null    object 
     6   body_style         205 non-null    object 
     7   drive_wheels       205 non-null    object 
     8   engine_location    205 non-null    object 
     9   wheel_base         205 non-null    float64
     10  length             205 non-null    float64
     11  width              205 non-null    float64
     12  height             205 non-null    float64
     13  curb_weight        205 non-null    int64  
     14  engine_type        205 non-null    object 
     15  num_of_cylinders   205 non-null    object 
     16  engine_size        205 non-null    int64  
     17  fuel_system        205 non-null    object 
     18  bore               205 non-null    object 
     19  stroke             205 non-null    object 
     20  compression_ratio  205 non-null    float64
     21  horsepower         205 non-null    object 
     22  peak_rpm           205 non-null    object 
     23  city_mpg           205 non-null    int64  
     24  highway_mpg        205 non-null    int64  
     25  price              205 non-null    object 
    dtypes: float64(5), int64(5), object(16)
    memory usage: 41.8+ KB


----------

### Data Cleaning

After importing the data, we need to do some data cleaning. For now, we just substitute the "?" values by NaN. We also transform some numerical columns that were objects to float. Finally, we drop all the rows without a price for the car, as they do not help us train our model. Finally, as we will only consider the numerical variables in this notebook, we will drop all the columns with categorical values.


```python
df_data['num_of_doors'].replace("?","four",inplace=True)
df_data.replace("?",np.nan,inplace=True)
df_data.dropna(subset = ["price"], inplace=True)
num_cols = ['symboling', 'normalized_losses', 'wheel_base', 'length', 'width',
       'height', 'curb_weight', 'engine_size', 'bore', 'stroke',
       'compression_ratio', 'horsepower', 'peak_rpm', 'city_mpg',
       'highway_mpg', 'price']
df_data[num_cols] = df_data[num_cols].apply(pd.to_numeric, errors='coerce')

cat_cols = df_data.select_dtypes(include=['object', 'bool']).columns

df_data.drop(columns=cat_cols, inplace=True)

df_data.reset_index(drop=True,inplace=True)
df_data.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 201 entries, 0 to 200
    Data columns (total 16 columns):
     #   Column             Non-Null Count  Dtype  
    ---  ------             --------------  -----  
     0   symboling          201 non-null    int64  
     1   normalized_losses  164 non-null    float64
     2   wheel_base         201 non-null    float64
     3   length             201 non-null    float64
     4   width              201 non-null    float64
     5   height             201 non-null    float64
     6   curb_weight        201 non-null    int64  
     7   engine_size        201 non-null    int64  
     8   bore               197 non-null    float64
     9   stroke             197 non-null    float64
     10  compression_ratio  201 non-null    float64
     11  horsepower         199 non-null    float64
     12  peak_rpm           199 non-null    float64
     13  city_mpg           201 non-null    int64  
     14  highway_mpg        201 non-null    int64  
     15  price              201 non-null    int64  
    dtypes: float64(10), int64(6)
    memory usage: 25.2 KB


### Data visualization

A "quick way to get a feel of the type of data you are dealing with is to plot a histogram for each numerical attribute" (GÉRON, 2019).

**For this initial exploration, we are only considering continous values. Categorical values will be examined later**.

Therefore, we will start our analysis making some histograms that are useful for understanding the dataset. We see that some histograms are tail-heavy: they extend much farther to the right of the median than to the left. Besides, only few attributes seems to have a normal distribution.

Next, we can look to which attributes have the higher correlation with the price. First, we create a correlation matrix. Then, we make some scatter plots and a heatmap to vizualaize the correlations. **We can see that the varibales that have the stronger postive correlations with the car price are the engine size, curb weight, horsepower, and with. Besides, city mpg and highway mpg have a strong negative correlation with the price**.


```python
import matplotlib.pyplot as plt

df_data.hist(bins=50,figsize=(25, 25))

plt.show()
```


    
![png](output_15_0.png)
    



```python
corr_matrix = df_data.corr()
corr_matrix['price'].sort_values(ascending=False)
```




    price                1.000000
    engine_size          0.872335
    curb_weight          0.834415
    horsepower           0.810533
    width                0.751265
    length               0.690628
    wheel_base           0.584642
    bore                 0.543436
    normalized_losses    0.203254
    height               0.135486
    stroke               0.082310
    compression_ratio    0.071107
    symboling           -0.082391
    peak_rpm            -0.101649
    city_mpg            -0.686571
    highway_mpg         -0.704692
    Name: price, dtype: float64




```python
import seaborn as sns

plt.figure(figsize=(20, 30))
sns.set_theme()
sns.set_context("notebook", font_scale=1.5)
sns.heatmap(corr_matrix,annot=True)
plt.show()
```


    
![png](output_17_0.png)
    


----------

### Creating the Train and Test sets

Creating a test set at the beginning of the project avoid *data snooping* bias, i.e., "when you estimate the generalization error using the test set, your estimate will be too optimistic, and you will launch a system that will not perform as well as expected" (GÉRON, 2019). To avoid this problem, we divide our data into a train and a test set. 

Besides, to avoid introducing a sampling bias into the sets, we will use a stratified sampling, taking the car brand as the reference because it is the categorical variable with more unique values. By doing this, the test set generated using stratified sampling has symboling category proportions almost identical to those in the full dataset.


```python
from sklearn.model_selection import train_test_split

car_X = df_data.drop('price',axis=1)
car_y = df_data['price']

X_train, X_test, y_train, y_test = train_test_split(car_X, car_y, test_size=0.2)
```


```python
car_X_train = X_train.copy()
car_y_train = y_train.copy()
car_X_test = X_test.copy()
car_y_test = y_test.copy()
```

----------

### Preparing the data for ML algorithms

Before creating the ML models, we need to prepare the data so that the ML algorithms will work properly.

First, we need to clean missing values from the dataset. We have three option to deal with it [(GÉRON, 2019)](https://www.amazon.com.br/Hands-Machine-Learning-Scikit-Learn-TensorFlow/dp/1492032646):

1. Get ride of the rows with missing values
2. Get ride of the whole attirbute that have missing values
3. Set the values to some value (the median, the mean, zero, etc)

We use the median value. We create a pipeline to be used with the models we will test next.

**Second, we need to put all the attributes in the same scale because "Machine Learning algorithms don’t perform well when the input numerical attributes have very different scales" (GÉRON, 2019). To do this we standardized all the numerical variables**.


```python
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

def estimator_transf(estimator):
    imputer = SimpleImputer(strategy='median')
    pipeline = Pipeline(steps=[('i', imputer), ('m', estimator)])
    return pipeline

def estimator_scaler(estimator):
    imputer = SimpleImputer(strategy='median')
    pipeline = Pipeline(steps=[('i', imputer), ('scaler',StandardScaler()),('model', estimator)])
    return pipeline  
```

----------

## 2. Train ML model

After preparing the data set, we are ready to select and train our ML model to predict the car price.

We start with Linear Regression (LR) model. "A regression model, such as linear regression, models an output value based on a linear combination of input values" [(BROWNLEE, 2020)](https://machinelearningmastery.com/introduction-to-time-series-forecasting-with-python/).

Then, we can try some regularized linear models. This kind of model constrain the weights of the model, avoiding overfitting (GÉRON, 2019). We try three regularized linear models [(BROWNLEE, 2016)](https://machinelearningmastery.com/machine-learning-with-python/):

1. Ridge regression. This model assumes that the input variables have a Gaussian distribution, that input variables are relevant to the output variable, and that they are not highly correlated with each other.
2. Lasso regression. This model is a modification of the LR model,"where the loss function is modified to minimize the complexity of the model measured as the sum absolute value of the coefficient values" (BROWNLEE, 2016).
3. Elastic Net. This model combines the Ridge and the Lasso models. "It seeks to minimize the complexity of the regression model (magnitude and number of regression coefficients) by penalizing the model using both the L2-norm (sum squared coefficient values) and the L1-norm (sum absolute coefficient values)" (BROWNLEE, 2016).

Finally, we also try some nonlinear algorithms:

1. Classification and Regression Trees (CART). It uses "the train- ing data to select the best points to split the data in order to minimize a cost metric" (BROWNLEE, 2016).
2. Support Vector Regression (SVR). This model is an extension of the Support Vector Machines (SVM) developed for binary classification.
3. k-Nearest Neighbors (KNN). This model "locates the k most similar instances in the training dataset for a new data instance" (BROWNLEE, 2016).

The models are evaluated using the mean absolute error (MAE) and root square mean error (RMSE). RMSE punish larger errors more than smaller errors, inflating or magnifying the mean error score. This is due to the square of the error value. MAE does not give more or less weight to different types of errors and instead the scores increase linearly with increases in error. MAE is the simplest evaluation metric and most easily interpreted ([HALE, 2020](https://towardsdatascience.com/which-evaluation-metric-should-you-use-in-machine-learning-regression-problems-20cdaef258e); [BROWNLEE, 2021](https://machinelearningmastery.com/regression-metrics-for-machine-learning/)).

Besides, "the key to a fair comparison of machine learning algorithms is ensuring that each algorithm is evaluated in the same way on the same data. You can achieve this by forcing each algorithm to be evaluated on a consistent test harness" (BROWNLEE, 2016). In this project, we do this by using the same split in the cross validation. We use the KFold function from the sklearn library with a random value rs as the random_satate parameter. Although the rs value change everytime the notebook is run, once it is set, the same rs value is used in all the models. This guarantees that all the models are evaluated on the same data.

The result of the tests of the models with the training data show that **the CART is the best model**. It has the lowest MAE and RMSE.

However, differing scales of the raw data may be negatively impacting the performance of some of the models. Therefore, we test the models again, but this time we standardized the data set.

We can see that the performance of all models improved with standardization. However, CART is still outperforming other models. So we can choose CART as our ML model.


```python
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_validate
from sklearn.model_selection import KFold

def estimator_cross_val (model,estimator,pipe,matriz,rs):
    pipe_ = pipe(estimator)
    scoring = ['neg_mean_absolute_error', 'neg_root_mean_squared_error']
    kfold = KFold(n_splits=10, random_state=rs,shuffle=True)
    scores = cross_validate(pipe_,car_X_train,car_y_train,cv=kfold,scoring=scoring)
    
    mae_scores = -scores.get('test_neg_mean_absolute_error')
    mae_mean = mae_scores.mean()
    mae_std = mae_scores.std()
    
    rmse_scores = -scores.get('test_neg_root_mean_squared_error')
    rmse_mean = rmse_scores.mean()
    rmse_std = rmse_scores.std()
    
    results_ = [model,mae_mean,mae_std,rmse_mean,rmse_std]
    results_ = pd.Series(results_, index = matriz.columns)
    results = matriz.append(results_,ignore_index=True)
    return results
```


```python
from random import randrange
#from random import random
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
import warnings

rs = randrange(10000)
matriz = pd.DataFrame(columns=['model','MAE_mean','MAE_std','RMSE_mean','RMSE_std'])

matriz = estimator_cross_val('Linear Regression',LinearRegression(),estimator_transf,matriz,rs)
matriz = estimator_cross_val('Ridge Regression',Ridge(),estimator_transf,matriz,rs)
matriz = estimator_cross_val('Lasso',Lasso(alpha=0.1),estimator_transf,matriz,rs)
matriz = estimator_cross_val('Elastic Net',ElasticNet(alpha=0.1, l1_ratio=0.5),estimator_transf,matriz,rs)
matriz = estimator_cross_val('KNN',KNeighborsRegressor(),estimator_transf,matriz,rs)
matriz = estimator_cross_val('CART',DecisionTreeRegressor(),estimator_transf,matriz,rs)
matriz = estimator_cross_val('SVR',SVR(),estimator_transf,matriz,rs)

matriz
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
      <th>model</th>
      <th>MAE_mean</th>
      <th>MAE_std</th>
      <th>RMSE_mean</th>
      <th>RMSE_std</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Linear Regression</td>
      <td>2403.116376</td>
      <td>441.157350</td>
      <td>3268.857268</td>
      <td>824.456687</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Ridge Regression</td>
      <td>2390.520811</td>
      <td>414.913015</td>
      <td>3244.085625</td>
      <td>768.385175</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Lasso</td>
      <td>2403.065810</td>
      <td>441.066071</td>
      <td>3268.763125</td>
      <td>824.207429</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Elastic Net</td>
      <td>2369.170629</td>
      <td>368.929932</td>
      <td>3199.399124</td>
      <td>665.182718</td>
    </tr>
    <tr>
      <th>4</th>
      <td>KNN</td>
      <td>2030.277500</td>
      <td>655.869904</td>
      <td>3144.075203</td>
      <td>946.750557</td>
    </tr>
    <tr>
      <th>5</th>
      <td>CART</td>
      <td>1736.275000</td>
      <td>543.044958</td>
      <td>2353.658439</td>
      <td>698.610918</td>
    </tr>
    <tr>
      <th>6</th>
      <td>SVR</td>
      <td>5419.272130</td>
      <td>1478.876285</td>
      <td>7899.508262</td>
      <td>2530.321373</td>
    </tr>
  </tbody>
</table>
</div>




```python
matriz2 = pd.DataFrame(columns=['model','MAE_mean','MAE_std','RMSE_mean','RMSE_std'])

matriz2 = estimator_cross_val('Linear Regression',LinearRegression(),estimator_scaler,matriz2,rs)
matriz2 = estimator_cross_val('Ridge Regression',Ridge(),estimator_scaler,matriz2,rs)
matriz2 = estimator_cross_val('Lasso',Lasso(alpha=0.1),estimator_scaler,matriz2,rs)
matriz2 = estimator_cross_val('Elastic Net',ElasticNet(alpha=0.1, l1_ratio=0.5),estimator_scaler,matriz2,rs)
matriz2 = estimator_cross_val('KNN',KNeighborsRegressor(),estimator_scaler,matriz2,rs)
matriz2 = estimator_cross_val('CART',DecisionTreeRegressor(),estimator_scaler,matriz2,rs)
matriz2 = estimator_cross_val('SVR',SVR(),estimator_scaler,matriz2,rs)
matriz2
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
      <th>model</th>
      <th>MAE_mean</th>
      <th>MAE_std</th>
      <th>RMSE_mean</th>
      <th>RMSE_std</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Linear Regression</td>
      <td>2403.116376</td>
      <td>441.157350</td>
      <td>3268.857268</td>
      <td>824.456687</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Ridge Regression</td>
      <td>2368.312324</td>
      <td>476.539069</td>
      <td>3236.154444</td>
      <td>847.627508</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Lasso</td>
      <td>2402.820834</td>
      <td>441.432629</td>
      <td>3268.737753</td>
      <td>824.680075</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Elastic Net</td>
      <td>2317.221346</td>
      <td>504.913302</td>
      <td>3188.935555</td>
      <td>828.986312</td>
    </tr>
    <tr>
      <th>4</th>
      <td>KNN</td>
      <td>2206.910000</td>
      <td>752.608974</td>
      <td>3249.923401</td>
      <td>1172.079984</td>
    </tr>
    <tr>
      <th>5</th>
      <td>CART</td>
      <td>1785.084375</td>
      <td>509.299840</td>
      <td>2555.319119</td>
      <td>907.515508</td>
    </tr>
    <tr>
      <th>6</th>
      <td>SVR</td>
      <td>5410.848841</td>
      <td>1478.345879</td>
      <td>7891.728383</td>
      <td>2530.981299</td>
    </tr>
  </tbody>
</table>
</div>



---------

# 3. Evaluate the ML model

Now evaluate the performance of our ML model in the test set, to see how it perform with unseen data.

We see that the model performs as expected with the test set. We verify that MAE and RMSE both have values close to those we estimated for the train data.


```python
from sklearn.metrics import mean_absolute_error, mean_squared_error

pipe = estimator_scaler(DecisionTreeRegressor())
pipe.fit(car_X_train,car_y_train)

car_y_hat = pipe.predict(car_X_test)
final_mae = mean_absolute_error(car_y_test,car_y_hat)
final_mse = mean_squared_error(car_y_test,car_y_hat)
final_rmse = np.sqrt(final_mse)
print('MAE:  %.2f'%final_mae,'\nRMSE: %.2f'%final_rmse)
```

    MAE:  1972.80 
    RMSE: 2933.36


----------------------

## 4. Conclusion

In this notebook, we were able to create a model to predict car prices base on its attributes. After preparing the data, we tried four different regression models. 

We verified that the CART model performed best with the train set. Then, we evaluated this model using the test set we separeted during the data preparation.

**We verify that the model had the same performance with the test set as with the train set. Moreover, the values of the MAE and RMSE are accepbtable for this kind of prediction**. 
