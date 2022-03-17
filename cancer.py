import time 
import random 
import numpy as np 
import pandas as pd 
import seaborn as sns 
import matplotlib.pyplot as plt 
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.model_selection import KFold, cross_val_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from urllib.request import urlopen
import joblib

plt.style.use('ggplot')

breast_cancer = pd.read_csv(r'C:\Users\Harish Sundaralingam\Desktop\Stats and ML\Machine Learning\tumor_data.csv')

# Setting ID number as index 
breast_cancer.set_index(['id'],inplace=True)
# Convert to binary to help later on with modles and plots
breast_cancer['diagnosis'] = breast_cancer['diagnosis'].map({'M':1, 'B':0})


# # Missing Value Check and Data Cleaning 
breast_cancer.apply(lambda x: x.isnull().sum())
del breast_cancer['Unnamed: 32']

print(breast_cancer.head())

print("Here's the dimension of our data frame:", breast_cancer.shape,'\n')
print("Here's the data type of our columns:\n", breast_cancer.dtypes)

print(breast_cancer.describe())

feature_space = breast_cancer.iloc[:,breast_cancer.columns != 'diagnosis']
feature_class = breast_cancer.iloc[:,breast_cancer.columns == 'diagnosis']

training_set, test_set, class_set, test_class_set = train_test_split(feature_space,feature_class,test_size = 0.20,random_state = 42)

# Cleaning test set to avoid future warning message
class_set = class_set.values.ravel()
test_class_set = test_class_set.values.ravel()

## Random Forest Classifier

# Set the random state for reproducibility
fit_rf = RandomForestClassifier(random_state=42)

## Hyper-Parameter Optimization using GridSearchCV 

np.random.seed(42)
start = time.time()

param_dist = {'max_depth': [6,7,8],
             'bootstrap':[True, False], # not necessary as randomforest will always set to true
             'max_features':['auto','sqrt','log2',None],
             'criterion':['gini','entropy']
             }

cv_rf = GridSearchCV(fit_rf, cv=10,
                    param_grid = param_dist,
                    n_jobs =3)

cv_rf.fit(training_set,class_set)
print('Best parameters using Grid Search: ', cv_rf.best_params_)
end = time.time()
print(r'Time taken in Grid Search: {0:.2f}',(end-start))

fit_rf.set_params(criterion = 'gini',max_depth = 7, max_features = 'log2')


## Out of Bag Rate (oob rate)

fit_rf.set_params(warm_start = True,oob_score = True)

min_estimators = 15
max_estimators = 1000

error_rate = {}

for i in range(min_estimators, max_estimators + 1):
    fit_rf.set_params(n_estimators = i)
    fit_rf.fit(training_set, class_set)
    
    oob_error = 1 - fit_rf.oob_score_
    error_rate[i] = oob_error

oob_series = pd.Series(error_rate) # Convert to Series for easy plotting

fig, ax = plt.subplots(figsize = (10,10))

ax.set_facecolor('#ffff')

oob_series.plot(kind='line', color ='red')
plt.axhline(0.045, color= '#875FDB', linestyle = '--')
plt.axhline(0.04, color= '#875FDB', linestyle = '--')
plt.xlabel('no.of.estimators')
plt.ylabel('OOB Error Rate')
plt.title("OOB Error Rate Across various Forest Sizes \n(From 15 to 1000 trees)")
plt.show()

print('OOB Error rate for 400 trees is: {0:.5f}'.format(oob_series[400]))


# Refine the tree via OOB Output
fit_rf.set_params(n_estimators = 400,
                 bootstrap = True, 
                 warm_start = False, 
                 oob_score = False)


## Train the Random Forest
fit_rf.fit(training_set,class_set)

## Variable Importance

def variable_importance(fit):
    """
    Purpose
    ----------
    Checks if model is fitted CART model then produces variable importance
    and respective indices in dictionary.

    Parameters
    ----------
    * fit:  Fitted model containing the attribute feature_importances_

    Returns
    ----------
    Dictionary containing arrays with importance score and index of columns
    ordered in descending order of importance.
    """
    try:
        if not hasattr(fit, 'fit'):
            return print("'{0}' is not an instantiated model from scikit-learn".format(fit)) 

        # Captures whether the model has been trained
        if not vars(fit)["estimators_"]:
            return print("Model does not appear to be trained.")
    except KeyError:
        print("Model entered does not contain 'estimators_' attribute.")

    importances = fit.feature_importances_
    indices = np.argsort(importances)[::-1]
    return {'importance': importances,
            'index': indices}

var_imp_rf = variable_importance(fit_rf)

importances_rf = var_imp_rf['importance']

indices_rf = var_imp_rf['index']

names_index = breast_cancer.columns


for i in range(indices_rf.shape[0]):
    plt.barh(names_index[i],importances_rf[i],color = '#875FDB')

def print_var_importance(importance, indices, name_index):
    """
    Purpose
    ----------
    Prints dependent variable names ordered from largest to smallest
    based on information gain for CART model.
    Parameters
    ----------
    * importance: Array returned from feature_importances_ for CART
                models organized by dataframe index
    * indices: Organized index of dataframe from largest to smallest
                based on feature_importances_
    * name_index: Name of columns included in model

    Returns
    ----------
    Prints feature importance in descending order
    """
    print("Feature ranking:")

    for f in range(0, indices.shape[0]):
        i = f
        print("{0}. The feature '{1}' has a Mean Decrease in Impurity of {2:.5f}"
              .format(f + 1,
                      name_index[indices[i]],
                      importance[indices[f]]))



print_var_importance(importances_rf, indices_rf, names_index)


## Predictions
predictions = fit_rf.predict(test_set)

def create_conf_mat(test_class_set, predictions):
    """Function returns confusion matrix comparing two arrays"""
    if (len(test_class_set.shape) != len(predictions.shape) == 1):
        return print('Arrays entered are not 1-D.\nPlease enter the correctly sized sets.')
    elif (test_class_set.shape != predictions.shape):
        return print('Number of values inside the Arrays are not equal to each other.\nPlease make sure the array has the same number of instances.')
    else:
        # Set Metrics
        test_crosstb_comp = pd.crosstab(index = test_class_set,
                                        columns = predictions)
        # Changed for Future deprecation of as_matrix
        test_crosstb = test_crosstb_comp.values
        return test_crosstb

## Confusion Matrix
conf_mat = create_conf_mat(test_class_set, predictions)
sns.heatmap(conf_mat, annot=True, fmt='d', cbar=False, linewidths=0.5,)
plt.xlabel('Predicted Values')
plt.ylabel('Actual Values')
plt.title('Actual vs. Predicted Confusion Matrix')
plt.show()


## Accuracy
accuracy_rf = fit_rf.score(test_set,test_class_set)
print("Here is out mean accuracy on the test set: {0:.3f}".format(accuracy_rf))


## Error Rate
# Calculating the test error rate

test_error_rate_rf = 1 - accuracy_rf

print("The test error rate for our model is: {0:.4f}".format(test_error_rate_rf))


# Area Under the Curve (AUC)

# We grab the second array from the output which corresponds to
# to the predicted probabilites of positive classes 
# Ordered wrt fit.classes_ in our case [0, 1] where 1 is our positive class
predictions_prob = fit_rf.predict_proba(test_set)[:, 1]

fpr2, tpr2, _ = roc_curve(test_class_set,
                          predictions_prob,
                          pos_label = 1)


auc_rf = auc(fpr2, tpr2)


def plot_roc_curve(fpr, tpr, auc, estimator, xlim=None, ylim=None):
    """
    Purpose
    ----------
    Function creates ROC Curve for respective model given selected parameters.
    Optional x and y limits to zoom into graph

    Parameters
    ----------
    * fpr: Array returned from sklearn.metrics.roc_curve for increasing
            false positive rates
    * tpr: Array returned from sklearn.metrics.roc_curve for increasing
            true positive rates
    * auc: Float returned from sklearn.metrics.auc (Area under Curve)
    * estimator: String represenation of appropriate model, can only contain the
    following: ['knn', 'rf', 'nn']
    * xlim: Set upper and lower x-limits
    * ylim: Set upper and lower y-limits
    """
    my_estimators = {'knn': ['Kth Nearest Neighbor', 'deeppink'],
              'rf': ['Random Forest', 'red'],
              'nn': ['Neural Network', 'purple']}

    try:
        plot_title = my_estimators[estimator][0]
        color_value = my_estimators[estimator][1]
    except KeyError as e:
        print("'{0}' does not correspond with the appropriate key inside the estimators dictionary. \nPlease refer to function to check `my_estimators` dictionary.".format(estimator))
        raise

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_facecolor('#fafafa')

    plt.plot(fpr, tpr,
             color=color_value,
             linewidth=1)
    plt.title('ROC Curve For {0} (AUC = {1: 0.3f})'              .format(plot_title, auc))

    plt.plot([0, 1], [0, 1], 'k--', lw=2) # Add Diagonal line
    plt.plot([0, 0], [1, 0], 'k--', lw=2, color = 'black')
    plt.plot([1, 0], [1, 1], 'k--', lw=2, color = 'black')
    if xlim is not None:
        plt.xlim(*xlim)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.show()
    plt.close()


plot_roc_curve(fpr2, tpr2, auc_rf, 'rf',
               xlim=(-0.01, 1.05), 
               ylim=(0.001, 1.05))


## Classification Report

dx = ['Benign', 'Malignant']
def print_class_report(predictions, alg_name):
    """
    Purpose
    ----------
    Function helps automate the report generated by the
    sklearn package. Useful for multiple model comparison

    Parameters:
    ----------
    predictions: The predictions made by the algorithm used
    alg_name: String containing the name of the algorithm used
    
    Returns:
    ----------
    Returns classification report generated from sklearn. 
    """
    print('Classification Report for {0}:'.format(alg_name))
    print(classification_report(predictions, 
            test_class_set, 
            target_names = dx))


class_report = print_class_report(predictions, 'Random Forest')

joblib.dump(fit_rf,'cancer_96.pkl')