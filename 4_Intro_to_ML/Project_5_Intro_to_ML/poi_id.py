#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")
import numpy as np
import matplotlib.pyplot as plt

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data

from sklearn.preprocessing import MinMaxScaler

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = ['poi', 'deferred_income', 'restricted_stock', 'bonus', 'expenses',
                 'salary', 'exercised_stock_options'] # You will need to use more features

all_financial_features = ['salary', 'deferral_payments', 'total_payments', 'loan_advances', 'bonus',
                          'restricted_stock_deferred', 'deferred_income', 'total_stock_value', 'expenses',
                          'exercised_stock_options', 'other', 'long_term_incentive', 'restricted_stock',
                          'director_fees']

all_mailing_features = ['to_messages', 'from_poi_to_this_person', 'from_messages',
                        'from_this_person_to_poi', 'shared_receipt_with_poi', 'percent_shared_with_poi']

all_features = ['poi'] + all_financial_features + all_mailing_features

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

def data_exploration(data_dict, features):
    """
    General exploration of data.
    Function returns the total number of elements, number of POIs and non-POIs,
    number of features
    """
    poi_list = []
    non_poi_list = []
    num_of_elements = len(data_dict.keys())
    print "total number of elements:", num_of_elements
    for key, value in data_dict.iteritems():
        if value["poi"] == 1:
            poi_list.append(key)
        else:
            non_poi_list.append(key)
    print "number of POIs: ", len(poi_list)
    print "number of non-POIs:", len(non_poi_list)
    print "number of features", len(features_list)

# Routine above is using to check percent of NANs in the
# selected features. Output is the sorted bar-chart,
# names of features - x-axis, percentages - y-axis

    y_values = {}
    y_percents = []
    for key, value in data_dict.iteritems():
        for sub_key in value.keys():
            if sub_key in features:
                if not sub_key in y_values:
                    y_values[sub_key] = [0, 0]
                else:
                    y_values[sub_key][0] += 1
                    if value[sub_key] == 'NaN':
                        y_values[sub_key][1] += 1
    for key1, value1 in y_values.iteritems():
        y_percents.append(float(value1[1])/value1[0])
    y_percents_srt = sorted(y_percents)
    x_objects = tuple([el1 for (el0, el1) in sorted(zip(y_percents, features))])
    y_pos = np.arange(len(x_objects))
    plt.bar(y_pos, y_percents_srt, align="center", alpha=0.5)
    plt.xticks(y_pos, x_objects, rotation="vertical")
    plt.ylabel("Percent of NaNs")
    plt.xlabel("Feature")
    plt.tight_layout()
    plt.show()

data_exploration(data_dict, all_financial_features)


# Function for test-drawing scatter plot from data_dict
# I will need it to check for outliers, patterns, classification, etc. below

def draw_from_dict(data_dict, feature1, feature2):
    """
        Plot the scatter plot of the two features
        from a given dictionary
        (input: dictionary, x_feature, y_feature)
        and mark POIs with red color (non-POIs are green)
    """

    feature1_list = []
    feature2_list = []
    feature1_list_poi = []
    feature2_list_poi = []
    for key, value in data_dict.iteritems():
        if value['poi'] == 0:
            feature1_list.append(value[feature1])
            feature2_list.append(value[feature2])
        else:
            feature1_list_poi.append(value[feature1])
            feature2_list_poi.append(value[feature2])

    feature1_ar = np.asarray(feature1_list, dtype=float)
    feature2_ar = np.asarray(feature2_list, dtype=float)
    feature1_poi_ar = np.asarray(feature1_list_poi, dtype=float)
    feature2_poi_ar = np.asarray(feature2_list_poi, dtype=float)

    plt.scatter(feature1_ar, feature2_ar, color="g", alpha=0.5)
    plt.scatter(feature1_poi_ar, feature2_poi_ar, color="r", alpha=0.5)
    plt.xlabel(feature1)
    plt.ylabel(feature2)
    plt.show()



print draw_from_dict(data_dict, "salary", "exercised_stock_options")

### Task 2: Remove outliers

# From the picture in section 1 we can clearly see one outlier, and it seems
# this outlier may not have one of the biggest residuals if we will try
# to identify it with the regression. So, as a first step, lets make a
# possibility to remove outliers by hand.

def manual_outlier_removal(data_dict, feature1, feature2, remove_flag, f1_min, f1_max, f2_min=0, f2_max=0):
    """
    Function for deleting the outliers defined by the minimum and maximum values of features:
    :param data_dict: dictionary
    :param feature1: x_feature
    :param feature2: y_feature
    :param f1_min: minimum value for x_feature
    :param f1_max: maximum value for x_feature
    :param f2_min: minimum value for y_feature
    :param f2_max: maximum value for y_feature
    :param remove_flag: True of False
    :return: depending on the value of remove_flag: if True - delete the outlier defined by values
             of x_ and y_ features, it False - print the name (dict key) and values of x_ and y_ features
             the outlier
    """

    for key in data_dict.keys():
        value = data_dict[key]
        if value[feature1] != 'NaN' and value[feature2] != 'NaN':
            if (float(value[feature1]) < f1_min or float(value[feature1]) > f1_max) and \
                    (float(value[feature2]) < f2_min or float(value[feature2]) > f2_max):
                if remove_flag:
                    data_dict.pop(key, 0)
                else:
                    print key, feature1, ":", value[feature1], feature2, ":", value[feature2]

SALARY_MIN = 0
SALARY_MAX = 2000000
EXERCISED_STOCK_OPTIONS_MIN = 0
EXERCISED_STOCK_OPTIONS_MAX = 250000000

manual_outlier_removal(data_dict, "salary", "exercised_stock_options", False, SALARY_MIN, SALARY_MAX,
                      EXERCISED_STOCK_OPTIONS_MIN, EXERCISED_STOCK_OPTIONS_MAX)
# output 'TOTAL salary : 26704229 exercised_stock_options : 311764000'

manual_outlier_removal(data_dict, "salary", "exercised_stock_options", True, SALARY_MIN, SALARY_MAX,
                       EXERCISED_STOCK_OPTIONS_MIN, EXERCISED_STOCK_OPTIONS_MAX)
print draw_from_dict(data_dict, "salary", "exercised_stock_options")
manual_outlier_removal(data_dict, "salary", "exercised_stock_options", False, 0, 1000000)

# output:
#
# LAY KENNETH L salary : 1072321 exercised_stock_options : 34348384
# SKILLING JEFFREY K salary : 1111258 exercised_stock_options : 19250000
# FREVERT MARK A salary : 1060932 exercised_stock_options : 10433518
#
# it is an actual people, two of whom are POIs,
# we definitely don't want to remove them

### Task 3: Create new feature(s)

for key,value in data_dict.iteritems():
    value["total_messages"] = float(value["to_messages"]) + float(value["from_messages"])
    if value["shared_receipt_with_poi"] != 'NaN' and value["total_messages"] != 'nan':
        value["percent_shared_with_poi"] = value["shared_receipt_with_poi"] / value["total_messages"]
    else:
        value["percent_shared_with_poi"] = 'NaN'

#draw_from_dict(data_dict, "percent_shared_with_poi", "exercised_stock_options")
#manual_outlier_removal(data_dict, "percent_shared_with_poi", "exercised_stock_options", False, 0, 0.4, 0, 15000000)


### Store to my_dataset for easy export below.
my_dataset = data_dict

### Extract features and labels from dataset for local testing
#data = featureFormat(my_dataset, features_list, sort_keys = True)
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

# Feature selection with SelectKBest
# from sklearn.feature_selection import SelectKBest
# selector = SelectKBest(k=10)
# selected_features = selector.fit_transform(features, labels)
# feature_scores = selector.scores_
# selected_features_names = [all_features[i+1] for i in selector.get_support(indices=True)]
# selected_features_scores = [feature_scores[i] for i in selector.get_support(indices=True)]
# selected_features_ns = sorted(zip(selected_features_names, selected_features_scores), key=lambda tup: tup[1],
#                               reverse=True)
#
# print()
# print('Best features automated search output:')
# for it in selected_features_ns:
#     print it

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.

# from sklearn.naive_bayes import GaussianNB
# clf = GaussianNB()

# from sklearn.svm import SVC
# svm = SVC()
# clf = SVC(kernel='linear', class_weight={1: 7})

# from sklearn import tree
# clf = tree.DecisionTreeClassifier()

from sklearn.ensemble import AdaBoostClassifier
clf = AdaBoostClassifier()

# from sklearn import neighbors
# clf = neighbors.KNeighborsClassifier(3)
# knn = neighbors.KNeighborsClassifier()

# from sklearn.ensemble import RandomForestClassifier
# clf = RandomForestClassifier()

# from sklearn.pipeline import Pipeline
#
# estimators = [('scaling', MinMaxScaler()), ('classifying', neighbors.KNeighborsClassifier())]
# clf = Pipeline(estimators)

### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# from sklearn import grid_search
#
# params = {'C': [1, 10, 100, 1000], 'gamma': [0.001, 0.0001], 'kernel': ['linear', 'rbf']}
# clf = grid_search.GridSearchCV(svm, params)

# params = {'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'], 'n_neighbors': [3, 4, 5]}

# Example starting point. Try investigating other evaluation techniques!
from sklearn.cross_validation import train_test_split
features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)

# scores = ['precision', 'recall']
#
# for score in scores:
#     clf = grid_search.GridSearchCV(knn, params)
#     clf.fit(features_train, labels_train)
#     print("Best parameters set found on development set:")
#     print()
#     print(clf.best_params_)
#     print()
#     print("Grid scores on development set:")
#     print()
#     for params, mean_score, scores in clf.grid_scores_:
#         print("%0.3f (+/-%0.03f) for %r"
#               % (mean_score, scores.std() * 2, params))

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)