import numpy as np
import random
from sklearn import tree
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import graphviz
import xgboost as xgb
import string
import nltk
import json
import csv
from prettytable import PrettyTable
from nltk.corpus import stopwords
from nltk.stem.porter import *
random.seed(0)

#read input files
def loadCSV(filename):
    dataSet=[]
    with open(filename,'r') as file:
        csvReader=csv.reader(file)
        for line in csvReader:
            dataSet.append(line)
    return dataSet
A = loadCSV('news-train.csv')
A.pop(0)
random.shuffle(A)
n = len(A) #number of ducuments

D = loadCSV('news-test.csv')
D.pop(0)

#read dictionary.txt
def loadtxt(filename):
    dataSet=[]
    with open(filename,'r') as file:
        txtfile = open(filename)
        for line in txtfile:
            line = line.strip('\n')
            dataSet.append(line)
    return dataSet
C = loadtxt("dictionary.txt")

# stemming tool from nltk
stemmer = PorterStemmer()
# a mapping dictionary that help remove punctuations
remove_punctuation_map = dict((ord(char), None) for char in string.punctuation)

#Get tokens
def get_tokens(text):
  # turn document into lowercase
  lowers = text.lower()
  # remove punctuations
  no_punctuation = lowers.translate(remove_punctuation_map)
  # tokenize document
  tokens = nltk.word_tokenize(no_punctuation)
  # remove stop words
  filtered = [w for w in tokens if not w in stopwords.words('english')]
  # stemming process
  stemmed = []
  for item in filtered:
      stemmed.append(stemmer.stem(item))
  # final unigrams
  return stemmed

#get unigram in news-train
B = []
for i in range(n):
    B.append(get_tokens(A[i][1]))

#category
category = []
for i in range(n):
    category.append(A[i][2])
for i in range(n):
    if category[i] == 'sport':
        category[i] = 0
    elif category[i] == 'business':
        category[i] = 1
    elif category[i] == 'politics':
        category[i] = 2
    elif category[i] == 'entertainment':
        category[i] = 3
    else:
        category[i] = 4

category_list = ["sport","business","politics","entertainment","tech"]

#Compute fij
f = np.empty((1490,1000))
f[:] = np.nan
for i in range(1490):
    for j in range(1000):
        f[i,j] = B[i].count(C[j])

#Compute max k
maxk = f.max(axis=1)
#Compute TF matrix
TF = f/maxk[:,None]

#Compute IDF
temp = f.copy()
temp[temp != 0] = 1
m = np.sum(temp, axis=0)
IDF = np.log(n/m)
IDF = IDF[None,:]

#Compute TFIDF as features
TFIDF = TF * IDF

#Divide Data into Train and Val 80%/20% Q2
num_train = int(len(A) * 0.8)
num_val = int(len(A) * 0.2)
X_news_train_train = TFIDF[:num_train,:]
X_news_train_val = TFIDF[num_train:,:]
Y_news_train_train = category[:num_train]
Y_news_train_val = category[num_train:]

#Decision Tree Random Split Q2a
def parameter_tune_2a(X_train,X_val,Y_train,Y_val):
    train_acc_all = []
    val_acc_all = []

    for i in range(2):
        train_acc = []
        val_acc = []
        ##########################
        train_X = X_train
        val_X = X_val

        train_y = Y_train
        val_y = Y_val

        if i == 0:
            dtc = tree.DecisionTreeClassifier(criterion='gini', max_depth=50)
        else:
            dtc = tree.DecisionTreeClassifier(criterion='entropy', max_depth=50)

        dtc.fit(train_X, train_y)
        train_acc.append(dtc.score(train_X, train_y))
        val_acc.append(dtc.score(val_X, val_y))
        ##########################
        avg_train_acc = sum(train_acc) / len(train_acc)
        avg_val_acc = sum(val_acc) / len(val_acc)
        if i == 0:
            print("Criterion:gini")
        else:
            print("Criterion:entropy")
        print("Training accuracy: ", avg_train_acc * 100, "%")
        print("Validation accuracy: ", avg_val_acc * 100, "%")

        train_acc_all.append(avg_train_acc)
        val_acc_all.append(avg_val_acc)

    return train_acc_all, val_acc_all

train_acc_all, val_acc_all = parameter_tune_2a(X_news_train_train,X_news_train_val,Y_news_train_train,Y_news_train_val)

#plot training/validation curves
criterion = ["gini", "engropy"]
plt.figure(1)
bar_width = 0.2
x_figure1_1 = [-0.1,0.9]
x_figure1_2 = [0.1,1.1]
x_figure1_3 = [0,1]
plt.bar(x_figure1_1, train_acc_all, width=bar_width, label="Training accuracy")
plt.bar(x_figure1_2, val_acc_all, width=bar_width, label="Validation accuracy")
plt.xlabel('Criterion')
plt.ylabel('Accuracy')
plt.xticks(x_figure1_3,criterion)
plt.legend()

#Decision Tree K-fold Q2b
kf = KFold(n_splits=5)

def parameter_tune_2b(train_val_X, train_val_y):
    train_acc_criterion = []
    val_acc_criterion = []

    #fine tune criterion:
    for i in range(2):
        train_acc = []
        val_acc = []
        for train_index, val_index in kf.split(train_val_X):
            ##########################
            train_X = train_val_X[train_index, :]
            val_X = train_val_X[val_index, :]

            train_val_y = np.array(train_val_y)   #list to ndarray
            train_y = train_val_y[train_index]
            val_y = train_val_y[val_index]

            if i == 0:
                dtc = tree.DecisionTreeClassifier(criterion='gini')
            else:
                dtc = tree.DecisionTreeClassifier(criterion='entropy')

            dtc.fit(train_X, train_y)
            train_acc.append(dtc.score(train_X, train_y))
            val_acc.append(dtc.score(val_X, val_y))
            ##########################
        avg_train_acc = sum(train_acc) / len(train_acc)
        avg_val_acc = sum(val_acc) / len(val_acc)

        if i == 0:
            print("Criterion:gini")
        else:
            print("Criterion:entropy")

        print("Training accuracy: ", avg_train_acc * 100, "%")
        print("Validation accuracy: ", avg_val_acc * 100, "%")

        train_acc_criterion.append(avg_train_acc)
        val_acc_criterion.append(avg_val_acc)

    #fine tune min_samples_leaf
    min_leafs = np.arange(1,11,1)

    train_acc_min_leafs = []
    val_acc_min_leafs = []

    for min_leaf in min_leafs:
        train_acc = []
        val_acc = []
        for train_index, val_index in kf.split(train_val_X):
            ##########################
            train_X = train_val_X[train_index, :]
            val_X = train_val_X[val_index, :]

            train_y = train_val_y[train_index]
            val_y = train_val_y[val_index]

            dtc = tree.DecisionTreeClassifier(min_samples_leaf=min_leaf)
            dtc.fit(train_X, train_y)
            train_acc.append(dtc.score(train_X, train_y))
            val_acc.append(dtc.score(val_X, val_y))
            ##########################
        avg_train_acc = sum(train_acc) / len(train_acc)
        avg_val_acc = sum(val_acc) / len(val_acc)

        print("Min_sample_leaf", min_leaf)
        print("Training accuracy: ", avg_train_acc * 100, "%")
        print("Validation accuracy: ", avg_val_acc * 100, "%")

        train_acc_min_leafs.append(avg_train_acc)
        val_acc_min_leafs.append(avg_val_acc)

    #fine tune max_features
    max_features = np.arange(20,401,20)

    train_acc_max_features = []
    val_acc_max_features = []
    #
    for max_feature in max_features:
        train_acc = []
        val_acc = []
        for train_index, val_index in kf.split(train_val_X):
            ##########################
            train_X = train_val_X[train_index, :]
            val_X = train_val_X[val_index, :]

            train_y = train_val_y[train_index]
            val_y = train_val_y[val_index]

            dtc = tree.DecisionTreeClassifier(max_features=max_feature)
            dtc.fit(train_X, train_y)
            train_acc.append(dtc.score(train_X, train_y))
            val_acc.append(dtc.score(val_X, val_y))
            ##########################
        avg_train_acc = sum(train_acc) / len(train_acc)
        avg_val_acc = sum(val_acc) / len(val_acc)

        print("Max_features", max_feature)
        print("Training accuracy: ", avg_train_acc * 100, "%")
        print("Validation accuracy: ", avg_val_acc * 100, "%")

        train_acc_max_features.append(avg_train_acc)
        val_acc_max_features.append(avg_val_acc)

    return train_acc_criterion, val_acc_criterion, train_acc_min_leafs, val_acc_min_leafs, min_leafs, train_acc_max_features, val_acc_max_features, max_features

train_acc_criterion, val_acc_criterion, train_acc_min_leafs, val_acc_min_leafs, min_leafs, train_acc_max_features, val_acc_max_features, max_features = parameter_tune_2b(TFIDF, category)

# plot training/validation curves
plt.figure(2)
plt.bar(x_figure1_1, train_acc_criterion, width=bar_width, label="Training accuracy")
plt.bar(x_figure1_2, val_acc_criterion, width=bar_width, label="Validation accuracy")
plt.xlabel('Criterion')
plt.ylabel('Accuracy')
plt.xticks(x_figure1_3,criterion)
plt.legend()
plt.figure(3)
plt.plot(min_leafs, train_acc_min_leafs, marker='.', label="Training accuracy")
plt.plot(min_leafs, val_acc_min_leafs, marker='.', label="Validation accuracy")
plt.xlabel('min_simple_leafs')
plt.ylabel('Accuracy')
plt.legend()
plt.figure(4)
plt.plot(max_features, train_acc_max_features, marker='.', label="Training accuracy")
plt.plot(max_features, val_acc_max_features, marker='.', label="Validation accuracy")
plt.xlabel('max_features')
plt.ylabel('Accuracy')
plt.legend()

###Random Forest Q3
def parameter_tune_3(train_val_X, train_val_y):
    num_trees = np.arange(10,300,10)
    train_acc_forest = []
    val_acc_forest = []
    train_std_forest = []
    val_std_forest = []

    for num_tree in num_trees:
        train_acc = []
        val_acc = []
        for train_index, val_index in kf.split(train_val_X):
            ##########################
            train_X = train_val_X[train_index, :]
            val_X = train_val_X[val_index, :]

            train_val_y = np.array(train_val_y)
            train_y = train_val_y[train_index]
            val_y = train_val_y[val_index]

            dtc = RandomForestClassifier(n_estimators=num_tree)
            dtc.fit(train_X, train_y)
            train_acc.append(dtc.score(train_X, train_y))
            val_acc.append(dtc.score(val_X, val_y))
            ##########################

        avg_train_acc = sum(train_acc) / len(train_acc)
        avg_val_acc = sum(val_acc) / len(val_acc)

        std_train = np.std(train_acc)
        std_val = np.std(val_acc)

        print("Number of trees", num_tree)
        print("Training accuracy: ", avg_train_acc * 100, "%")
        print("Validation accuracy: ", avg_val_acc * 100, "%")

        train_acc_forest.append(avg_train_acc)
        val_acc_forest.append(avg_val_acc)
        train_std_forest.append(std_train)
        val_std_forest.append(std_val)

    return train_acc_forest, val_acc_forest, train_std_forest, val_std_forest, num_trees

train_acc_forest, val_acc_forest, train_std_forest, val_std_forest, num_trees = parameter_tune_3(TFIDF, category)

# plot training/validation curves
plt.figure(5)
plt.subplot(2,1,1)
plt.plot(num_trees, train_acc_forest, marker='.', label="Training accuracy")
plt.plot(num_trees, val_acc_forest, marker='.', label="Validation accuracy")
plt.xlabel('Number of trees')
plt.ylabel('Accuracy')
plt.legend()
plt.subplot(2,1,2)
plt.plot(num_trees, train_std_forest, marker='.', label="Training standard deviation")
plt.plot(num_trees, val_std_forest, marker='.', label="Validation standard deviation")
plt.xlabel('Number of trees')
plt.ylabel('Standard Deviation')
plt.legend()

#Form a table
num_trees = num_trees.tolist()
Table_Q3 = PrettyTable()
Table_Q3_title = num_trees.copy()
Table_Q3_title.insert(0,'type/number of trees')
Table_Q3.field_names = Table_Q3_title
Table_Q3_train_acc_forest = train_acc_forest.copy()
Table_Q3_train_acc_forest.insert(0,"train_acc_forest")
Table_Q3.add_row(Table_Q3_train_acc_forest)
Table_Q3_val_acc_forest = val_acc_forest.copy()
Table_Q3_val_acc_forest.insert(0,"val_acc_forest")
Table_Q3.add_row(Table_Q3_val_acc_forest)
Table_Q3_train_std_forest = train_std_forest.copy()
Table_Q3_train_std_forest.insert(0,"train_std_forest")
Table_Q3.add_row(Table_Q3_train_std_forest)
Table_Q3_val_std_forest = val_std_forest.copy()
Table_Q3_val_std_forest.insert(0,"val_std_forest")
Table_Q3.add_row(Table_Q3_val_std_forest)

print(Table_Q3)

# Q4 XGboost
def parameter_tune_4(train_val_X, train_val_y):
    etas = np.arange(0.1,1,0.1)
    train_acc_xgboost = []
    val_acc_xgboost = []
    train_std_xgboost = []
    val_std_xgboost = []

    for eta in etas:
        train_acc = []
        val_acc = []
        for train_index, val_index in kf.split(train_val_X):
            ##########################
            train_X = train_val_X[train_index, :]
            val_X = train_val_X[val_index, :]

            train_val_y = np.array(train_val_y)
            train_y = train_val_y[train_index]
            val_y = train_val_y[val_index]

            dtc = xgb.XGBClassifier(eta=eta)
            dtc.fit(train_X, train_y)
            train_acc.append(dtc.score(train_X, train_y))
            val_acc.append(dtc.score(val_X, val_y))
            ##########################

        avg_train_acc = sum(train_acc) / len(train_acc)
        avg_val_acc = sum(val_acc) / len(val_acc)

        std_train = np.std(train_acc)
        std_val = np.std(val_acc)

        print("Number of eta", eta)
        print("Training accuracy: ", avg_train_acc * 100, "%")
        print("Validation accuracy: ", avg_val_acc * 100, "%")

        train_acc_xgboost.append(avg_train_acc)
        val_acc_xgboost.append(avg_val_acc)
        train_std_xgboost.append(std_train)
        val_std_xgboost.append(std_val)

    return train_acc_xgboost, val_acc_xgboost,train_std_xgboost, val_std_xgboost, etas

train_acc_xgboost, val_acc_xgboost, train_std_xgboost, val_std_xgboost, etas = parameter_tune_4(TFIDF, category)

# plot training/validation curves
plt.figure(6)
plt.subplot(2,1,1)
plt.plot(etas, train_acc_xgboost, marker='.', label="Training accuracy")
plt.plot(etas, val_acc_xgboost, marker='.', label="Validation accuracy")
plt.xlabel('Number of eta')
plt.ylabel('Accuracy')
plt.legend()
plt.subplot(2,1,2)
plt.plot(etas, train_std_xgboost, marker='.', label="Training standard deviation")
plt.plot(etas, val_std_xgboost, marker='.', label="Validation standard deviation")
plt.xlabel('Number of etas')
plt.ylabel('Standard Deviation')
plt.legend()

#Form a table
etas = etas.tolist()
Table_Q4 = PrettyTable()
Table_Q4_title = etas.copy()
Table_Q4_title.insert(0,'type/learning rate')
Table_Q4.field_names = Table_Q4_title
Table_Q4_train_acc_xgboost = train_acc_xgboost.copy()
Table_Q4_train_acc_xgboost.insert(0,"train_acc_xgboost")
Table_Q4.add_row(Table_Q4_train_acc_xgboost)
Table_Q4_val_acc_xgboost = val_acc_xgboost.copy()
Table_Q4_val_acc_xgboost.insert(0,"val_acc_xgboost")
Table_Q4.add_row(Table_Q4_val_acc_xgboost)
Table_Q4_train_std_xgboost = train_std_xgboost.copy()
Table_Q4_train_std_xgboost.insert(0,"train_std_xgboost")
Table_Q4.add_row(Table_Q4_train_std_xgboost)
Table_Q4_val_std_xgboost = val_std_xgboost.copy()
Table_Q4_val_std_xgboost.insert(0,"val_std_xgboost")
Table_Q4.add_row(Table_Q4_val_std_xgboost)

print(Table_Q4)

#Question5
#Compare random forest and xgboost
best_number_trees = num_trees[np.argmax(val_acc_forest)]
best_learning_rate = etas[np.argmax(val_acc_xgboost)]

if val_acc_forest[np.argmax(val_acc_forest)] >= val_acc_xgboost[np.argmax(val_acc_xgboost)]:
    print("Select random forest model")
    print("best num of trees =", best_number_trees)

else:
    print("Select xgboost model")
    print("best learning rate =", best_learning_rate)

#Select and train a random forest model using all data
def parameter_tune_5(train_val_X, train_val_y, best_number_trees):
    num_tree = best_number_trees
    train_acc = []

    dtc = RandomForestClassifier(n_estimators=num_tree)
    dtc.fit(train_val_X, train_val_y)
    train_acc = dtc.score(train_val_X, train_val_y)

    print("Number of trees", num_tree)
    print("Training accuracy: ", train_acc * 100, "%")

    return train_acc, dtc

train_acc_best, Model_dtc = parameter_tune_5(TFIDF, category, best_number_trees)

#compute TFIDF for raw test data
len_test = len(D)
E = []
for i in range(len_test):
    E.append(get_tokens(D[i][1]))

#Compute fij for raw test data
f_test = np.empty((len_test,1000))
f_test[:] = np.nan
for i in range(len_test):
    for j in range(1000):
        f_test[i,j] = E[i].count(C[j])

#Compute max k
maxk_test = f_test.max(axis=1)

#Compute TF matrix
TF_test = f_test/maxk_test[:,None]

#Compute IDF
temp_test = f_test.copy()
temp_test[temp_test != 0] = 1
m_test = np.sum(temp_test, axis=0)
IDF_test = np.log(len_test/m_test)
IDF_test = IDF_test[None,:]

#Compute TFIDF as features
TFIDF_test = TF_test * IDF_test

###compute test result using trained random forest model
Test_result = Model_dtc.predict(TFIDF_test)

#category
Test_result_name = []
for i in range(len_test):
    if Test_result[i] == 0:
        Test_result_name.append('sport')
    elif Test_result[i] == 1:
        Test_result_name.append('business')
    elif Test_result[i] == 2:
        Test_result_name.append('politics')
    elif Test_result[i] == 3:
        Test_result_name.append('entertainment')
    else:
        Test_result_name.append("tech")

index_test = np.arange(1,len_test+1,1)

index_test = index_test.tolist()

Final_predict = [str(a)+','+str(b) for a,b in zip(index_test,Test_result_name)]

#Write my prediction into CSV file
with open("labels.csv", 'w', newline='') as file:
    writer = csv.writer(file)
    for i in range(len_test):
        writer.writerow([Final_predict[i]])

plt.show()
print()

