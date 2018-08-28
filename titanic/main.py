import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import accuracy_score
from sklearn.cross_validation import KFold

data_train = pd.read_csv('./all/train.csv')
data_test = pd.read_csv('./all/test.csv')

#function to group ages
def group_ages(passenger):
  passenger.Age = passenger.Age.fillna(-0.5) #fill NA/NaN values with -0.5
  #sort data into bins
  bins = (-1, 0, 5, 12, 18, 25, 35, 60, 120)
  group_names = ['Unknown', 'Baby', 'Child', 'Teenager', 'Student', 'Young Adult', 'Adult', 'Senior']
  categories = pd.cut(passenger.Age, bins, labels=group_names)
  passenger.Age = categories 
  return passenger

#group cabins by the letter since it represents location, slice of trailing numbers
def local_cabin(passenger):
  passenger.Cabin = passenger.Cabin.fillna('N')
  passenger.Cabin = passenger.Cabin.apply(lambda x: x[0])
  return passenger

#classify the fares
def group_fares(passenger):
  passenger.Fare = passenger.Fare.fillna(-0.5)
  bins = (-1, 0, 8, 15, 31, 1000)
  group_names = ['Unknown', '1_quartile', '2_quartile', '3_quartile', '4_quartile']
  categories = pd.cut(passenger.Fare, bins, labels=group_names)
  passenger.Fare = categories
  return passenger

def format_name(passenger):
  passenger['Lname'] = passenger.Name.apply(lambda x: x.split(' ')[0])
  passenger['NamePrefix'] = passenger.Name.apply(lambda x: x.split(' ')[1])
  return passenger

def drop_features(passenger):
    return passenger.drop(['Ticket', 'Name', 'Embarked'], axis=1)

def apply_features(passenger):
  passenger = group_ages(passenger)
  passenger = local_cabin(passenger)
  passenger = group_fares(passenger)
  passenger = format_name(passenger)
  passenger = drop_features(passenger)
  return passenger

data_test = apply_features(data_test)
data_train = apply_features(data_train)
#print(data_train.head())

#normalize labels convert each unique string to a number
def encode_features(df_train, df_test):
    features = ['Fare', 'Cabin', 'Age', 'Sex', 'Lname', 'NamePrefix']
    df_combined = pd.concat([df_train[features], df_test[features]])
    
    for feature in features:
        le = preprocessing.LabelEncoder()
        le = le.fit(df_combined[feature])
        df_train[feature] = le.transform(df_train[feature])
        df_test[feature] = le.transform(df_test[feature])
    return df_train, df_test
    
data_train, data_test = encode_features(data_train, data_test)
#print(data_train.head())

#split training data x will be all inputs except survived
#y will be the prediction which will be survived
x = data_train.drop(['Survived','PassengerId'],axis=1)
y = data_train['Survived']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=41)


#Use nonlinear SVM Classification
clf = Pipeline([
  ("poly_features", PolynomialFeatures(degree=3)),
  ("scaler",StandardScaler()),
  ("svm_clf",LinearSVC(C=10, loss="hinge"))
])

clf.fit(x_train,y_train)
predictions = clf.predict(x_test)
print(accuracy_score(y_test, predictions))

def run_kfold(clf):
  kf = KFold(891, n_folds=10)
  outcomes=[]
  fold =0
  for train_index, test_index in kf:
    fold += 1
    x_train, x_test = x.values[train_index], x.values[test_index]
    y_train, y_test = y.values[train_index], y.values[test_index]
    clf.fit(x_train, y_train)
    predictions = clf.predict(x_test)
    accuracy = accuracy_score(y_test, predictions)
    outcomes.append(accuracy)
    print("Fold {0} accuracy: {1}".format(fold, accuracy))     
  mean_outcome = np.mean(outcomes)
  print("Mean Accuracy: {0}".format(mean_outcome)) 

run_kfold(clf)

# ids = data_test['PassengerId']
# predictions = clf.predict(data_test.drop('PassengerId',axis=1))
# output = pd.DataFrame({ 'PassengerId' : ids, 'Survived': predictions })
# output.to_csv('titanic-predictions.csv', index = False)