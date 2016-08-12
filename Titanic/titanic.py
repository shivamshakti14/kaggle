# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

import pandas as pd
import numpy as np
train = pd.read_csv('train.csv')
#print train
test = pd.read_csv('test.csv')
#print test
#print train.describe
#print train.shape
#print test.describe()
#print test.shape
#print train['Survived'].value_counts() # value_counts(normalize=true) for proportions
#print train['Survived'][train['Sex']=='male'].value_counts()
#print train['Survived'][train['Sex']=='female'].value_counts()
train['Child']=float('NaN')
train['Child'][train['Age']<18]=1
train['Child'][train['Age']>=18]=0
train['Age'] = train['Age'].fillna(train['Age'].median())  ## fill the missing values
test['Age'] = test['Age'].fillna(test['Age'].median())
#print train['Child']
from sklearn import tree
train['Sex'][train['Sex']=='male']=0
train['Sex'][train['Sex']=='female']=1
train['Embarked'][train['Embarked']=='S']=0
train['Embarked'][train['Embarked']=='C']=1
train['Embarked'][train['Embarked']=='Q']=2


test['Sex'][test['Sex']=='male']=0
test['Sex'][test['Sex']=='female']=1
test['Embarked'][test['Embarked']=='S']=0
test['Embarked'][test['Embarked']=='C']=1
test['Embarked'][test['Embarked']=='Q']=2

target = train['Survived'].values

features_one = train[['Pclass','Age','Sex','Fare']].values
my_tree_one = tree.DecisionTreeClassifier()
my_tree_one = my_tree_one.fit(features_one, target)
print my_tree_one.feature_importances_    # more value more important
print my_tree_one.score(features_one, target)


test['Fare'] = test['Fare'].fillna(test['Fare'].median())
test_features = test[['Pclass','Age','Sex','Fare']].values
myprediction = my_tree_one.predict(test_features)
#print myprediction
PassengerId = np.array(test['PassengerId']).astype(int)
my_solution = pd.DataFrame(myprediction, PassengerId, columns= ['Survived'])
#print my_solution
#print my_solution.shape
my_solution.to_csv("my_solution_one.csv", index_label = ["PassengerId"])

## Overfitting and control
train['Embarked'] = train['Embarked'].fillna(train['Embarked'].median())
#train['Child'] = train['Child'].fillna(train['Child'].median())
features_two = train[["Pclass","Age","Sex","Fare", 'SibSp', 'Parch', 'Embarked']].values

my_tree_two = tree.DecisionTreeClassifier(max_depth= 10,min_samples_split=5, random_state=1)
my_tree_two = my_tree_two.fit(features_two, target)
print my_tree_two.feature_importances_
print my_tree_two.score(features_two, target)


## Random Forest
from sklearn.ensemble import RandomForestClassifier
features_forest = train[["Pclass","Age","Sex","Fare", 'SibSp', 'Parch', 'Embarked']].values
forest = RandomForestClassifier(max_depth = 10, min_samples_split=2, n_estimators=100, random_state=1)
my_forest = forest.fit(features_forest, target)
print my_forest.score(features_forest, target)
test_features = test[["Pclass", "Age", "Sex", "Fare", "SibSp", "Parch", "Embarked"]].values
predict_forest = my_forest.predict(test_features)
print pre

# <codecell>


# <codecell>


# <codecell>


