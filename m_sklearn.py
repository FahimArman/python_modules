# datasets
from sklearn.datasets import load_digits
digits = load_digits()
print(dir(digits))

from sklearn.datasets import load_iris
iris = load_iris()

print(dir(iris))
print(list(iris.feature_names))
print(iris.data)
print(iris.DESCR)
print(iris.target)
print(list(iris.target_names))

#fillna
df.column_name=df.column_name.fillna(median_values)

#LabelEncoder() // 0,1,2,...
from sklearn,preprocessing import LabelEncoder
le = LabelEncoder()

df.column_name = df.fit_transform(df.column_name)
X = df[['column_name1','column_name2']].values
y = df.column_name


# OneHotEncoder() // 1,1,1
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
ct=ColumnTransformer([('column_name',OneHotEncoder(),[0])],remainder='passthrough')
X = ct.fit_transform(X)


#MinMaxScaler
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(df[['column_name']])
df['column_name'] = scaler.transform(df['column_name'])

#CountVectorizer
from sklearn.feature_extraction.text import CountVectorizer
v = CountVectorizer()
X_train_count = v.fit_transform(X_train.values)
print(X_train_count.toarray()[:3])


# Train_Test_Split
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y, test_size = 0.2,random_state=10)

# length of the train, test sample
print(len(X_train))
print(len(X_test))

#cross validation
from sklearn.model_selection import cross_val_score
print(cross_val_score(model_name(),X,y,cv=3))


#LinearRegression()
from sklearn.linear_model import LinearRegression

#LogisticRegression()
from sklearn.linear_model import LogisticRegression

#DecisionTree
from sklearn.tree import DecisionTreeClassifier

#svm
from sklearn.svm import SVC
model = SVC()
model = SVC(C=10,gamma = 10, kernerl='linear')

#RandomForest
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier()
model = RandomForestClassifier(n_estimators = 20) #// 20 trees

#GaussianNaiveBayes
from sklearn.naive_bayes import GaussianNB

#MultinomialNB
from sklearn.naive_bayes import MultinomialNB


#KMeans
from sklearn.cluster import KMeans
km = KMeans(n_clusters=3)
y_predicted = km.fit_predict(df['column1','column2'])
df['new_column_name'] = y_predicted


#pipeline
from sklearn.pipeline import Pipeline
clf = Pipeline([
		('vectorizer', CountVectorizer()),
		('vb', MultinomialNB())
])


# training and testing the model
model = model_name()
model.fit(X_train,y_train) // training
model.predict(X_test)
model.score(X_test,y_test)


# LinearRegression coef_
model.coef_
model.intercept_

# LogisticRegression (probability)
model.predict_proba(X_test)


#KMeans
model.inertia_


#confusion_matrix
y_predicted = model.predict(X_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_predicted)
print(cm)





#joblib
from sklearn.external import joblib

#save
joblib.dump(model,'model_save_name')

#load
mj = joblib.load('model_file_name')

# use 
mj.predict([[5000]]) # LinearRegression()




#GridSearchCV, RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
