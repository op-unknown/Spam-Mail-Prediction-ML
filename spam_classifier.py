import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

data = pd.read_csv("/content/spam.csv")

data['label'] = data['label'].map({'ham':0, 'spam':1})

X = data['message']
y = data['label']

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = MultinomialNB()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))

msg = input("Enter message: ")
msg_vec = vectorizer.transform([msg])
result = model.predict(msg_vec)

if result[0]==1:
    print("Spam Mail")
else:
    print("Not Spam Mail")