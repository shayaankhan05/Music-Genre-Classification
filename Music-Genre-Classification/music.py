from sklearn.svm import SVC
from Metadata import getmetadata
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from flask import Flask, render_template, request
import os
import pandas as pd
import numpy as np

app = Flask(__name__)

def predict_genre(file):

    # Read csv file and convert into pandas dataframes
    df = pd.read_csv('data.csv')
    df = df.drop(['beats'], axis=1)

    # Change the dtype for preprocessing
    df['class_name'] = df['class_name'].astype('category')
    df['class_label'] = df['class_name'].cat.codes

    # Create a dictionary to get genre name
    lookup_genre_name = dict(zip(df.class_label.unique(), df.class_name.unique()))

    # Splitting the data

    # Features of matrix
    X = df.iloc[:, 1:28]
    # Dependent variable vector (output)
    y = df['class_label']

    # Split Training set and testing set
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=3)

    # Feature scaling using min-max normalization
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train the model using KNN algorithm
    knn = KNeighborsClassifier(n_neighbors=13)
    knn.fit(X_train_scaled, y_train)
    knn.score(X_test_scaled, y_test)

    # Extract metadata from the wav file
    metadata = getmetadata(file)

    # Transform the data 
    np_metadata = np.array(metadata)
    transformed_metadata = scaler.transform([np_metadata])

    # Predicts the genre key
    genre_prediction = knn.predict(transformed_metadata)

    # print(lookup_genre_name[genre_prediction[0]])
    return lookup_genre_name[genre_prediction[0]]

   
@app.route('/',  methods=['POST', 'GET'])
def index():
    if request.method == "GET":
        return render_template('index.html', genre="")

    elif request.method == "POST":
        
        file = request.files['audio']
         
        genre = predict_genre(file)

        # print(genre)
        
        return render_template('index.html', genre=genre)

if __name__ == '__main__':
    app.run(debug=True)




