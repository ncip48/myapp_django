from django.shortcuts import render
import pandas as pd
import datetime
import logging
import math

from rest_framework.decorators import api_view, renderer_classes
from rest_framework.renderers import JSONRenderer
from rest_framework.response import Response

# import class Task dari file todo/models.py
from .models import StudentStress

#anu
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import mean_absolute_error

def search_id(data):
    # Assuming `YourModel` is the Django model representing your data
    data_map = {
        'anxiety_level': data[0],
        'self_esteem': data[1],
        'mental_health_history': data[2],
        'depression': data[3],
        'headache': data[4],
        'blood_pressure': data[5],
        'sleep_quality': data[6],
        'breathing_problem': data[7],
        'noise_level': data[8],
        'living_conditions': data[9],
        'safety': data[10],
        'basic_needs': data[11],
        'academic_performance': data[12],
        'study_load': data[13],
        'teacher_student_relationship': data[14],
        'future_career_concerns': data[15],
        'social_support': data[16],
        'peer_pressure': data[17],
        'extracurricular_activities': data[18],
        'bullying': data[19],
    }

    # Assuming `Datas` is the Django model representing your data table
    # Replace `YourModel` with the actual name of your Django model
    find = StudentStress.objects.filter(**data_map).first()
    if find:
        return find.id
    else:
        return None

# Membuat View untuk halaman daftar task
def index_dataset(request):
    # Mengambil semua data task
    predictions = StudentStress.objects.all()
    context = {
        'datas': predictions
    }
    # memparsing data task ke template todo/index.html dan merender nya
    return render(request, 'dataset/index.html', context)

def predict(X_manual=None):
    df_net = pd.DataFrame(list(StudentStress.objects.all().values()))
    df_net.drop(columns=['created_at', 'updated_at', 'id'], inplace=True)
    X = df_net.iloc[:, :-1].values
    y = df_net.iloc[:, -1].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=len(X)-len(X_manual), random_state=13) if X_manual is not None else train_test_split(X, y, test_size=0.3, random_state=13)
    data_id = [search_id(data_point) for data_point in X_test] if X_manual is None else []
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_manual) if X_manual is not None else sc.transform(X_test)
    classifier = DecisionTreeClassifier(criterion='entropy', random_state=0)
    classifier.fit(X_train, y_train)
    print(X_test.shape)
    return classifier, X_test, data_id, y_test, y_train, X_train

def index_prediksi(request):
    p = predict()
    classifier = p[0]
    X_test = p[1]
    data_id = p[2]
    y_test = p[3]
    y_train = p[4]
    
    y_pred = classifier.predict(X_test)
    mae = mean_absolute_error(y_pred,y_test)
    predictions = []
    for idx, pred in enumerate(y_pred):
        prediction = StudentStress.objects.get(id=data_id[idx])
        prediction.prediction = pred
        predictions.append(prediction)
    predictions.sort(key=lambda x: x.id)
    matchCount = 0
    mismatchCount = 0
    totalCount = 0
    for idx, true_label in enumerate(y_test):
        if true_label == y_pred[idx]:
            matchCount += 1
        else:
            mismatchCount += 1
    totalCount = len(y_test)
    entropy = 0
    for label in np.unique(y_train):
        p = len(y_train[y_train == label]) / len(y_train)
        entropy -= p * np.log2(p)
    entropy = round(entropy, 4)
    accuracy = round(accuracy_score(y_test, y_pred) * 100, 2)
    error = round(100-accuracy, 2)
    print(f'Classification Report: \n{classification_report(y_test, y_pred)}')
    context = {
        'accuracy': accuracy,
        'error': error,
        'datas': predictions,
        'matchCount': matchCount,
        'totalCount': totalCount,
        'mismatchCount': mismatchCount,
        'entropy': entropy,
        'mae': mae
    }
    return render(request, 'prediksi/index.html', context)

@api_view(['POST'])
# @renderer_classes((JSONRenderer))
def api_predict(request):
    manual = [[request.data.get('anxiety_level'),27,0,3,1,2,5,1,2,4,5,4,4,2,4,1,3,2,2,1]]
    p = predict(manual)
    classifier = p[0]
    X_test = p[1]
    
    y_pred = classifier.predict(X_test)
    context = {
        'prediction': y_pred[0]
    }
    return Response(context)