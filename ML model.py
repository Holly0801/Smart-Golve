from pandas import DataFrame, read_csv
import pandas as pd 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split #将数据集划分为训练集/测试集
import numpy as np
#import serial #串口通信
#import pyttsx3 #文本转语音库
from sklearn.metrics import accuracy_score #表示分类模型的准确率
from sklearn import svm
from sklearn.metrics import r2_score #度量模型预测效果
from sklearn.utils import joblib


df = pd.read_csv('dataset.csv')
X=df[['THUMB','INDEX','MIDDLE','RING','LITTLE']]
Y=df[['LABEL']]


#ser = serial.Serial('COM5', baudrate = 9600, timeout =1) #串口程序
#for i in range (3):
#    arduinodata=ser.readline().strip()
#values = arduinodata.decode('ascii').split(',')
#a, b, c, d, e = [int(s) for s in values]
#example = np.array([a, b, c, d, e])
#example = example.reshape(1, -1)


        # KNN
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.1, random_state = 60)
model = KNeighborsClassifier()
model.fit(x_train,y_train)
joblib.dump(model,'KNN') #序列化并保存到文件中
mj7 = joblib.load('KNN') #从文件中加载保存的对象（反序列化）
#print(mj7.predict(example))
predicted=model.predict(x_test)

accuracy_knn = accuracy_score(y_test,predicted)
print("Accuracy for KNN is",accuracy_knn)


        # SVM 
x_train1, x_test1, y_train1, y_test1 = train_test_split(X, Y, test_size=0.3, random_state = 60)
svc = svm.SVC(kernel='linear').fit(x_train,y_train)
joblib.dump(svc,'SVM')
mj2 = joblib.load('SVM')
#print(mj2.predict(example))
predicted1 = svc.predict(x_test)

accuracy_svm = accuracy_score(y_test,predicted1)
print("Accuracy for SVM is",accuracy_svm)




