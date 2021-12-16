import cvxpy as cp
import numpy as np
import pandas as pd

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import minmax_scale
from sklearn import linear_model

# network attack data processing
print('read data and preprocessing')
# df1 = pd.read_csv('ack.csv')
# df2 = pd.read_csv('benign_traffic.csv')
# benign_data = df1.to_numpy()
# benign_data = minmax_scale(benign_data,axis=0)
# malicious_data = df2.to_numpy()
# malicious_data = minmax_scale(malicious_data,axis=0)
benign_data = np.load('train_ben.npy')
print(benign_data.shape)
malicious_data = np.load('test.npy')
print(malicious_data.shape)
labels_benign = np.zeros((benign_data.shape[0],1))
labels_malicious = np.ones((malicious_data.shape[0],1))
benign_data= np.concatenate((benign_data,labels_benign),axis=1)
malicious_data = np.concatenate((malicious_data,labels_malicious),axis=1)
data = np.concatenate((benign_data,malicious_data),axis=0)
print('preprocessing over')
# benign_data = data[0:84,0:data.shape[1]-1]
# benign_train, benign_test = train_test_split(benign_data,test_size=0.20, random_state=42)
# malicious_data = data[84:data.shape[0],0:data.shape[1]-1]
# malicious_train, malicious_test = train_test_split(malicious_data,test_size=0.20, random_state=42)

# df = pd.read_excel('divorce.xlsx')
print('data shuffling')
np.random.shuffle(data)
X = data[:,0:data.shape[1]-1]
# X = data[:,0:2]
y = data[:,data.shape[1]-1:data.shape[1]]
print('data spliting')
data_train, data_test, labels_train, labels_test = train_test_split(X, y, test_size=0.9, random_state=42)
# X = np.array([[-1, -0.5], [-2, -1], [1, 1], [2, 1]])
# y = np.array([0, 0, 1, 1])
print('model training')
# more simple model
# reg = linear_model.LinearRegression()
# reg.fit(data_train,labels_train)
# A = reg.coef_
# b = reg.intercept_
# print(reg.predict(data_test))

# SVC model
svm = SVC(kernel='linear')
svm.fit(data_train,labels_train)
A = svm.coef_
b = svm.intercept_


#----------------------------------

print("test without augmentation")
# predict
# result = svm.predict(data_test)
# predict
# score1 = reg.score(data_test,labels_test)
# print(score1)
# correct = 0
# result = reg.predict(data_test)
# for i in range(data_test.shape[0]):
#     if(result[i]>=0.5):
#         result[i]=1
#     else:
#         result[i] = 0
#     if(result[i]==labels_test[i]):
#         correct += 1
# print(correct/data_test.shape[0])
score1 = svm.score(data_test,labels_test)
print(score1)
# scalar
# x = cp.Variable()
# Vector
# x = cp.Variable(X.shape[1])
# matrix 
# A = cp.Variable((5,1))
V = []
data_generated = []
label_gen = []
augmented_size = malicious_data.shape[0]
augmented_size = 1000
print('augmented size:',augmented_size)
for i in range(augmented_size):
    x = cp.Variable(X.shape[1])
    objective = cp.Minimize(cp.norm(x-data_train[i]))
    if labels_train[i]==0:
        constraints = [cp.sum(cp.multiply(np.squeeze(A),x))+b>=0.6]
        g = 0
    else:
        constraints = [cp.sum(cp.multiply(np.squeeze(A),x))+b<=0.4]
        g = 1
    prob = cp.Problem(objective, constraints)
    v = prob.solve()
    if(prob.status!="infeasible" and prob.status!="unbounded"):   
        data_generated.append(x.value)
        label_gen.append(g)
        print("Optimal value",v)
        V.append(v)
data_generated = np.asarray(data_generated)
print(data_generated.shape)
label_gen = np.asarray(label_gen)
V = np.asarray(V)
data_gen_accept = []
label_accept = []
max_ = np.max(V)
min_ = np.min(V)
average = (max_+min_)/2
print("V.shape",V.shape)
for i in range(V.shape[0]):
    # if V[i]<=average:
    data_gen_accept.append(data_generated[i])
    label_accept.append(label_gen[i])
new_train_gen = np.asarray(data_gen_accept)
new_label_gen = np.asarray(label_accept)
new_label_gen = np.reshape(new_label_gen,(new_label_gen.shape[0],1))
new_train = np.concatenate((data_train,new_train_gen),axis=0)
new_label = np.concatenate((labels_train,new_label_gen),axis=0)
# svm = SVC(kernel='linear')
# svm.fit(new_train,new_label)
reg = linear_model.LinearRegression()
reg.fit(new_train,new_label)
# print("new_coffecient",reg.coef_)
# print("new intercept",reg.intercept_)
# print("old_coffecient",A)
# print("old_intercept",b)
print("augmented score:",reg.score(data_test,labels_test))
print("old score:",score1)
print("original data:",data_train.shape)
print("original test:",data_test.shape)
print("new data:",new_train.shape)