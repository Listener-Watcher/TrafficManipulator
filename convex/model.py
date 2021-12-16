import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt
import pickle as pkl
import argparse

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import minmax_scale
from sklearn import linear_model

def RunKN(A,b,data):
    RMSE = []
    for i,x in enumerate(data):
        rmse = np.dot(A,x)+b
        RMSE.append(rmse)
        # print(rmse)
        # print('prcoessing packet number',i)
    RMSE = np.array(RMSE)
    return RMSE


def test_mut(mut_feat,model_save_path):

    with open(model_save_path, "rb") as f:
            A = pkl.load(f)
            b = pkl.load(f)
            _ = pkl.load(f)
    rmse = RunKN(A,b, mut_feat)
    return rmse




if __name__ == "__main__":
    parse = argparse.ArgumentParser()

    parse.add_argument('-M', '--mode', type=str, default='exec', help="{train,exec,gen,enhance}")
    parse.add_argument('-rf', '--RMSE_file_path', type=str,
                       help="resulting rmse file (.pkl) path, only for execute mode!")
    parse.add_argument('-mf', '--model_file_path', type=str, default='./convex_programming/model.pkl',
                       help="for train mode, model is saved into 'mf'; for execute mode, model is loaded from 'mf'")
    arg = parse.parse_args()

    if arg.mode == 'train':
        print("Warning: under TRAIN mode!")
        print('read data and preprocessing')
        benign_data = np.load('train_ben.npy')
        print(benign_data.shape)
        malicious_data = np.load('test.npy')
        print(malicious_data.shape)
        labels_benign = np.zeros((benign_data.shape[0],1))
        labels_malicious = np.ones((malicious_data.shape[0],1))
        benign_data_= np.concatenate((benign_data,labels_benign),axis=1)
        malicious_data_ = np.concatenate((malicious_data,labels_malicious),axis=1)
        data = np.concatenate((benign_data_,malicious_data_),axis=0)
        print('preprocessing over')
        print('data shuffling')
        np.random.shuffle(data)
        X = data[:,0:data.shape[1]-1]
        y = data[:,data.shape[1]-1:data.shape[1]]
        print('data spliting')
        data_train, data_test, labels_train, labels_test = train_test_split(X, y, test_size=0.99, random_state=42)
        # np.save('split_test_set_svm.npy',data_test)
        # np.save('split_test_label_set_svm.npy',labels_test)
        # np.save('split_train_set_svm.npy',data_train)
        # np.save('split_train_label_set_svm.npy',labels_train)
        print('model training')
        # more simple model
        # reg = linear_model.LinearRegression()
        # reg.fit(data_train,labels_train)
        # A = reg.coef_
        # b = reg.intercept_
        svm = SVC(kernel='linear')
        # print(data_train.shape)
        # print(labels_train.shape)
        svm.fit(data_train,labels_train.reshape(labels_train.shape[0]))
        A = svm.coef_
        b = svm.intercept_
        print("svm score:",svm.score(data_test,labels_test))
        rmse = RunKN(A,b, data_test)

        x = np.arange(0,len(rmse),1)
        plt.scatter(x,rmse)
        plt.show()


        AD_threshold = 0.5
        print("AD_threshold:", AD_threshold)
        with open(arg.model_file_path, "ab") as f:
            pkl.dump(A,f)
            pkl.dump(b,f)
            pkl.dump(AD_threshold, f)

    elif arg.mode == 'exec':
        print("Warning: under EXECUTE mode!")
        #data_test = np.load('split_test_set_svm.npy')
        data_test = np.load('test.npy')
        #labels_test = np.load('split_test_label_set_svm.npy')

        # delete pcc-related features
        data_test[:, 33:50:4] = 0.
        data_test[:, 83:100:4] = 0.
        
        with open(arg.model_file_path, "rb") as f:
            A = pkl.load(f)
            b = pkl.load(f)
            AD_threshold = pkl.load(f)


        rmse = RunKN(A,b, data_test)
        with open(arg.RMSE_file_path, 'wb') as f:
            pkl.dump(rmse, f)

        print('AD_threshold:', AD_threshold)
        print('# rmse over AD_t:', rmse[rmse > AD_threshold].shape)
        print('Total number:', len(rmse))
        print("rmse mean:", np.mean(rmse))

        x = np.arange(0,len(rmse),1)
        plt.figure()
        plt.scatter(x,rmse,s=12, c='r')
        plt.plot(x,[AD_threshold]*len(rmse),c='black',linewidth=2,label="AD_threshold")
        plt.title("RMSE of Test set")
        plt.xlabel('pkt no.')
        plt.ylabel('RMSE in SVM')
        plt.legend()
        plt.show()
    elif arg.mode == 'gen':
        print("Warning: under generation mode!")
        malicious_data = np.load('test.npy')

        # delete pcc-related features
        # malicious_data[:, 33:50:4] = 0.
        # malicious_data[:, 83:100:4] = 0.
        
        with open(arg.model_file_path, "rb") as f:
            A = pkl.load(f)
            b = pkl.load(f)
            AD_threshold = pkl.load(f)
        V = []
        data_generated = []
        label_gen = []
        augmented_size = malicious_data.shape[0]
        augmented_size = 1000
        # np.random.shuffle(malicious_data)
        print('augmented size:',augmented_size)
        for i in range(augmented_size):
            x = cp.Variable(malicious_data.shape[1])
            objective = cp.Minimize(cp.norm((x-malicious_data[i]),'inf'))
            constraints = [cp.sum(cp.multiply(np.squeeze(A),x))+b<=0.4]
            prob = cp.Problem(objective, constraints)
            v = prob.solve()
            if(prob.status!="infeasible" and prob.status!="unbounded"):   
                data_generated.append(x.value)
                print("Optimal value",v)
                V.append(v)
        data_generated = np.asarray(data_generated)
        print("check mimic_set before saving:",data_generated.shape)
        np.save('mimic_set_SVM_convex_norm2.npy',data_generated)
    elif arg.mode =='enhance':
        malicious_data = np.load('mimic_set_SVM_convex_norm2.npy')
        train_malicious_data,_ = train_test_split(malicious_data,test_size=0.99,random_state=42)
        labels_malicious = np.ones((train_malicious_data.shape[0],1))
        malicious_data_ = np.concatenate((train_malicious_data,labels_malicious),axis=1)
        benign_data = np.load('split_train_set_svm.npy')
        labels = np.load('split_train_label_set_svm.npy')
        benign_data_ = np.concatenate((benign_data,labels),axis=1)
        data = np.concatenate((benign_data_,malicious_data_),axis=0)
        np.random.shuffle(data)
        X = data[:,0:data.shape[1]-1]
        y = data[:,data.shape[1]-1:data.shape[1]]
        print('data spliting')
        svm = SVC(kernel='linear')
        # data_train, data_test, labels_train, labels_test = train_test_split(X, y, test_size=0.99, random_state=42)
        svm.fit(X,y.reshape(y.shape[0]))
        data_test = np.load('split_test_set_svm.npy')
        labels_test = np.load('split_test_label_set_svm.npy')
        print('enhanced score',svm.score(data_test,labels_test))
    else:
        raise RuntimeError("argument -M is wrong! choose 'train' or 'execute'")
    

