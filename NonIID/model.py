import torch
import torch.nn as nn
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
import torch
import torch.nn as nn
import torch.optim as optim
import optuna
from sklearn.model_selection import train_test_split
import numpy as np
import collections
from collections import OrderedDict
class Net(nn.Module):
    def __init__(self, dropout_rate):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(28, 64)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(64, 128)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.fc3 = nn.Linear(128, 64)
        self.dropout3 = nn.Dropout(dropout_rate)
        self.fc4 = nn.Linear(64,8)
        self.params_key=[]
        params=self.state_dict()
        keys=[]
        for key,_ in params.items():
            keys.append(key)
        self.params_key=keys
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout1(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout2(x)
        x = torch.relu(self.fc3(x))
        x = self.dropout3(x)
        x = self.fc4(x)
        return x
    def Quick_evaluate(self, outputs,y_test):
        _, predicted = torch.max(outputs.data, 1)
        accuracy = (predicted == y_test.long()).sum().item() / y_test.size(0)
        criterion = nn.CrossEntropyLoss()
        loss = criterion(outputs, y_test)
        return loss, accuracy
    

    def evaluate(self, X_in, y_in):
        self.eval()
        X_test = torch.tensor(X_in.values,dtype=torch.float32)
        y_test = torch.tensor(y_in.values)

        with torch.no_grad():
            outputs = self(X_test)
            loss,accuracy=self.Quick_evaluate(outputs,y_test)
        return loss.item(), accuracy
    
    def fit(self, inputs, targets,learning_rate: float,  val_size: float , num_epochs=10):
        # Lịch sử huấn luyện
        optimizer=torch.optim.RMSprop(self.parameters(), lr=learning_rate)
        history = {'loss': [], 'accuracy': [], 'val_loss': [], 'val_accuracy': []}
        X_tr, X_va, y_tr, y_va = train_test_split(inputs, targets, test_size=val_size)
        X_train=torch.tensor(X_tr.values,dtype=torch.float32)
        y_train = torch.tensor(y_tr.values)
        #X_val = torch.tensor(X_va.values,dtype=torch.float32)
        #y_val = torch.tensor(y_va.values)
        for epoch in range(num_epochs):
            # Tính toán đầu ra mô hình
            outputs = self(X_train)
            loss,accuracy=self.Quick_evaluate(outputs,y_train)
            # Lưu thông tin mất mát vào lịch sử
            history['loss'].append(loss.item())
            history['accuracy'].append(accuracy)

            # Tính toán val_loss và val_accuracy (nếu có)
            if val_size!=0.00:
                val_loss,val_accuracy=self.evaluate(X_va,y_va)
                history['val_loss'].append(val_loss)
                history['val_accuracy'].append(val_accuracy)

            # Lan truyền ngược và cập nhật tham số mô hình
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        return history
    def get_parameter(self):
        params=self.state_dict()
        parameters=[]
        keys=[]
        for key,tensor in params.items():
            parameters.append(tensor)
            keys.append(key)
        self.params_key=keys
        return parameters
    
    def load_parameter(self, parameters_tensor):
        if isinstance(parameters_tensor, OrderedDict):
            self.load_state_dict(parameters_tensor)
        else:
            tensor=[]
            for par in parameters_tensor:
                tensor.append(torch.tensor(par))
            params = collections.OrderedDict(zip(self.params_key,tensor))
            self.load_state_dict(params)


    def get_weigthdivegence(self, par):
        t=float(0)
        m=float(0)
        param=self.get_parameter()
        for i in range(0,len(param),2):
            size=param[i].size()
            if len(size)==1:
                for k in range(size[0]):
                    m_model=param[i][k].item()
                    m_SGD=par[i][k].item()
                    t =t+ (m_model-m_SGD)*(m_model-m_SGD)
                    m = m+abs(m_SGD)
            else:
                for k in range(size[0]):
                    for j in range(size[1]):
                        m_model=param[i][k][j].item()
                        m_SGD=par[i][k][j].item()
                        t =t+ (m_model-m_SGD)*(m_model-m_SGD)
                        m = m+abs(m_SGD)
        return float(t/m)
    


    def Cross_validation(self,X,y,k,learning_rate: float=0.001):
        optimizer = torch.optim.RMSprop(self.parameters(), lr=learning_rate)
        criterion = nn.CrossEntropyLoss()
        kf = KFold(n_splits=k, shuffle=True, random_state=42)
        z=1
        for train_index, test_index in kf.split(X):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            X_train = X_train.float()
            X_test = X_test.float()
            y_train = y_train.float()
            y_test = y_test.float()
            print(f"length of trainset {len(y_train)}, length of testset{len(y_test)}")
            self.train()
            for epoch in range(5):
                optimizer.zero_grad()
                outputs = self(X_train)
                loss = criterion(outputs, y_train.long())
                loss.backward()
                optimizer.step()
            self.eval()
            with torch.no_grad():
                outputs = self(X_test)
                _, predicted = torch.max(outputs.data, 1)
                accuracy = (predicted == y_test.long()).sum().item() / y_test.size(0)
                print(f'Accuracy for fold {z} : {accuracy}')
                z=z+1
        return self



