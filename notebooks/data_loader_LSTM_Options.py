import torch
from sklearn.preprocessing import StandardScaler
import torch.utils.data as data_utils
import numpy as np


def create_inout_sequences(data, tw, test_size):
    
    data = np.array(data)
    L = len(data)
    data = data[:(L//tw)*tw]
    scaler = StandardScaler()
    scaler = scaler.fit(data[:-tw*test_size])
    data = scaler.transform(data)
    
    x = []
    y = []
    
    for i in range(len(data)-tw):
        train_seq = data[i:i+tw]
        train_label = data[i+tw]
        x.append(train_seq)
        y.append(train_label)

    y_train = torch.tensor(y[:-test_size])
    y_test = torch.tensor(y[-test_size:])
    x_train = torch.tensor(x[:-test_size])
    x_test = torch.tensor(x[-test_size:])  
    
    train_dataset = data_utils.TensorDataset(x_train, y_train)
    train_loader = data_utils.DataLoader(train_dataset, batch_size=len(train_dataset), shuffle=False)

    test_dataset = data_utils.TensorDataset(x_test, y_test)
    test_loader = data_utils.DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False)
    
    return train_loader, test_loader, scaler

