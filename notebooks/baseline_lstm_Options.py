import torch
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt
import os
import copy
import numpy as np

os.chdir("C:/Users/tonyz/Desktop/GRUVAE/data/usingmean")
data = pd.read_csv("AMZN_put.csv").dropna()
data = data[["Stock"]]

os.chdir("C:/Users/tonyz/Desktop/RNN notebook")
from data_loader_LSTM_Options import create_inout_sequences

train_steps = 300
val_steps = 30
tw = 5
pred_size = 30
train_size = train_steps/tw
validation_size = int(val_steps/tw)
test_size = int(len(data)//tw-train_size)
train_loader, test_loader,scaler = create_inout_sequences(data, tw, test_size)


class LSTM(nn.Module):
    def __init__(self, input_size=1, hidden_layer_size=64, output_size=1):
        super().__init__()
        self.hidden_layer_size = hidden_layer_size

        self.lstm = nn.GRU(input_size, hidden_layer_size, batch_first=True)

        self.linear = nn.Linear(hidden_layer_size, output_size)

    def forward(self, inputs):
        
        _,lstm_out = self.lstm(inputs.view(1,len(inputs),1).float())
        lstm_out = lstm_out.view(1,-1)
        prediction = self.linear(lstm_out)
        return prediction
        
torch.manual_seed(0)
device = torch.device("cuda")
model = LSTM().to(device)
loss_function = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.005)

epochs = 50

def train_and_val(epochs):
    models = []
    val_loss_list = []
    x = train_loader.dataset.tensors[0].to(device)
    y = train_loader.dataset.tensors[1].to(device)
    
    x2 = test_loader.dataset.tensors[0].to(device)
    y2 = test_loader.dataset.tensors[1].to(device)
    for j in range(epochs):
        train_loss = 0
        model.train()
        optimizer.zero_grad()
        for i in range(len(x)):
            y_pred = model(x[i])
            single_loss = loss_function(y_pred.float(), y[i].float().view(-1,1))
            train_loss += single_loss
        train_loss.backward()
        optimizer.step()
        copymodel = copy.deepcopy(model)
        models.append(copymodel)

        val_loss = nn.MSELoss()
        with torch.no_grad():
            model.eval()
            validation_x = x2[:validation_size]
            validation_y = y2[:validation_size]
            validation_loss = 0
            for k in range(len(validation_x)):
                prediction = model(validation_x[k])
                loss = val_loss(prediction.float(), validation_y[k].view(-1,1).float())
                validation_loss += loss
            val_loss_list.append(validation_loss.item())
        print(f"\tEpoch: {j}, Train Loss: {train_loss:.4f}, Validation Loss: {validation_loss:.4f}")
    val, idx = min((val, idx) for (idx, val) in enumerate(val_loss_list))
    print(idx)
    return models[idx]


chosen_model = train_and_val(epochs).to("cpu")
test_x = test_loader.dataset.tensors[0][validation_size:]
test_y = test_loader.dataset.tensors[1][validation_size:]

model.to("cpu")
test_predictions = []
tests = []
for i in range(len(test_x)):
    prediction = chosen_model(test_x[i]).view(-1).detach().tolist()
    test_predictions.append(prediction)
    test = test_y[i].tolist()
    tests.append(test)
    
def inverse(y, scaler):
    d = torch.cat([torch.tensor(y),torch.tensor(y)], dim=-1).numpy()
    d = scaler.inverse_transform(d)
    return d

test_predictions = inverse(torch.tensor(test_predictions).view(-1,1),scaler)[:,1][:pred_size]
tests = inverse(torch.tensor(tests).view(-1,1),scaler)[:,1][:pred_size]
plt.plot(test_predictions)
plt.plot(tests)

def compute_nrmse(prediction,actual):
    #prediction = prediction.tolist()
    actual = actual.tolist()
    mse = []
    for i in range(len(prediction)):
       mse.append((prediction[i]-actual[i])**2)
    nrmse = ((sum(mse)/len(actual))**0.5)/np.mean(actual)
    return nrmse 

def compute_mape(prediction,actual):
    #prediction = prediction.tolist()
    actual = actual.tolist()
    mape = []
    for i in range(len(prediction)):
       mape.append(abs(actual[i]-prediction[i])/actual[i])  
    mape = sum(mape)/len(actual)
    return mape



index = np.linspace(5,30,6)
for i in index:
    i = int(i)
    ours = test_predictions[:i]
    nrmse = compute_nrmse(ours,tests[:i])
    print(f"\tstep: {i}, ours: {nrmse:.4f}")
    






