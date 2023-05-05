#导入相关包
import numpy as np
import torch as t
import random
import matplotlib.pyplot as plt
#import netCDF4
import datetime
import torch
from tqdm import tqdm
from torch import nn
from torch import optim
from torch.autograd import Variable
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.autograd as autograd
import torch.nn.functional as F
from einops import rearrange, repeat
import os
#from config import args
#import seaborn as sns
#from global_land_mask import globe
#from scipy import interpolate
#plt.rcParams['font.sans-serif'] = ['SimHei'] #中文支持
%matplotlib inline

# 固定随机种子
SEED = 22

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    
seed_everything(SEED)

path='/WdHeDisk/users/zhangnong/MJO/908_test/data/s2s_dataset_for7_7_35/'
wei = '_for7_7_35_sample.npy'

X_train=np.load(path+'X_train'+wei)
Y_train=np.load(path+'Y_train'+wei)
X_valid=np.load(path+'X_valid'+wei)
Y_valid=np.load(path+'Y_valid'+wei)
X_test=np.load(path+'X_test'+wei)
Y_test=np.load(path+'Y_test'+wei)

# 构造数据管道
class MJODataset(Dataset):
    def __init__(self, data,label):
        self.data = torch.tensor(data, dtype=torch.float32)
        self.label = torch.tensor(label, dtype=torch.float32)

    def __len__(self):
        return len(self.label)
    
    def __getitem__(self, idx):
        return self.data[idx],self.label[idx]
batch_size = 16
#刚刚 batchsize = 32 还不错15～16天，lr = 1e-3， deacy = 0.001 ，数据集是22800， 1600， 1600， 模型三层4096， 1024， 90
#无dropout
trainset = MJODataset(X_train, Y_train)
trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)

validset = MJODataset(X_valid, Y_valid)
validloader = DataLoader(validset, batch_size=batch_size, shuffle=True)

testset = MJODataset(X_test, Y_test)
testloader = DataLoader(testset, batch_size=batch_size, shuffle=False)
print(len(trainloader))

#cnn模块
# 构建CNN单元
class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        if (in_channels == out_channels) and (stride == 1):
            self.res = lambda x: x
        else:
            self.res = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride)
        
    def forward(self, x):
        # 残差
        res = self.res(x)
        res = self.bn2(res)

        x = F.relu(self.bn1(x))
        x = self.conv(x)
        x = self.bn2(x)
        
        x = x + res
        
        return x
      
   #编码
class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super().__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_directions = 1
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=False)


    def forward(self, x):
        
        batch_size, T, C = x.shape
        
        h = torch.randn(self.num_directions * self.num_layers, batch_size, self.hidden_size).to(device)
        c = torch.randn(self.num_directions * self.num_layers, batch_size, self.hidden_size).to(device)

        
        #lstm训练
        output, (h, c) = self.lstm(x, (h, c))
        #output[batch_size, time_squence, hidden_size]
        #h[2, batch_size, hidden_size]

        return   h, c
 
class Attention(nn.Module):
    def __init__(self, hidden_dize, hidden_size):
        super().__init__()
        self.attn = nn.Linear(hidden_size * 2 , hidden_size, bias=False)
        self.v = nn.Linear(hidden_size, 1, bias = False)
        
    def forward(self, s, enc_output):
        
        # s = [batch_size, dec_hid_dim]
        # enc_output = [src_len, batch_size, enc_hid_dim * 2]
        

        batch_size, src_len, _ = enc_output.shape
        
        # repeat decoder hidden state src_len times
        # s = [batch_size, src_len, dec_hid_dim]
        # enc_output = [batch_size, src_len, enc_hid_dim * 2]
        s = s.unsqueeze(1).repeat(1, src_len, 1)

        
        # energy = [batch_size, src_len, dec_hid_dim]
        energy = torch.tanh(self.attn(torch.cat((s, enc_output), dim = 2)))
        
        # attention = [batch_size, src_len]
        attention = self.v(energy).squeeze(2)
        
        return F.softmax(attention, dim=1)
      
 
      
  #解码
class Decoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        
        #outsize = 45
        self.num_directions = 1  
        
        self.lstm = nn.LSTM(self.input_size, self.hidden_size, self.num_layers, batch_first=True, bidirectional=False)
        
        self.lstm2 = nn.LSTM(self.hidden_size, 128, self.num_layers, batch_first = True)
        
        self.fc1 = nn.Linear(self.hidden_size, self.output_size)
        #self.fc2 = nn.Linear(self.hidden_size, self.output_size)
        self.fc2 = nn.Linear(128, self.output_size)

    def forward(self, x, h, c):
        # x = [batchsize, input_size]
        batch_size = x.shape[0]
        x = x.view(batch_size, 1, self.input_size)
        
        output, (h, c) = self.lstm(x, (h, c))
        
        # output(batch_size, seq_len, num * hidden_size)
        rmm1 = self.fc1(output)  # pred(batch_size, 1, output_size)
        
        output, _ = self.lstm2(output)
        rmm2 = self.fc2(output)
        
        rmm1 = rmm1[:, -1, :]
        rmm2 = rmm2[:, -1, :]

        return rmm1, rmm2, h, c
      
     #定义模型
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_size = 576
        self.output_size = 35
        self.hidden_size = 256
        self.num_layers = 1
        
        self.conv1 = nn.Conv2d(3, 16, kernel_size=7, stride=2, padding=3)
        self.conv2=CNNBlock(16, 16, 3, 1, 1)
        self.conv3=CNNBlock(16, 32, 3, 2, 1)
        self.conv4=CNNBlock(32, 32, 3, 1, 1)
        self.conv5=CNNBlock(32, 64, 3, 2, 1)
        self.conv6=CNNBlock(64, 64, 3, 1, 1)

        
        self.avgpool = nn.AdaptiveAvgPool2d((3, 3))#(16,4,69)
        self.flatten = nn.Flatten()
        
        self.Encoder = Encoder(self.input_size, self.hidden_size, self.num_layers)
        self.Decoder = Decoder(self.input_size, self.hidden_size, self.num_layers, self.output_size)

    def forward(self, x):
        
        x = x[:, :, :, :, 0:3]
        batch_size, seq_len, H, W, C = x.shape
        x = x.permute(0, 1, 4, 2, 3).contiguous()
        x = x.view(batch_size * seq_len, C, H, W)
        
        #cnn部分
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)

        
        x = self.avgpool(x)
        x = self.flatten(x)
        
        _, C_new = x.shape
        
        x = x.view(batch_size, seq_len, C_new)
        
        
        
        h, c = self.Encoder(x)
        
        outputs_rmm1 = torch.zeros(batch_size, seq_len, self.output_size).to(device)
        outputs_rmm2 = torch.zeros(batch_size, seq_len, self.output_size).to(device)
        
        for t in range(seq_len):
            _input = x[:, t, :]
            
            output1, output2, h, c = self.Decoder(_input, h, c)
            
            outputs_rmm1[:, t, :] = output1
            outputs_rmm2[:, t, :] = output2
        
        rmm1 = outputs_rmm1[:, -1, :]
        rmm2 = outputs_rmm2[:, -1, :]
        
        rmm1 = rmm1.squeeze()
        rmm2 = rmm2.squeeze()
        
        rmm1 = rmm1.unsqueeze(2)
        rmm2 = rmm2.unsqueeze(2)

        RMM = torch.cat((rmm1, rmm2), dim = 2)
        
        return RMM
      
    #cor函数
def cor(Y_test,t_preds):
    #Y_test = mean_std(Y_test)
    #t_preds = mean_std(t_preds)
    cor_day = []
    a=0
    b=0
    c=0
    score_cor=0
    for i in range(0,35):
        for j in range(0,len(Y_test)):
            a+=(Y_test[j,i,:]*t_preds[j,i,:]).sum()
            b+=(Y_test[j,i]**2).sum()
            c+=(t_preds[j,i]**2).sum()
        b=np.sqrt(b)
        c=np.sqrt(c)
        cor=a/(b*c)
        cor_day.append(cor)
        a=0
        b=0
        c=0
    cor_day=np.array(cor_day)
    #print(cor_day)
    for i in range(0,len(cor_day)):
        if cor_day[i]<0.50:
            break
    score_cor=i
    return score_cor

def rmse_new(Y_valid,preds):
    #Y_valid = mean_std(Y_valid)
    #preds = mean_std(preds)
    rmse_day = []
    a=0
    score_rmse=0
    for i in range(0,35):
        for j in range(0,len(Y_valid)):
            a+=(torch.pow((Y_valid[j,i,:]-preds[j,i,:]),2)).sum()
        rmse=a/len(Y_valid)
        rmse=np.sqrt(rmse)
        rmse_day.append(rmse)
        a=0
    rmse_day=np.array(rmse_day)
    #print(rmse_day)
    for i in range(0,len(rmse_day)):
        if rmse_day[i]>1.40:
            break
    score_rmse=i
    '''
    day_cal=np.argwhere(rmse_day<1.40)
    print(day_cal)
    if len(day_cal)==0:
        score_rmse=0
    else:
        score_rmse=day_cal.max()+1
    '''
    #score_rmse=rmse_day.max()
    return score_rmse

def rmse_max(Y_valid,preds):
    #Y_valid = mean_std(Y_valid)
    #preds = mean_std(preds)
    rmse_day = []
    a=0
    score_rmse=0
    for i in range(0,35):
        for j in range(0,len(Y_valid)):
            a+=(torch.pow((Y_valid[j,i,:]-preds[j,i,:]),2)).sum()
        rmse=a/len(Y_valid)
        rmse=np.sqrt(rmse)
        rmse_day.append(rmse)
        a=0
    rmse_day=np.array(rmse_day)
    score_rmse=rmse_day.max()
    return score_rmse
  
  #自定义损失函数
class My_MSELoss(nn.Module):
    def __init__(self):
        super(My_MSELoss, self).__init__()
        self.mseloss = nn.MSELoss(reduction='mean')

    def forward(self, pred, label):
        loss1 = self.mseloss(pred[:,:,0], label[:,:,0])
        loss2 = self.mseloss(pred[:,:,1], label[:,:,1])
        
        loss = (loss1 +loss2)/2

        return loss
      
   #设置参数
model_weights = '/WdHeDisk/users/zhangnong/MJO/908_test/cnn_seq2seq_s2s_data_weights.pth'

model = Model().to(device)

optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=0.001)  # weight_decay是L2正则化参数

epochs = 25

criterion = My_MSELoss()
#criterion = nn.MSELoss()

train_losses, valid_losses = [], []
scores_rmse = []
#scores_cor = []
best_score1 = float('-inf')
best_score2 = float('inf')
preds = np.zeros((len(Y_valid),35,2))
y_valid=np.zeros((len(Y_valid),35,2))
#Y_valid=t.from_numpy(Y_valid)
print(preds.shape)

#模型训练
for epoch in range(epochs):
    print('Epoch: {}/{}'.format(epoch+1, epochs))
    model.train()
    losses = 0
    for data1, labels in tqdm(trainloader):
        data1 = data1.to(device)
        labels = labels.to(device)
        model.zero_grad()
        #optimizer.zero_grad()
        pred = model(data1)
        #print('pred.size:',pred.size())
        #pred= pred.permute(0,2,1).contiguous()
        loss = criterion(pred, labels)
        losses += loss.cpu().detach().numpy()
        #print(pred.dtype)
        #print(labels.dtype)
        #print(loss.dtype)
        loss.backward()
        optimizer.step()
        #scheduler.step()
    train_loss = losses / len(trainloader)
    train_losses.append(train_loss)
    print('Training Loss: {:.3f}'.format(train_loss))
# 模型验证
    model.eval()
    losses = 0
    s=0
    ss=0
    s_rmse=0
    with torch.no_grad():
        for i, data in tqdm(enumerate(validloader)):
            data1, labels = data
            data1 = data1.to(device)
            y_valid[i*batch_size:(i+1)*batch_size]=labels.detach().cpu()
            labels = labels.to(device)
            pred = model(data1)

            #pred= pred.permute(0,2,1).contiguous()
            loss = criterion(pred, labels)
            #print('loss:',loss)
            losses += loss.cpu().detach().numpy()
            preds[i*batch_size:(i+1)*batch_size] = pred.detach().cpu()
        valid_loss = losses / len(validloader)
        valid_losses.append(valid_loss)
        print('Validation Loss: {:.3f}'.format(valid_loss))
    #print(len(valid_losses))
        preds=torch.as_tensor(preds)
        y_valid=torch.as_tensor(y_valid)
    #print(preds.dtype,Y_valid.dtype)
    #print(len(preds))
    #print(len(preds[1]))
        s=rmse_new(y_valid,preds)
    #s=rmse(Y_valid,preds)
    #s=rmse_caluate(Y_valid,preds)
        ss=cor(y_valid,preds)
        s_rmse=rmse_max(y_valid,preds)
    #s.item()
    #print(s)
    #s=score_rmse(Y_valid,preds)
        scores_rmse.append(s)
    #scores_cor.append(ss)
        print('day_rmse: {:}'.format(s))
        print('day_cor: {:}'.format(ss))
        print('rmse_max: {:.3f}'.format(s_rmse))
    #print('Score_cor: {:.3f}'.format(ss))
# 保存最佳模型权重
    #s=s.numpy()
    #ss=ss.numpy()
    #fina_score=score(s,ss)
    #print(s.type)
    #print(ss.type)
    final_s=min(s,ss)
    if (final_s > best_score1) :
        best_score1 = final_s
        best_score2 = s_rmse
        #best_score2 = ss
        checkpoint = {'best_score': best_score1,'state_dict': model.state_dict()}
        torch.save(checkpoint, model_weights)
    if (final_s==best_score1)&(s_rmse < best_score2):
        best_score1 = final_s
        best_score2 = s_rmse
        #best_score2 = ss
        checkpoint = {'best_score': best_score1,'state_dict': model.state_dict()}
        torch.save(checkpoint, model_weights) 
    print('best_day:{:}'.format(best_score1))
    print('best_rmse:{:.3f}'.format(best_score2))
    #scheduler.step()
    
  
