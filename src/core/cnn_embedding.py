import torch
import torch.nn as nn


class CNNEmbeddingSimple(nn.Module):
    def __init__(self, input_features, d_model):
        super(CNNEmbeddingSimple, self).__init__()
        self.conv1 = nn.Conv1d(input_features, d_model, 1)
        
    def forward(self, x):
        x = self.conv1(x)
        return x
        
class CNNEmbeddingBottonUp(nn.Module):
    def __init__(self, input_features, d_model, kernel_sizes):
        super(CNNEmbeddingBottonUp, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=input_features, out_channels=64, kernel_size=kernel_sizes[0], padding=(kernel_sizes[0]-1)//2)
        self.relu = nn.ReLU()
        #self.maxpool1 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=d_model, kernel_size=kernel_sizes[1], padding=(kernel_sizes[1]-1)//2)
        #self.maxpool2 = nn.MaxPool1d(kernel_size=2, stride=2)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        #x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.relu(x)
        #x = self.maxpool2(x)
        
        return x
    
class CNNEmbeddingTopDown(nn.Module):
    def __init__(self, input_features, d_model, kernel_sizes):
        super(CNNEmbeddingTopDown, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=input_features, out_channels=512, kernel_size=kernel_sizes[0], padding=(kernel_sizes[0]-1)//2)
        self.relu = nn.ReLU()
        # if d_model is 256
        self.conv2 = nn.Conv1d(in_channels=512, out_channels=d_model, kernel_size=kernel_sizes[1], padding=(kernel_sizes[1]-1)//2)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        return x
    
    
class CNNEmbedding3LayerBottonUp(nn.Module):
    def __init__(self, input_features, d_model, kernel_sizes):
        super(CNNEmbedding3LayerBottonUp, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=input_features, out_channels=64, kernel_size=kernel_sizes[0], padding=(kernel_sizes[0]-1)//2)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=kernel_sizes[1], padding=(kernel_sizes[1]-1)//2)
        #self.maxpool2 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv1d(in_channels=128, out_channels=d_model, kernel_size=kernel_sizes[2], padding=(kernel_sizes[2]-1)//2)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        #x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.relu(x)
        #x = self.maxpool2(x)
        
        return x
    
    
class CNNEmbedding3LayerTopDown(nn.Module):
    def __init__(self, input_features, d_model, kernel_sizes):
        super(CNNEmbedding3LayerTopDown, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=input_features, out_channels=512, kernel_size=kernel_sizes[0], padding=(kernel_sizes[0]-1)//2)
        self.relu = nn.ReLU()
        #self.maxpool1 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv1d(in_channels=512, out_channels=256, kernel_size=kernel_sizes[1], padding=(kernel_sizes[1]-1)//2)
        #self.maxpool2 = nn.MaxPool1d(kernel_size=2, stride=2)
        # if d_model is 128
        self.conv3 = nn.Conv1d(in_channels=256, out_channels=d_model, kernel_size=kernel_sizes[1], padding=(kernel_sizes[1]-1)//2)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        #x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.relu(x)
        #x = self.maxpool2(x)
        
        return x
    
    
class CNNEmbeddingReducePooling(nn.Module):
    def __init__(self, input_features, d_model, kernel_sizes):
        super(CNNEmbeddingReducePooling, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=input_features, out_channels=16, kernel_size=kernel_sizes[0], padding=(kernel_sizes[0]-1)//2)
        self.relu = nn.ReLU()
        self.maxpool1 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=d_model, kernel_size=kernel_sizes[1], padding=(kernel_sizes[1]-1)//2)
        self.maxpool2 = nn.MaxPool1d(kernel_size=2, stride=2)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        #x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.maxpool2(x)
        
        return x

class CNNEmbeddingLayerBottonUpPooling(nn.Module):
    def __init__(self, input_features, d_model, kernel_sizes):
        super(CNNEmbeddingLayerBottonUpPooling, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=input_features, out_channels=64, kernel_size=kernel_sizes[0], padding=(kernel_sizes[0]-1)//2)
        self.relu = nn.LeakyReLU() #ReLU()
        #self.maxpool1 = nn.MaxPool1d(kernel_size=2, stride=2)
        #self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=kernel_sizes[1], padding=(kernel_sizes[1]-1)//2)
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=d_model, kernel_size=kernel_sizes[1], padding=(kernel_sizes[1]-1)//2, stride=2)
        #self.maxpool2 = nn.MaxPool1d(kernel_size=2, stride=2)
        #self.avgpool = nn.AvgPool1d(kernel_size=2,  stride=2)
        self.maxpool1 = nn.MaxPool1d(kernel_size=2,  stride=2)
        #self.conv3 = nn.Conv1d(in_channels=128, out_channels=d_model, kernel_size=kernel_sizes[2], padding=(kernel_sizes[2]-1)//2)#, stride=2)
        self.conv3 = nn.Conv1d(in_channels=64, out_channels=d_model, kernel_size=kernel_sizes[2], padding=(kernel_sizes[2]-1)//2, stride=2)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        #x = self.maxpool1(x)
        x = self.conv3(x)
        x = self.relu(x)
        #x = self.conv3(x)
        #x = self.relu(x)
        #x = self.maxpool1(x)
            
        return x