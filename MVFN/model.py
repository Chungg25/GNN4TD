import torch
import torch.nn as nn
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class GCN(nn.Module): # GCN模型，向空域的第一个图卷积
    def __init__(self, adj_data, input_dim, hidden_dim, output_dim):
        super(GCN, self).__init__()  # 表示继承父类的所有属性和方法
        self.graph = adj_data
        self.linear_1 = nn.Linear(input_dim, hidden_dim)  # 定义一个线性层
        self.linear_2 = nn.Linear(hidden_dim, output_dim)  # 定义一个线性层
        self.act = nn.ReLU()  # 定义激活函数

    def forward(self, x): # (B, T, N, D)
        print(f"GCN input shape: {x.shape}")
        x = x.transpose(1,3) # (B, D, N, T)

        x = self.linear_1(x)
        x = self.act(torch.einsum('bdnt,nn->bdnt', [x, self.graph])) # (B, D, N, T)
        print(f"GCN after first graph conv: {x.shape}, values[0,0,0,:5]: {x[0,0,0,:5].detach().cpu().numpy()}")

        x = self.linear_2(x)
        x = self.act(torch.einsum('bdnt,nn->bdnt', [x, self.graph])) # (B, D, N, T)

        x = x.transpose(1, 3)  # (B, T, N, D)
        print(f"GCN output shape: {x.shape}, values[0,0,0,:5]: {x[0,0,0,:5].detach().cpu().numpy()}")
        return x


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLP, self).__init__()
        self.linear = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, output_dim))

    def forward(self, x): # (B, T, N, D)
        print(f"MLP input shape: {x.shape}")
        x = x.transpose(1, 3) # (B, D, N, T)
        x = self.linear(x)
        x = x.transpose(1, 3) # (B, T, N, D)
        print(f"MLP output shape: {x.shape}, values[0,0,0,:5]: {x[0,0,0,:5].detach().cpu().numpy()}")
        return x

class GCM(nn.Module):
    def __init__(self, adj_data, num_heads, input_dim,hidden_dim,output_dim):
        super(GCM, self).__init__()
        self.graph = adj_data
        dmodel = input_dim * 2

        self.dmodel = dmodel
        self.num_heads = num_heads
        self.relu = nn.ReLU()
        self.eps = 1e-6

        self.key = nn.Linear(dmodel, dmodel)
        self.query = nn.Linear(dmodel, dmodel)
        self.value = nn.Linear(dmodel, dmodel)

        self.gcn = GCN(adj_data, input_dim,hidden_dim,output_dim)

        self.out = nn.Linear(dmodel, dmodel)

    def get_index(self, Nodes):
        index = np.pi / 2 * torch.arange(1, Nodes + 1).reshape(1, -1, 1)
        return nn.Parameter(index, requires_grad=False)

    def forward(self,x): # (B, T, N, D)
        print(f"GCM input shape: {x.shape}")
        B, T, N, D = x.shape

        # GCN
        gcn_output = self.gcn(x) + x # (B, T, N, D)
        print(f"GCM gcn_output shape: {gcn_output.shape}, values[0,0,0,:5]: {gcn_output[0,0,0,:5].detach().cpu().numpy()}")

        # CLA
        x = x.transpose(1, 2).reshape(B, N, T * D)  # (B, N, T, D) -> (B, N, T*D)
        res = x

        q = self.query(x)
        k = self.key(x)
        v = self.value(x)

        q = self.relu(q)*q
        k = self.relu(k)*k

        weight_index = self.get_index(N).to(device)
        # (B * h, N, 2 * d)

        q_ = torch.cat( [q * torch.sin(weight_index[:, :N, :] / N), q * torch.cos(weight_index[:, :N, :] / N)], dim=-1)
        k_ = torch.cat( [k * torch.sin(weight_index[:, :N, :] / N), k * torch.cos(weight_index[:, :N, :] / N)], dim=-1)

        # (B * h, N, 2 * d) (B * h, N, d) -> (B * h, 2 * d, d)
        kv_ = torch.einsum('nld,nlm->ndm', k_, v)
        # (B * h, N, 2 * d) (B * h, 2 * d) -> (B * h, N)
        z_ = 1 / torch.clamp_min(torch.einsum('nld,nd->nl', q_, torch.sum(k_, axis=1)), self.eps)
        # (B * h, N, 2 * d) (B * h, d, 2 * d) (B * h, N) -> (B * h, N, d)
        attn_output = torch.einsum('nld,ndm,nl->nlm', q_, kv_, z_)
        # (B * h, N, d) -> (B * h, d, N) -> (B, D, N)-> (B, N, D)
        attn_output = attn_output.transpose(1, 2).reshape(B, -1, N).transpose(1, 2)
        # (B, N, D) -> (B, N, T, D) -> (B, N, D, T)

        attn_output = self.out(attn_output+res).reshape(B, N, T, D).transpose(1, 2)
        print(f"GCM attn_output shape: {attn_output.shape}, values[0,0,0,:5]: {attn_output[0,0,0,:5].detach().cpu().numpy()}")

        output = attn_output + gcn_output
        print(f"GCM output shape: {output.shape}, values[0,0,0,:5]: {output[0,0,0,:5].detach().cpu().numpy()}")

        return output


class Chomp(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        x = x[:, :, :-self.chomp_size]
        return x

class MSTCN(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=2, dropout=0.1):
        super(MSTCN, self).__init__()

        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, stride=1, padding=(kernel_size - 1) * 1, dilation=1,groups=1)
        self.chomp1 = Chomp((kernel_size - 1) * 1)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        self.mtcn1 = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1)

        self.conv1S = nn.Conv1d(in_channels, out_channels, kernel_size, stride=1, padding=(kernel_size - 1) * 1,dilation=1, groups=in_channels)
        self.chomp1S = Chomp((kernel_size - 1) * 1)
        self.relu1S = nn.ReLU()
        self.dropout1S = nn.Dropout(dropout)
        self.stcn1 = nn.Sequential(self.conv1S, self.chomp1S, self.relu1S, self.dropout1S)


        self.conv2 = nn.Conv1d(in_channels, out_channels, kernel_size, stride=1, padding=(kernel_size - 1) * 2,dilation=2, groups=1)
        self.chomp2 = Chomp((kernel_size - 1) * 2)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
        self.mtcn2 = nn.Sequential(self.conv2, self.chomp2, self.relu2, self.dropout2)

        self.conv2S = nn.Conv1d(in_channels, out_channels, kernel_size, stride=1, padding=(kernel_size - 1) * 2,dilation=2, groups=in_channels)
        self.chomp2S = Chomp((kernel_size - 1) * 2)
        self.relu2S = nn.ReLU()
        self.dropout2S = nn.Dropout(dropout)
        self.stcn2 = nn.Sequential(self.conv2S, self.chomp2S, self.relu2S, self.dropout2S)


        self.conv3 = nn.Conv1d(in_channels, out_channels, kernel_size, stride=1, padding=(kernel_size - 1) * 4, dilation=4, groups=1)
        self.chomp3 = Chomp((kernel_size - 1) * 4)
        self.relu3 = nn.ReLU()
        self.dropout3 = nn.Dropout(dropout)
        self.mtcn3 = nn.Sequential(self.conv3, self.chomp3, self.relu3, self.dropout3)

        self.conv3S = nn.Conv1d(in_channels, out_channels, kernel_size, stride=1, padding=(kernel_size - 1) * 4,dilation=4, groups=in_channels)
        self.chomp3S = Chomp((kernel_size - 1) * 4)
        self.relu3S = nn.ReLU()
        self.dropout3S = nn.Dropout(dropout)
        self.stcn3 = nn.Sequential(self.conv3S, self.chomp3S, self.relu3S, self.dropout3S)

        self.conv4 = nn.Conv1d(in_channels, out_channels, kernel_size, stride=1, padding=(kernel_size - 1) * 4,dilation=4, groups=1)
        self.chomp4 = Chomp((kernel_size - 1) * 4)
        self.relu4 = nn.ReLU()
        self.dropout4 = nn.Dropout(dropout)
        self.mtcn4 = nn.Sequential(self.conv4, self.chomp4, self.relu4, self.dropout4)


        self.conv4S = nn.Conv1d(in_channels, out_channels, kernel_size, stride=1, padding=(kernel_size - 1) * 4,dilation=4, groups=in_channels)
        self.chomp4S = Chomp((kernel_size - 1) * 4)
        self.relu4S = nn.ReLU()
        self.dropout4S = nn.Dropout(dropout)
        self.stcn4 = nn.Sequential(self.conv4S, self.chomp4S, self.relu4S, self.dropout4S)


    def forward(self, x):  # (B, T, N, D)
        print(f"MSTCN input shape: {x.shape}")
        B, T, N, D = x.shape
        x = x.reshape(B, T, -1) # (B, T, N*D)
        x = x.transpose(1, 2)  # (B, N*D, T)
        res = x
        
        x1 = self.mtcn1(x) + self.stcn1(x)
        print(f"MSTCN after layer 1: {x1.shape}, values[0,0,:5]: {x1[0,0,:5].detach().cpu().numpy()}")
        
        x2 = self.mtcn2(x1) + self.stcn2(x1)
        print(f"MSTCN after layer 2: {x2.shape}, values[0,0,:5]: {x2[0,0,:5].detach().cpu().numpy()}")
        
        x3 = self.mtcn3(x2) + self.stcn3(x2)
        print(f"MSTCN after layer 3: {x3.shape}, values[0,0,:5]: {x3[0,0,:5].detach().cpu().numpy()}")
        
        x4 = self.mtcn4(x3) + self.stcn4(x3)
        print(f"MSTCN after layer 4: {x4.shape}, values[0,0,:5]: {x4[0,0,:5].detach().cpu().numpy()}")
        
        x = res + x4
        x = x.transpose(1, 2).reshape(B, T, N, D)
        print(f"MSTCN output shape: {x.shape}, values[0,0,0,:5]: {x[0,0,0,:5].detach().cpu().numpy()}")
        return x # (B, T, N, D)

class ST_layer(nn.Module):
    def __init__(self, adj_data, input_dim, hidden_dim, output_dim):
        super(ST_layer, self).__init__()
        Nodes = len(adj_data)
        channels = Nodes * 2
        num_heads = 4

        self.gcm = GCM(adj_data, num_heads, input_dim, hidden_dim, output_dim)
        self.mstcn = MSTCN(channels, channels)

    def forward(self, x): # (B, T, N, D)
        print(f"ST_layer input shape: {x.shape}")
        res = x
        x = self.gcm(x)
        print(f"ST_layer after GCM: {x.shape}, values[0,0,0,:5]: {x[0,0,0,:5].detach().cpu().numpy()}")
        x = x + res
        x = self.mstcn(x)
        print(f"ST_layer after MSTCN: {x.shape}, values[0,0,0,:5]: {x[0,0,0,:5].detach().cpu().numpy()}")
        x = x + res
        print(f"ST_layer output shape: {x.shape}, values[0,0,0,:5]: {x[0,0,0,:5].detach().cpu().numpy()}")
        return x # (B, T, N, D)

class Network(nn.Module):
    def __init__(self, adj_data,input_dim, hidden_dim, output_dim):
        super(Network, self).__init__()
        self.st = ST_layer(adj_data,input_dim, hidden_dim, output_dim)
        self.mlp = MLP(input_dim, hidden_dim, output_dim)

    def forward(self, x): # (B, T, N, D)
        print(f"Network input shape: {x.shape}")
        res = x
        x = self.st(x)
        print(f"Network after first ST_layer: {x.shape}, values[0,0,0,:5]: {x[0,0,0,:5].detach().cpu().numpy()}")
        x = res+x
        x = self.st(x)
        print(f"Network after second ST_layer: {x.shape}, values[0,0,0,:5]: {x[0,0,0,:5].detach().cpu().numpy()}")
        x = res + x
        x = self.mlp(x) # (B, T, N, D)
        print(f"Network output shape: {x.shape}, values[0,0,0,:5]: {x[0,0,0,:5].detach().cpu().numpy()}")
        return x