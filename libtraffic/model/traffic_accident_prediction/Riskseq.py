#%%

import math
import torch
from torch.nn import init
from torch.nn.parameter import Parameter
from bisect import bisect
import numpy as np


from libcity.model.abstract_model import AbstractModel
import torch.nn as nn
import torch.nn.functional as F

class GraphConvolution(nn.Module):
    """
    Simple GCN layer
    """
    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):

        support = torch.matmul(adj, input)
        output = torch.matmul(support, self.weight)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

class GCN(nn.Module):
    def __init__(self,config,dropout):
        super(GCN, self).__init__()
        self.inputs=config['GCN_input']
        self.hidden = config['GCN_hidden']
        self.Fine = config['Fine_inputsize']
        self.gc1 = GraphConvolution(self.inputs,self.hidden)
        self.gc2 = GraphConvolution(self.hidden, self.hidden)
        self.bn = nn.BatchNorm1d(self.Fine)
        self.gc3 = GraphConvolution(self.hidden, self.hidden)
        self.gc4 = GraphConvolution(self.hidden, self.hidden)
        self.gc5 = GraphConvolution(self.hidden, self.hidden)
        self.gc6 = GraphConvolution(self.hidden, 1)
        self.dropout = dropout

    def forward(self, x, adj):
        x = self.gc1(x, adj)
        layer2 = self.gc2(x, adj)
        x = F.leaky_relu(layer2,0.8)
        #x = F.dropout(x, self.dropout, training=self.training)
        x = self.bn(x)
        x = self.gc3(x, adj)
        layer4 = self.gc4(x, adj)
        x = F.leaky_relu(layer4, 0.8)
        x = self.bn(x)
        x = self.gc5(x, adj)
        x = x + layer2 + layer4
        x = self.gc6(x, adj)
        x = F.leaky_relu(self.bn(x), 0.8)
        x = torch.squeeze(x, -1)
        #print("GCN_forward",x.shape)
        return x
        #return F.log_softmax(x, dim=1)


class M2LSTM(nn.Module): #修改激活函数
    def __init__(self, input_sz, hidden_sz,activate,num_layers):
        super().__init__()
        self.input_sz = input_sz
        self.hidden_size = hidden_sz
        self.activate=activate
        self.num_layers=num_layers
        W=[]
        U=[]
        B=[]
        W.append(nn.Parameter(torch.Tensor(input_sz, hidden_sz * 4)))
        U.append(nn.Parameter(torch.Tensor(hidden_sz, hidden_sz * 4)))
        B.append(nn.Parameter(torch.Tensor(hidden_sz * 4)))
        for i in range(num_layers-1):
            W.append(nn.Parameter(torch.Tensor(hidden_sz, hidden_sz * 4)))
            U.append(nn.Parameter(torch.Tensor(hidden_sz, hidden_sz * 4)))
            B.append(nn.Parameter(torch.Tensor(hidden_sz * 4)))
        self.wl = nn.ParameterList(W)
        self.ul = nn.ParameterList(U)
        self.bl = nn.ParameterList(B)
        self.init_weights()

    def init_weights(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def forward(self, x,
                init_states=None):
        """Assumes x is of shape (batch, sequence, feature)"""
        seq_sz, bs, hid = x.size() #batchsize_first bs is seq
        hidden_seq = []
        for t in range(bs):
            hidden_seq.append(x[:, t, :].unsqueeze(0))
        if init_states is None:
            a = torch.zeros(bs,self.hidden_size)

            h_t, c_t = (torch.zeros(bs, self.hidden_size).to(x.device))

        else:
            h_t, c_t = init_states

        HS = self.hidden_size
        for i in range(self.num_layers):
            w=self.wl[i]
            u=self.ul[i]
            b=self.bl[i]
            for t in range(bs):
                ht = h_t[:, t, :]
                ct = c_t[:, t, :]
                x_t = hidden_seq[t+i*bs].squeeze(0)
                # batch the computations into a single matrix multiplication
                gates = x_t @ w + ht @ u + b
                i_t, f_t, g_t, o_t = (
                    torch.sigmoid(gates[:, :HS]),  # input
                    torch.sigmoid(gates[:, HS:HS * 2]),  # forget
                    torch.tanh(gates[:, HS * 2:HS * 3]),
                    torch.sigmoid(gates[:, HS * 3:]),  # output
                )
                if self.activate=="relu":
                    ct = f_t * ct + i_t * g_t
                    ht = o_t * F.relu(ct)
                    hidden_seq.append(ht.unsqueeze(0))
                elif self.activate=="Leaky_relu":
                    ct = f_t * ct + i_t * g_t
                    # h_t = o_t * torch.tanh(c_t)
                    ht = o_t * F.leaky_relu(ct,0.8)
                    hidden_seq.append(ht.unsqueeze(0))
                elif self.activate=="tanh":
                    ct = f_t * ct + i_t * g_t
                    ht = o_t * torch.tanh(ct)
                    hidden_seq.append(ht.unsqueeze(0))
                else:
                    raise Exception("Unsupported activate function")

        hidden_seq = torch.cat(hidden_seq[(self.num_layers)*bs:(self.num_layers+1)*bs], dim=0)


        # reshape from shape (sequence, batch, feature) to (batch, sequence, feature)
        hidden_seq = hidden_seq.transpose(0, 1).contiguous()
        return hidden_seq, (h_t, c_t)




class Riskseq(AbstractModel):
    def __init__(self, config,data_feature):
        super(Riskseq, self).__init__(config, data_feature)
        # self.input_size = input_size
        # self.hidden_size = hidden_size
        # self.num_layers = num_layers
        self.Dynamic_m = data_feature['Dynamic_Feature']
        self.Matrix_Aff = data_feature['Matrix_Aff']
        self.Input_Seq_C = data_feature['Input_Seq_C']
        self.dec_inp_C = data_feature['dec_inp_C']
        self.dec_inp_F = data_feature['dec_inp_F']
        self.Acc_SumOut = data_feature['Acc_SumOut']
        self.Output_Seq_C = data_feature['Output_Seq_C']
        self.Output_Seq_F = data_feature['Output_Seq_F']
        self.Output_Seq = data_feature['Output_Seq']
        self.Fine_inputs = config['Fine_inputsize']
        self.Fine_hidden = config['Fine_hidsize']
        self.Coarse_inputs = config['Coarse_inputsize']
        self.Coarse_hidden = config['Coarse_hidsize']
        self.batchsize = config['batchsize']

        self.gcn = GCN(config,dropout=0.5)
        self.F_lstm1 = M2LSTM(input_sz=self.Fine_inputs,hidden_sz=self.Fine_hidden,activate="Leaky_relu",num_layers=1)
        self.F_lstm2 = M2LSTM(input_sz=self.Fine_inputs, hidden_sz=self.Fine_hidden, activate="relu", num_layers=1)
        self.C_lstm1 = M2LSTM(input_sz=self.Coarse_inputs,hidden_sz=self.Coarse_hidden,activate="Leaky_relu",num_layers=1)
        self.C_lstm2 = M2LSTM(input_sz=self.Coarse_inputs, hidden_sz=self.Coarse_hidden, activate="Leaky_relu", num_layers=1)
        self.gcFine = GraphConvolution(self.Fine_hidden, self.Fine_inputs, bias=True)
        self.gcCoarse = GraphConvolution(self.Coarse_hidden, self.Coarse_inputs, bias=True)
        self.gcSumout = GraphConvolution(self.Coarse_inputs, 1, bias=True)
        self.gcCoarse2Fine = GraphConvolution(self.Coarse_inputs,self.Fine_inputs, bias=True)
       #self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)


    def forward(self, batchsize):
        # Set initial hidden and cell states
        Input_Seq_F = []
        for i in range(8):
            if i <= 1:
                Input_Seq_F.append(self.gcn(self.Dynamic_m[i], self.Matrix_Aff[:, i]).unsqueeze(1))
            if i >= 2:
                Input_Seq_F.append(self.gcn(self.Dynamic_m[i], self.Matrix_Aff[:, 2]).unsqueeze(1))

        Input_Seq_F=torch.cat(Input_Seq_F,dim=1)
        h1 = torch.randn(batchsize, 8, self.Fine_hidden)
        c1 = torch.randn(batchsize, 8, self.Fine_hidden)
        _, (Fhs, Fcs) = self.F_lstm1(Input_Seq_F, (h1, c1))

        dec_outputs_F, (dec_memory_Fh, dec_memory_Fc) = self.F_lstm2(self.dec_inp_F, (Fhs, Fcs))
        h2 = torch.randn(batchsize, 8, self.Coarse_hidden)
        c2 = torch.randn(batchsize, 8, self.Coarse_hidden)

        _, (Chs, Ccs) = self.C_lstm1(self.Input_Seq_C, (h2, c2))

        C_lstm2 = M2LSTM(input_sz=self.Coarse_inputs, hidden_sz=self.Coarse_hidden, activate="relu", num_layers=1)

        dec_outputs_C, (dec_memory_Ch, dec_memory_Cc) = C_lstm2(self.dec_inp_C, (Chs, Ccs))

        adj1 = torch.randn(self.batchsize, self.batchsize)
        adj2 = torch.randn(self.batchsize, self.batchsize)
        adj3 = torch.randn(self.batchsize, self.batchsize)
        adj4 = torch.randn(self.batchsize, self.batchsize)
        reshaped_outputsF = [self.gcFine(dec_outputs_F[:, i, :], adj1) for i in range(6)]
        reshaped_outputsC = [self.gcCoarse(dec_outputs_C[:, i, :], adj2) for i in range(6)]
        SUM_outputs = [self.gcSumout(reshaped_outputsC[i], adj3).unsqueeze(0) for i in range(6)]
        reshaped_outputsF2 = [(self.gcCoarse2Fine(reshaped_outputsC[i], adj4) + reshaped_outputsF[i]) for i in range(6)]

        return reshaped_outputsF,reshaped_outputsC,SUM_outputs,reshaped_outputsF2

    def predict(self,batch):
        return self.forward(batch)

    def calculate_loss(self,batchsize):
        outputsF,outputsC,SUM_outputs,outputsF2 = self.predict(batchsize)
        output_lossT = torch.zeros([1],dtype=torch.float32)
        output_lossC = torch.zeros([1], dtype=torch.float32)
        output_lossF = torch.zeros([1],dtype=torch.float32)
        loss = torch.zeros([1],dtype=torch.float32)
        lamda_1 = 1
        lamda_2 = 1
        lamda_3 = 1
        for _y, _Y in zip(SUM_outputs, self.Acc_SumOut):
            output_lossT = torch.mean((_y - _Y) ** 2)

        for i in range(6):
            for _y, _Y in zip(outputsC[i], self.Output_Seq_C[:, i, :]):
                output_lossC += torch.mean((_y - _Y) ** 2)

        for i in range(6):
            for _y, _Y in zip(outputsF2[i], self.Output_Seq_F[:, i, :]):
                output_lossF += torch.mean((_y - _Y) ** 2)

        loss = lamda_1 * output_lossT + lamda_2 * output_lossC + lamda_3 * output_lossF
        print("loss:",loss)
        f=[]
        for i in range(6):
            #print(outputsF2[i].unsqueeze(0).shape)
            f.append(outputsF2[i].unsqueeze(0))
        f = torch.cat(f, dim=0)
        fn = f.detach().numpy()
        Acc_Real = np.zeros([6])
        Acc_Iden = np.zeros([6])
        Total_Acc = 0
        Pred_Acc = 0
        for batch in range(batchsize):
            for seq_step in range(6):
                Real = list(
                    np.flatnonzero((self.Output_Seq[batch, seq_step, :]) > 0))  # 10,6,354 batch_size,time_step,output_dim
                Predicted = list(np.argsort(-fn[seq_step, batch, :])[
                                 0:30])  # 1,6,10,354  1,time_step,batch_size,output_dim
                Total_Acc = Total_Acc + len(Real)
                Cross = list(set(Predicted).intersection(set(Real)))
                # print(Cross)
                Pred_Acc = Pred_Acc + len(Cross)
                Acc_Real[seq_step] = Acc_Real[seq_step] + len(Real)
                Acc_Iden[seq_step] = Acc_Iden[seq_step] + len(Cross)
        Total_Accarcy = Pred_Acc / Total_Acc
        print("事故总数为:",Total_Acc, "预测正确事故数为:",Pred_Acc,"准确率为:",Total_Accarcy)
        return loss



