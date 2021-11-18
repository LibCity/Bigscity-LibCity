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
        # print(input)
        # print(adj)
        # print(input.shape)
        # print(adj.shape)
        support = torch.matmul(adj, input)
        output = torch.matmul(support, self.weight)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

class GCN(nn.Module):
    def __init__(self, nfeat, nhid, dropout):
        super(GCN, self).__init__()

        self.gc1 = GraphConvolution(5, 320)
        self.gc2 = GraphConvolution(320, 320)
        self.bn = nn.BatchNorm1d(354)
        self.gc3 = GraphConvolution(320, 320)
        self.gc4 = GraphConvolution(320, 320)
        self.gc5 = GraphConvolution(320, 320)
        self.gc6 = GraphConvolution(320, 1)
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

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
        #print(seq_sz , bs, hid)
        #print(self.hidden_size)
        hidden_seq = []
        for t in range(bs):
            hidden_seq.append(x[:, t, :].unsqueeze(0))
        if init_states is None:
            a = torch.zeros(bs,self.hidden_size)

            h_t, c_t = (torch.zeros(bs, self.hidden_size).to(x.device))

        else:
            h_t, c_t = init_states
            # h_t = torch.mean(h_t,dim=0)
            # c_t = torch.mean(c_t, dim=0)

        HS = self.hidden_size
        for i in range(self.num_layers):
            w=self.wl[i]
            u=self.ul[i]
            b=self.bl[i]
            #print("w",w.shape)
            for t in range(bs):
                ht = h_t[:, t, :]
                ct = c_t[:, t, :]
                x_t = hidden_seq[t+i*bs].squeeze(0)
                #print("x_t",x_t.shape)
                # batch the computations into a single matrix multiplication
                #print("type:",x_t.dtype,w.dtype,ht.dtype,u.dtype)
                gates = x_t @ w + ht @ u + b
                i_t, f_t, g_t, o_t = (
                    torch.sigmoid(gates[:, :HS]),  # input
                    torch.sigmoid(gates[:, HS:HS * 2]),  # forget
                    torch.tanh(gates[:, HS * 2:HS * 3]),
                    torch.sigmoid(gates[:, HS * 3:]),  # output
                )
                if self.activate=="relu":
                    ct = f_t * ct + i_t * g_t
                    #h_t = o_t * torch.tanh(c_t)
                    ht = o_t * F.relu(ct)
                    #print("ht:",ht.shape)
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




def data_preprocess():
    T_start = 9000
    T_end = 20000

    Total_Num = 21744
    Acc_total = np.load('./Dataset/Acc_sum.npy')  # 5 month 01~05/2017
    Pickup_total = np.load('Dataset/Taxipick_sum.npy')
    Dropoff_total = np.load('Dataset/Taxidrop_sum.npy')
    Taxi_pickd_total = np.load('Dataset/Taxi_pick_diff.npy')
    Speed_total = np.load('Dataset/Speed_sum_fillFM.npy')
    Y_coarse = np.load('Dataset/Y_Coarse_Risk.npy')
    External_total = np.load('Dataset/ex_time_stamp_10min.npy')
    Static_affinity = np.load('Dataset/static_affinity.npy')

    AccSUM = np.sum(Acc_total, axis=0)
    Speed_total = np.where(Speed_total > 100, 0, Speed_total)
    Speed_total = np.where(Speed_total > 75, 75, Speed_total)
    Speed_total_diff = np.zeros([354, Total_Num])
    # 选定哪些区域有taxi经过 可以计算交通模式的相似度
    TaxiRegIndex = list(np.flatnonzero(np.sum(Pickup_total, axis=1)))
    for i in range(1, Total_Num):
        Speed_total_diff[:, i] = abs(Speed_total[:, i] - Speed_total[:, i - 1])

    # 对pickup的数据做处理（log化减小差距）
    Taxi_pickd_total = np.log2(Taxi_pickd_total + 2)
    Pickup_total = np.log2(Pickup_total + 2)
    Speed_total = Speed_total / 10
    # 对speed进行处理
    Mean_speed = np.mean(Speed_total)
    Speed_total = np.where(Speed_total == 0, Mean_speed, Speed_total)
    Speed_total_diff = np.log2(Speed_total_diff + 2)
    Acc_prob_negnew = np.load('Dataset/Acc_prob_negnew.npy')
    Acc_total_new = np.zeros([Acc_total.shape[0], Acc_total.shape[1]])
    for i in range(354):
        Acc_total_new[i, :] = np.where(Acc_total[i, :] == 0, Acc_prob_negnew[i], Acc_total[i, :] * 3.5)
    Acc_total = Acc_total_new / 10

    import scipy.stats

    def JS_divergence(p, q):
        M = (p + q) / 2
        return 0.5 * scipy.stats.entropy(p, M) + 0.5 * scipy.stats.entropy(q, M)

    def Cal_Matrix(t):
        TrafficPatt = np.zeros([354, 7])  # t时刻之前7个interval
        for i in range(7):
            TrafficPatt[:, i] = Pickup_total[:, t - 144 * i]
        Dynamic_Aff = np.zeros([354, 354])
        for i in TaxiRegIndex:
            for j in TaxiRegIndex:
                if np.sum(TrafficPatt[i, :]) != 0 and np.sum(TrafficPatt[j, :]) != 0:
                    Dynamic_Aff[i, j] = JS_divergence(TrafficPatt[i, :], TrafficPatt[j, :])
                    Dynamic_Aff[i, j] = np.exp(-Dynamic_Aff[i, j] / 0.35)
                    Dynamic_Aff[j, i] = Dynamic_Aff[i, j]
        Dynamic_Aff = Dynamic_Aff + Static_affinity

        #     A = Dynamic_Aff + np.eye(354) #Node_Num
        #     D_12 = np.diag(np.power(np.array(A.sum(1)), -0.5).flatten(), 0)
        #     L = A.dot(D_12).transpose().dot(D_12)*10
        #    return L
        return Dynamic_Aff

    def Construct_inp_out(t, h):  # t，h 表示对某一个时刻t之后h个interval进行预测，Acc_Risk, Pick, Speed, Pick_diff, Speed_diff
        # organize input sequence
        scale = 10
        Acc_D1 = (np.mean(Acc_total[:, t - 432:t - 288], axis=1) * scale).reshape([354, 1])
        Acc_D2 = ((Acc_total[:, t - 432] + Acc_total[:, t - 288])).reshape([354, 1]) / 2
        Acc_R6 = Acc_total[:, t - 5:t + 1].reshape([354, 6])
        Input_Batch_Acc = np.concatenate((Acc_D1, Acc_D2, Acc_R6), axis=1)

        Pick_D1 = (np.mean(Pickup_total[:, t - 432:t - 288], axis=1)).reshape([354, 1])
        Pick_D2 = ((Pickup_total[:, t - 432] + Pickup_total[:, t - 288])).reshape([354, 1]) / 2
        Pick_R6 = Pickup_total[:, t - 5:t + 1].reshape([354, 6])
        Input_Batch_Pick = np.concatenate((Pick_D1, Pick_D2, Pick_R6), axis=1)

        Speed_D1 = (np.mean(Speed_total[:, t - 432:t - 288], axis=1)).reshape([354, 1])
        Speed_D2 = ((Speed_total[:, t - 432] + Speed_total[:, t - 288])).reshape([354, 1]) / 2
        Speed_R6 = Speed_total[:, t - 5:t + 1].reshape([354, 6])
        Input_Batch_Speed = np.concatenate((Speed_D1, Speed_D2, Speed_R6), axis=1)

        Pickdiff_D1 = (np.mean(Taxi_pickd_total[:, t - 432:t - 288], axis=1)).reshape([354, 1])
        Pickdiff_D2 = ((Taxi_pickd_total[:, t - 432] + Taxi_pickd_total[:, t - 288])).reshape([354, 1]) / 2
        Pickdiff_R6 = Taxi_pickd_total[:, t - 5:t + 1].reshape([354, 6])
        Input_Batch_PickDif = np.concatenate((Pickdiff_D1, Pickdiff_D2, Pickdiff_R6), axis=1)

        Speeddf_D1 = (np.mean(Speed_total_diff[:, t - 432:t - 288], axis=1)).reshape([354, 1])
        Speeddf_D2 = ((Speed_total_diff[:, t - 432] + Speed_total_diff[:, t - 288])).reshape([354, 1]) / 2
        Speeddf_R6 = Speed_total_diff[:, t - 5:t + 1].reshape([354, 6])
        Input_Batch_SpeedDif = np.concatenate((Speeddf_D1, Speeddf_D2, Speeddf_R6), axis=1)

        Ycoarse_D1 = np.mean(Y_coarse[:, t - 432:t - 288], axis=1).reshape([1, 18])
        Ycoarse_D2 = ((Y_coarse[:, t - 432] + Y_coarse[:, t - 288]) / 2).reshape([1, 18])
        Ycoarse_R6 = Y_coarse[:, t - 5:t + 1].transpose()
        Input_Batch_Ycoarse = np.concatenate([Ycoarse_D1, Ycoarse_D2, Ycoarse_R6])

        Input_Batch = []
        Input_Batch.append(Input_Batch_Acc)
        Input_Batch.append(Input_Batch_Pick)
        Input_Batch.append(Input_Batch_Speed)
        Input_Batch.append(Input_Batch_PickDif)
        Input_Batch.append(Input_Batch_SpeedDif)
        Input_Batch = np.array(Input_Batch)

        External_D1 = np.mean(External_total[t - 432:t - 288, :], axis=0).reshape([1, 9])
        External_D1[0, 3] = 0
        External_D2 = ((External_total[t - 432, :] + External_total[t - 288, :]) / 2).reshape([1, 9])
        External_R6 = External_total[t - 5:t + 1, :]
        input_Ext = np.concatenate((External_D1, External_D2, External_R6), axis=0)

        ##-------------option to hide these codes-------------#
        # Matrix_D1 = (Cal_Matrix(t-432)+Cal_Matrix(t-288))/2
        # Matrix_D1 = Matrix_D1.reshape([1,354,354])
        # Matrix_D2 = Matrix_D1
        # Matrix_D2 = Matrix_D2.reshape([1,354,354])
        # Matrix_R6 = np.zeros([354,354])
        # for i in range(0,6):
        #      Matrix_R6 = Matrix_R6 + Cal_Matrix(t-i*144)
        # Matrix_R6 = Matrix_R6/6
        # Matrix_R6 = Matrix_R6.reshape([1,354,354])
        # Matrix_Aff = np.concatenate((Matrix_D1,Matrix_D2,Matrix_R6),axis=0)
        ##-------------option to hide these codes-------------#

        ##-------------option to hide these codes-------------#
        Matrix_D1 = Cal_Matrix(t - 3).reshape([1, 354, 354])
        Matrix_Aff = np.concatenate((Matrix_D1, Matrix_D1, Matrix_D1), axis=0)
        ##-------------option to hide these codes-------------#
        # organize output sequence
        Output_Risk = Acc_total[:, t + 1:t + h + 1].transpose()
        Output_Ycoarse = Y_coarse[:, t + 1:t + h + 1].transpose()
        Output_Ext = External_total[t + 1:t + h + 1, :]
        Output_AccSUM = AccSUM[t + 1:t + h + 1]
        return Input_Batch, input_Ext, Matrix_Aff, Output_Risk, Input_Batch_Ycoarse, Output_Ycoarse, Output_Ext, Output_AccSUM

    def generate_samples(t1, t2, batch_size):
        tt = np.random.choice(range(t1, t2), batch_size)
        #     tt = np.array([10287, 10289, 10293, 10296, 10298, 10302,
        #        10304, 10305, 10307, 10309, 10311, 10313, 10315, 10316, 10318,
        #        10320, 10322, 10323, 10325, 10327, 10329, 10330, 10332, 10334,
        #        10336, 10338, 10341, 10342, 10343, 10345])
        Input_Batch = []
        Intput_Ycoarse_B = []
        Input_Ext_B = []
        Matrix_Aff_B = []
        Output_Risk_B = []
        Output_Ycoarse_B = []
        Output_AccSUM_B = []
        Output_Ext_B = []
        for i in tt:
            Input_Seq, input_Ext, Matrix_Aff, Output_Risk, input_seq_Ycoarse, Output_Ycoarse, Output_Ext, Output_AccSUM = Construct_inp_out(
                i, 6)
            Input_Batch.append(Input_Seq)
            Input_Ext_B.append(input_Ext)
            Matrix_Aff_B.append(Matrix_Aff)
            Output_Risk_B.append(Output_Risk)
            Output_Ycoarse_B.append(Output_Ycoarse)
            Output_Ext_B.append(Output_Ext)
            Intput_Ycoarse_B.append(input_seq_Ycoarse)
            Output_AccSUM_B.append(Output_AccSUM)
        Input_Batch = np.array(Input_Batch)
        Input_Ext_B = np.array(Input_Ext_B)
        Matrix_Aff_B = np.array(Matrix_Aff_B)
        Output_Risk_B = np.array(Output_Risk_B)
        Intput_Ycoarse_B = np.array(Intput_Ycoarse_B)
        Output_Ycoarse_B = np.array(Output_Ycoarse_B)
        Output_Ext_B = np.array(Output_Ext_B)
        Output_AccSUM_B = np.array(Output_AccSUM_B)
        return Input_Batch, Input_Ext_B, Matrix_Aff_B, Output_Risk_B, Intput_Ycoarse_B, Output_Ycoarse_B, Output_Ext_B, Output_AccSUM_B, tt

    Batchsize = 30
    Input_Batch, Input_Ext, Matrix_Aff, Output_Seq, Input_Ycoarse, Output_Ycoarse, Ext_O, Acc_SumS, SelectedTime = generate_samples(
            2500, 4500, Batchsize)
    Acc_Risk = Input_Batch[:, 0, :, :]
    Pick = Input_Batch[:, 1, :, :]
    Speed = Input_Batch[:, 2, :, :]
    Pick_diff = Input_Batch[:, 3, :, :]
    Speed_diff = Input_Batch[:, 4, :, :]

    Dynamic_Feature = []
    for i in range(8):
        Input = np.concatenate([np.expand_dims(Acc_Risk[:, :, i], -1), np.expand_dims(Pick[:, :, i], -1),
                                np.expand_dims(Speed[:, :, i], -1), np.expand_dims(Pick_diff[:, :, i], -1),
                                np.expand_dims(Speed_diff[:, :, i], -1)], axis=-1)
        Dynamic_Feature.append(Input)
    Dynamic_Feature = torch.from_numpy(np.array(Dynamic_Feature))
    Dynamic_Feature = Dynamic_Feature.to(torch.float32)

    Matrix_Aff = torch.from_numpy(Matrix_Aff)
    Matrix_Aff = Matrix_Aff.to(torch.float32)

    Input_Seq_C = []
    for i in range(8):
        Input_Seq_C.append(torch.from_numpy(Input_Ycoarse[:, i, :]).unsqueeze(1))
    Input_Seq_C = torch.cat(Input_Seq_C, dim=1)
    Input_Seq_C = Input_Seq_C.to(torch.float32)


    Output_Seq_F = []  # 6*354
    for i in range(6):
        Output_Seq_F.append(torch.from_numpy(Output_Seq[:, i, :]).unsqueeze(1))
    Output_Seq_F = torch.cat(Output_Seq_F, dim=1)


    Output_Seq_C = []  # 6*18
    for i in range(6):
        Output_Seq_C.append(torch.from_numpy(Output_Ycoarse[:, i, :]).unsqueeze(1))
    Output_Seq_C = torch.cat(Output_Seq_C, dim=1)


    target_seq_F = Output_Seq_F
    target_seq_C = Output_Seq_C


    dec_inp_F = torch.cat((torch.zeros_like(target_seq_F[:, 0, :].unsqueeze(1)), target_seq_F[:, :-1, :]), dim=1)
    dec_inp_F = dec_inp_F.to(torch.float32)

    dec_inp_C = torch.cat((torch.zeros_like(target_seq_C[:, 0, :].unsqueeze(1)), target_seq_C[:, :-1, :]), dim=1)
    dec_inp_C = dec_inp_C.to(torch.float32)
    # accident 总数
    Acc_SumOut = []
    for i in range(6):
        Acc_SumOut.append(torch.from_numpy(Acc_SumS[:, i]).unsqueeze(0))
    Acc_SumOut = torch.cat(Acc_SumOut, dim=0)

    datadic={}
    datadic['Dynamic_Feature']=Dynamic_Feature
    datadic['Input_Seq_C'] = Input_Seq_C
    datadic['Output_Seq'] = Output_Seq
    datadic['Output_Seq_F'] = Output_Seq_F
    datadic['Output_Seq_C'] = Output_Seq_C
    datadic['dec_inp_F'] = dec_inp_F
    datadic['dec_inp_C'] = dec_inp_C
    datadic['Acc_SumOut'] = Acc_SumOut
    datadic['Matrix_Aff'] = Matrix_Aff
    datadic['dec_inp_F'] = dec_inp_F
    datadic['dec_inp_C'] = dec_inp_C
    datadic['Input_Ycoarse'] = Input_Ycoarse
    return datadic



class Riskseq(AbstractModel):
    def __init__(self, config,data_feature):
        super(Riskseq, self).__init__(config, data_feature)
        # self.input_size = input_size
        # self.hidden_size = hidden_size
        # self.num_layers = num_layers
        self.Dynamic_m = data_feature['Dynamic_Feature']
        self.Matrix_Aff = data_feature['Matrix_Aff']
        self.Input_Ycoarse = data_feature['Input_Ycoarse']
        self.Input_Seq_C = data_feature['Input_Seq_C']
        self.dec_inp_C = data_feature['dec_inp_C']
        self.dec_inp_F = data_feature['dec_inp_F']
        self.Acc_SumOut = data_feature['Acc_SumOut']
        self.Output_Seq = data_feature['Output_Seq']
        self.Output_Seq_C = data_feature['Output_Seq_C']
        self.Output_Seq_F = data_feature['Output_Seq_F']

        self.device = config['device']

        self.gcn = GCN(nfeat=354,nhid=320,dropout=0.5)
        self.F_lstm1 = M2LSTM(input_sz=354,hidden_sz=192,activate="Leaky_relu",num_layers=1)
        self.F_lstm2 = M2LSTM(input_sz=354, hidden_sz=192, activate="relu", num_layers=1)
        self.C_lstm1 = M2LSTM(input_sz=18,hidden_sz=16,activate="Leaky_relu",num_layers=1)
        self.C_lstm2 = M2LSTM(input_sz=18, hidden_sz=16, activate="Leaky_relu", num_layers=1)
        self.gcFine = GraphConvolution(192, 354, bias=True)
        self.gcCoarse = GraphConvolution(16, 18, bias=True)
        self.gcSumout = GraphConvolution(18, 1, bias=True)
        self.gcCoarse2Fine = GraphConvolution(18, 354, bias=True)
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
        h1 = torch.randn(batchsize, 8, 192)
        c1 = torch.randn(batchsize, 8, 192)
        _, (Fhs, Fcs) = self.F_lstm1(Input_Seq_F, (h1, c1))

        dec_outputs_F, (dec_memory_Fh, dec_memory_Fc) = self.F_lstm2(self.dec_inp_F, (Fhs, Fcs))
        h2 = torch.randn(batchsize, 8, 16)
        c2 = torch.randn(batchsize, 8, 16)

        _, (Chs, Ccs) = self.C_lstm1(self.Input_Seq_C, (h2, c2))

        C_lstm2 = M2LSTM(input_sz=18, hidden_sz=16, activate="relu", num_layers=1)

        dec_outputs_C, (dec_memory_Ch, dec_memory_Cc) = C_lstm2(self.dec_inp_C, (Chs, Ccs))

        adj1 = torch.randn(30, 30)
        adj2 = torch.randn(30, 30)
        adj3 = torch.randn(30, 30)
        adj4 = torch.randn(30, 30)
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

data_dic=data_preprocess()

batch_size=30
train_epoch = 50
learning_rate = 0.01
#data_dic=data_dic.to("cuda")
config={}
config['device'] = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#model = Riskseq(config,data_dic).to(device)
model = Riskseq(config,data_dic)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)



for epoch in range(train_epoch):
    optimizer.zero_grad()
    model.calculate_loss(batch_size).backward()
    optimizer.step()

