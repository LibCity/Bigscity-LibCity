from torch.utils.data import Dataset, DataLoader
import torch
import pandas as pd
import numpy as np
import os
def dataprocess():
    T_start = 9000
    T_end = 20000

    Total_Num = 21744
    # l = ['./Dataset/Acc_sum.csv', './Dataset/Taxipick_sum.csv', './Dataset/Taxidrop_sum.csv', './Dataset/Taxi_pick_diff.csv', './Dataset/Speed_sum_fillFM.csv',
    #      './Dataset/Y_Coarse_Risk.csv', './Dataset/static_affinity.csv','./Dataset/ex_time_stamp_10min.csv']
    #
    # Acc_total = np.delete(pd.read_csv(l[0]).to_numpy(),0,axis=1)
    # Pickup_total = np.delete(pd.read_csv(l[1]).to_numpy(),0,axis=1)
    # Dropoff_total = np.delete(pd.read_csv(l[2]).to_numpy(),0,axis=1)
    # Taxi_pickd_total = np.delete(pd.read_csv(l[3]).to_numpy(),0,axis=1)
    # Speed_total = np.delete(pd.read_csv(l[4]).to_numpy(),0,axis=1)
    # Y_coarse = np.delete(pd.read_csv(l[5]).to_numpy(),0,axis=1)
    # External_total = np.delete(pd.read_csv(l[7]).to_numpy(),0,axis=1)
    # Static_affinity = np.delete(pd.read_csv(l[6]).to_numpy(),0,axis=1)

    p1 = pd.read_csv("./risk_seq.rel", usecols=[2, 3, 4])
    Static_affinity = np.zeros(shape=(354, 354))
    for i in range(354 * 354):
        Static_affinity[p1['origin_id'][i]][p1['destination_id'][i]] = p1['affinity'][i]

    Acc_total = np.zeros(shape=(354, 21744))
    Pickup_total = np.zeros(shape=(354, 21744))
    Dropoff_total = np.zeros(shape=(354, 21744))
    Taxi_pickd_total = np.zeros(shape=(354, 21744))
    Speed_total = np.zeros(shape=(354, 21744))
    Y_coarse = np.zeros(shape=(18, 21744))
    External_total = np.zeros(shape=(354, 21744))
    Acc_prob_negnew = np.zeros(shape=(354,1))
    p_dyna = pd.read_csv("./risk_seq.dyna", usecols=[2, 3, 4, 5, 6, 7, 8, 9])
    p_dyna2 = pd.read_csv("./risk_seq2.dyna", usecols=[2])
    for i in range(354 * 21744):
        Acc_total[i // 21744][i % 21744] = p_dyna['acc_sum'][i]
        Pickup_total[i // 21744][i % 21744] = p_dyna['taxi_pick'][i]
        Dropoff_total[i // 21744][i % 21744] = p_dyna['taxi_trop'][i]
        Taxi_pickd_total[i // 21744][i % 21744] = p_dyna['taxi_diff'][i]
        Speed_total[i // 21744][i % 21744] = p_dyna['speed'][i]

    for i in range(18 * 21744):
        Y_coarse[i // 21744][i % 21744] = p_dyna2['risk'][i]

    for i in range(354): #前354行，后面置0
        Acc_prob_negnew[i][0] = p_dyna['acc_new'][i]


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
        return Input_Batch, Matrix_Aff, Output_Risk, Input_Batch_Ycoarse, Output_Ycoarse, Output_Ext, Output_AccSUM

    def generate_samples(t1, t2, batch_size):
        tt = np.random.choice(range(t1, t2), batch_size)
        #     tt = np.array([10287, 10289, 10293, 10296, 10298, 10302,
        #        10304, 10305, 10307, 10309, 10311, 10313, 10315, 10316, 10318,
        #        10320, 10322, 10323, 10325, 10327, 10329, 10330, 10332, 10334,
        #        10336, 10338, 10341, 10342, 10343, 10345])
        Input_Batch = []
        Intput_Ycoarse_B = []

        Matrix_Aff_B = []
        Output_Risk_B = []
        Output_Ycoarse_B = []
        Output_AccSUM_B = []
        Output_Ext_B = []
        for i in tt:
            Input_Seq, Matrix_Aff, Output_Risk, input_seq_Ycoarse, Output_Ycoarse, Output_Ext, Output_AccSUM = Construct_inp_out(
                i, 6)
            Input_Batch.append(Input_Seq)

            Matrix_Aff_B.append(Matrix_Aff)
            Output_Risk_B.append(Output_Risk)
            Output_Ycoarse_B.append(Output_Ycoarse)
            Output_Ext_B.append(Output_Ext)
            Intput_Ycoarse_B.append(input_seq_Ycoarse)
            Output_AccSUM_B.append(Output_AccSUM)
        Input_Batch = np.array(Input_Batch)

        Matrix_Aff_B = np.array(Matrix_Aff_B)
        Output_Risk_B = np.array(Output_Risk_B)
        Intput_Ycoarse_B = np.array(Intput_Ycoarse_B)
        Output_Ycoarse_B = np.array(Output_Ycoarse_B)
        Output_Ext_B = np.array(Output_Ext_B)
        Output_AccSUM_B = np.array(Output_AccSUM_B)
        return Input_Batch, Matrix_Aff_B, Output_Risk_B, Intput_Ycoarse_B, Output_Ycoarse_B, Output_Ext_B, Output_AccSUM_B, tt

    Batchsize = 30
    Input_Batch,  Matrix_Aff, Output_Seq, Input_Ycoarse, Output_Ycoarse, Ext_O, Acc_SumS, SelectedTime = generate_samples(
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

    datadic = {}
    datadic['Dynamic_Feature'] = Dynamic_Feature
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