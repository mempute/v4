

from morphic import *
from mpi4py import MPI

import numpy as np
import mempute as mp

import  sys
from operator import eq
import matplotlib.pyplot as plt
import time
import math
import datetime

from tools import create_sin, make_timeseries, split_train_test#, visualize_random_data
from tools import batch_loader, shuffle

def print_array(a, fp_name=None, limit=0):
    if fp_name is None: fp = None
    else: fp = open(fp_name, 'w')
    if len(a.shape) == 2:
        batch = a.shape[0]
        seq = a.shape[1]
        i = 0
        if limit == 0 or limit > batch: limit = batch
        while i < batch:
            if i > limit and i < batch - limit: 
                i += 1
                continue
            j = 0
            print("[", end='')
            if fp: fp.write("[")
            while j < seq:
                s = str(a[i, j])
                print(s + " ", end='')
                if fp: fp.write(s + " ")
                j = j + 1
            print("]")
            if fp: fp.write("]\n")
            i = i + 1

    elif len(a.shape) == 3:
        batch = a.shape[0]
        seq = a.shape[1]
        dim = a.shape[2]
        i = 0
        if limit == 0 or limit > batch: limit = batch
        while i < batch:
            if i > limit and i < batch - limit: 
                i += 1
                continue
            j = 0
            print("[  ", end='')
            if fp: fp.write("[  ")
            while j < seq:
                k = 0
                print("[", end='')
                if fp: fp.write("[")
                while k < dim:
                    s = str(a[i, j, k])
                    print(s + " ", end='')
                    if fp: fp.write(s + " ")
                    k = k + 1
                print("]", end='')
                if fp: fp.write("]")
                j = j + 1
            print("   ]\n")
            if fp: fp.write("   ]\n")
            i = i + 1
    else:
        batch = a.shape[0]
        bind = a.shape[1]
        seq = a.shape[2]
        dim = a.shape[3]
        i = 0
        if limit == 0 or limit > batch: limit = batch
        while i < batch:
            if i > limit and i < batch - limit: 
                i += 1
                continue
            b = 0
            print("[  ", end='')
            if fp: fp.write("[  ")
            while b < bind:
                j = 0
                print("[  ", end='')
                if fp: fp.write("[  ")
                while j < seq:
                    k = 0
                    print("[", end='')
                    if fp: fp.write("[")
                    while k < dim:
                        s = str(a[i, b, j, k])
                        print(s + " ", end='')
                        if fp: fp.write(s + " ")
                        k = k + 1
                    print("]", end='')
                    if fp: fp.write("]")
                    j = j + 1
                print("   ]\n")
                if fp: fp.write("   ]\n")
                b = b + 1
            print("   ]batch\n")
            if fp: fp.write("   ]batch\n")
            i = i + 1
    if fp: fp.close()


X_LEN = 31#63#31 #리그레션이면 2의 승수 단위에서 go mark 한개가 모자르게 설정        
Y_LEN = 31  #레벨 또는 nontis를 2의 승수, 즉 32면 5, 64면 6을 설정
data = create_sin()
data_x, data_y = make_timeseries(data, x_size=X_LEN, y_size=Y_LEN)

data_x = np.squeeze(data_x, axis=-1)
data_y = np.squeeze(data_y, axis=-1)

x_train, x_test = split_train_test(data_x)
y_train, y_test = split_train_test(data_y)

print('x_train:', x_train.shape)
print('y_train:', y_train.shape)
print('x_test :', x_test.shape)
print('y_test :', y_test.shape)


embedding = False
auto_regress = True

#np.random.seed(0)
#x_train2 = x_train

if auto_regress:
    xtrain_other = x_train
    z = np.zeros((x_train.shape[0], 1), dtype = x_train.dtype)
    x_train = np.concatenate((z, x_train), axis=1) #go mark
    xtrain_other = np.concatenate((xtrain_other, z), axis=1) #end mark
    xtrain_other = np.expand_dims(xtrain_other, -1)
    dec_infeat_other = 1


inp_size = x_train.shape[-1]
gid = int(sys.argv[2])
model_name = sys.argv[4]
batch_size = 256
sensor = 4.23

def regress_predict(net, x, i):#i는 x 입력 상에서 예측 시작 인덱스, x의 첫번째는 go mark

    n = x.shape[1]
    while i < n:
        x_preds = mp.xpredict(net, x, 1)
        x[:, i] = x_preds[:, i-1]
        i += 1
    return x
"""
x_train = x_train[:2]
print(x_train)
print(x_train.shape)
x_train = np.expand_dims(x_train, -1)
i = int(x_train.shape[1] / 2)
z = np.zeros((x_train.shape[0], x_train.shape[1] - i, x_train.shape[2]), dtype = x_train.dtype)
print(z)
print(z.shape)
x_preds = np.concatenate((x_train[:, :i], z), axis=1) #go mark
print(x_preds)
print(x_preds.shape)
"""

#stmt = get_handle(1, int(sys.argv[3]))

def schema(stmt, percep_name):
    rv = mp.direct(stmt,f"execute mempute('clear', '{percep_name} 0 percep_loc 0 0 all erase 0')")
    rv = mp.direct(stmt, f"execute mempute('perception', '{percep_name} locale percep_loc')")

    rv = mp.direct(stmt, f"execute mempute('sequence', {X_LEN} + 1, {Y_LEN}, 2)") #X_LEN + go mark

    #rv = mp.direct(stmt, "execute mempute('channel', 1, '{u0.0002d}')")
    #rv = mp.direct(stmt, "execute mempute('channel', 0, '{u0.0002d}')")
    rv = mp.direct(stmt, "execute mempute('channel', 1, '{u0.03d}')")
    rv = mp.direct(stmt, "execute mempute('channel', 0, '{u0.03d}')")

if sys.argv[1] == '0':

    rv = mp.direct(stmt, f"execute mempute('perception', '{model_name}')")

    rv = mp.direct(stmt, "execute mempute('phyper', 'bottom_length 5')")
    rv = mp.direct(stmt, "execute mempute('phyper', 'bottom_extract 5')")

    rv = mp.array(stmt, "execute mempute('array', 'eval_input 1 1 0 0 0 0')")

    rv = mp.array(stmt, "execute mempute('array', 'eval_output 0 1 0 0 0 0')")

    sample_x = x_train[0:9]
    sample_y = y_train[0:9]
    
    rv = mp.focus(stmt, "execute mempute('array', 'eval_input')")
    mp.inarray(stmt, sample_x, 1)
    y_preds = mp.mempute(stmt, "execute mempute('predict', 'eval_input', 'eval_output')")
    print(y_preds.shape)
    #y_preds = y_preds.reshape(-1, 1)
    #sample_y = sample_y.reshape(-1, 1)
    
    fig, plots = plt.subplots(3, 3)
    fig.set_figheight(5)
    fig.set_figwidth(15)

    plots = plots.reshape(-1)
    for i, p in enumerate(plots):
        p.plot(y_preds[i])
        p.plot(sample_y[i], color='red')
    plt.show()

elif sys.argv[1] == '9':
    
    net, _ = mp.loadnet(gid, model_name)
 
elif sys.argv[1] == '1':# command gid port perception-name dec_lev_opt mse, ex. 1 0 0 sign 1

    if len(sys.argv) >= 6: dec_lev_opt = int(sys.argv[5])
    else: dec_lev_opt = 0

    if len(sys.argv) >= 7:
        mse = int(sys.argv[6])
    else:
        mse = 0

    #schema(stmt, model_name)

    num_layer = 6
    nydisc = []
    
    train_params = dict(
        input_size=inp_size,
        hidden_sz=32,
        learn_rate=1e-4,
        drop_rate=0.0,
        signid_mse = mse,
        wgt_save = 0,
        layer_norm = 1,
        n_epair = 2,
        residual = 1,
        on_schedule = False,
        dtype = mp.tfloat,
        levelhold = 0.7,
        tunehold = 0.98,
        seed = 777, #-1,
        decay = 0.0001,
        decode_active = 1,
        fast_once = 1,
        nblock = 2,
        nontis = -1,
        gpt_model = 0,
        pos_encoding = 1,
        n_heads = 4,
        d_ff = 64,
        regression = auto_regress,
        dec_lev_learn = dec_lev_opt, #0 if auto_regress is True else dec_lev_opt,
        decode_infeat = dec_infeat_other,
        size_embed = dec_infeat_other + 1 if embedding else 0, #+1 은 패딩 추가
        batch_size=batch_size)

    train_params = default_param(**train_params)
    #train_params['aaaa11'] = 0.5 #rfilter
    #train_params['aaaa12'] = True 
    #train_params['least_accuracy'] = 0.0
    #train_params['aaa8'] = 0 #approximate
    #train_params['aaaa14'] = 1 #conv pos
    #train_params['aaaa15'] =-1#0#-1
    #train_params['aaaa16'] = 64
    #train_params['aaaa18'] = 8 
    #train_params['aaaa19'] = 4 
    #train_params['aaaaa10'] = 1
    train_params['aaaaa11'] = 0
    #train_params['aaaaa12'] = 2 #incre_lev
    #train_params['aaaa17'] = 0 #oneway
    #train_params['aaaaa13'] = 1
    #train_params['aaaaa14'] = 4
    #train_params['aaaaa15'] = 1
    train_params['aaaaa16'] = 0#-10 #npk
    #train_params['aaaaa17'] = 1e-3#0.001 #spk_lr
    train_params['aaaaa18'] = 1.0 #reduce_svar
    #train_params['aaaaa19'] = 4 #glam
    train_params['aaaaaa10'] = 0.5 #rspk_lr
    #train_params['aaaaaa11'] = 1 #monitor
    #train_params['aaaaa15'] = 1
    #train_params['aaaaaa13'] = 8
    #train_params['aaaaaa14'] = 50 #sampling

    train_params['aaaaaa15'] = 0#8
    train_params['aaaaaa16'] = 0.1
    train_params['aaaaaa17'] = 0.8
    train_params['aaaaaa18'] = 65536 / 2
    train_params['aaaaaa19'] = 32
    train_params['aaaaa19'] = 0 
    train_params['aaaaaa14'] = -1 
    train_params['aaaaaa11'] = 1 #monitor

    train_params['aaaaa12'] = 0#8 
    
    train_params['aab0'] = 2 
    train_params['aab1'] = 8 

    param = param2array(**train_params)
    if param is None:
        exit()
    net = mp.neuronet(param, gid, model_name)
    #mp.neurogate(net, stmt, model_name, sensor, x_train, y_train, 1)
    mp.close_net(net)
else:
    param = mp.load_param(model_name)
    train_params = arr2py_param(param)
    #train_params['drop_rate'] = 0.0
    #train_params['aaaa13'] = 0 #prePooling
    #train_params['aaa7'] = 0 #multis
    #train_params['seed'] = -1
    #train_params['aaaa12'] =-1
    #train_params['hidden_sz'] = 1
    param = param2array(**train_params)
    mp.update_param(param, model_name)
    
    net = mp.loadnet(gid, model_name)

    #mp.neurogate(net, stmt, model_name, sensor, x_train, y_train, 0)

    if sys.argv[1] == '2':


        level = int(sys.argv[5])
        epoch = int(sys.argv[6])
        if len(sys.argv) == 8: relay = int(sys.argv[7])
        else: relay = 0

        if level: tuning = 0
        else: 
            tuning = 1
            level = 1

        reent = np.expand_dims(x_train, -1)
        for i in range(level):
            reent = mp.xtrain(net, reent, xtrain_other, tuning, epoch, relay)

        
    elif sys.argv[1] == '5':   
        
        level = int(sys.argv[5])
        epoch = int(sys.argv[6])
        if len(sys.argv) >= 8: relay = int(sys.argv[7])
        else: relay = 0
        print(f'level: {level} epoch: {epoch}')
        
        if len(sys.argv) == 9: decord_mode = int(sys.argv[8])
        else: decord_mode = 1 

        if level: 
            tuning = 0 
        else: 
            tuning = 1
            level = 1

        reent = np.expand_dims(x_train, -1)
        for i in range(level):
            reent = mp.xtrain(net, reent, xtrain_other, tuning, epoch, relay, decord_mode)


        if relay < 0 or (relay > 0 and relay < 3):#이어서 학습 모드이면 cleo를 해제 반납하지 
            rv = mp.direct(stmt, "execute mempute('remain')")#않게 한다.
            exit(0)
            
        #x_test = x_train2

        if auto_regress:
            z = np.zeros((x_test.shape[0], 1), dtype = x_test.dtype)
            x_test = np.concatenate((z, x_test), axis=1) #go mark

        #np.random.seed(0)

        N = x_test.shape[0]
        idices = np.random.choice(np.arange(N), size=9, replace=False)

        result_pred = []
        result_true = []

        x_test = np.expand_dims(x_test, -1)
        i = int(x_test.shape[1] / 2)
        z = np.zeros((1, x_test.shape[1] - i, x_test.shape[2]), dtype = x_train.dtype)

        for idx in idices:
            sample_x = x_test[idx:idx+1]
            x_preds = np.concatenate((sample_x[:, :i], z), axis=1)
            x_preds = mp.xpredict(net, x_preds, 1, i) #x_preds = regress_predict(net, x_preds, i)
            sample_x = sample_x[:,1:]#go mark cut
            x_preds = x_preds[:,1:]#go mark cut

            #print(x_preds.shape)
            #print(x_preds)

            x_preds = x_preds.reshape(-1, 1)
            result_pred.append(x_preds)
            result_true.append(sample_x.reshape(-1, 1))

        fig, plots = plt.subplots(3, 3)
        fig.set_figheight(5)
        fig.set_figwidth(15)

        plots = plots.reshape(-1)
        for i, p in enumerate(plots):
            p.plot(result_true[i])
            p.plot(result_pred[i], color='red')
        plt.show()  
        print('end')

    elif sys.argv[1] == '6':  

        if auto_regress:
            z = np.zeros((x_test.shape[0], 1), dtype = x_test.dtype)
            x_test2 = np.concatenate((z, x_test), axis=1) #go mark

        #np.random.seed(0)

        N = x_test2.shape[0]
        idices = np.random.choice(np.arange(N), size=9, replace=False)

        result_pred = []
        result_true = []

        x_test2 = np.expand_dims(x_test2, -1)
        i = int(x_test2.shape[1] / 2)
        z = np.zeros((1, x_test2.shape[1] - i, x_test2.shape[2]), dtype = x_train.dtype)

        for idx in idices:
            sample_x = x_test2[idx:idx+1]
            x_preds = np.concatenate((sample_x[:, :i], z), axis=1)
            x_preds = mp.xpredict(net, x_preds, 1, i) #x_preds = regress_predict(net, x_preds, i)
            sample_x = sample_x[:,1:]#go mark cut
            x_preds = x_preds[:,1:]#go mark cut

            #print(x_preds.shape)
            #print(x_preds)

            x_preds = x_preds.reshape(-1, 1)
            result_pred.append(x_preds)
            result_true.append(sample_x.reshape(-1, 1))

        fig, plots = plt.subplots(3, 3)
        fig.set_figheight(5)
        fig.set_figwidth(15)
        
        plots = plots.reshape(-1)
        for i, p in enumerate(plots):
            p.plot(result_true[i])
            p.plot(result_pred[i], color='red')
        plt.show()  
        print('end')

