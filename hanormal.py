
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from scipy import stats
import mempute as mp
import seaborn as sns
from pylab import rcParams
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import  sys
import time
import json
import os

from morphic import *

bos = 6
pos = 5

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


def convert_row_first(d_type, aarray):
    n_input_row = aarray.shape[0]
    n_input_col = aarray.shape[1]
    output_arr = np.full((int(n_input_row), n_input_col), pos, dtype = d_type)
    in_row = 0
    while in_row < n_input_row:
        output_arr[in_row,:] = aarray[in_row,:]
        in_row += 1
    return output_arr

def data_standardization(x):
    x_np = np.asarray(x)
    return (x_np - x_np.mean()) / x_np.std()

def reverse_standardization(org_x, x):
    org_x_np = np.asarray(org_x)
    x_np = np.asarray(x)
    return (x_np * org_x_np.std() + org_x_np.mean())

def min_max_scaling(x):
    x_np = np.asarray(x)
    return (x_np - x_np.min()) / (x_np.max() - x_np.min() + 1e-7) # 1e-7은 0으로 나누는 오류 예방차원

# 정규화된 값을 원래의 값으로 되돌린다
# 정규화하기 이전의 org_x값과 되돌리고 싶은 x를 입력하면 역정규화된 값을 리턴한다
def reverse_min_max_scaling(org_x, x):
    org_x_np = np.asarray(org_x)
    x_np = np.asarray(x)
    return (x_np * (org_x_np.max() - org_x_np.min() + 1e-7)) + org_x_np.min()

#get_ipython().run_line_magic('matplotlib', 'inline')

sns.set(style='whitegrid', palette='muted', font_scale=1.5)

rcParams['figure.figsize'] = 14, 8

RANDOM_SEED = 42
LABELS = ["Normal", "Fraud"]

np.set_printoptions(precision=4, suppress=True)

df = pd.read_csv("anormdata/creditcard.csv")

df.isnull().values.any()

from sklearn.preprocessing import StandardScaler

#print(df.describe())

data = df.drop(['Time'], axis=1)
#data['Class'] = 1

data['Amount'] = StandardScaler().fit_transform(data['Amount'].values.reshape(-1, 1))



deatt = 0
nblock = 1
n_epair = 2
hidden_sz = 32
gpt_model = -1
pos_enc = 1
n_heads = 4
ffn_hidden = 256
qa_kernel = 0#8
precision = 0.2
qlearn_lev = 0
qgen_lev = 0
cut_over = 0.8
infini_kernel = 0

batch_size = 32
auto_regress = 1
decord_optimize = 1
small_test = 0
test_predict = 1
auto_morphic = False
decode_learn = 1
inner_mul = 5
outer_mul = 13
jit_count = 60#27
dec_lev_opt = 1
embedding = False
decode_xother = False
idpred_print = 0
dense_data = 0
xdisc_adopt = 0

if embedding:
    decode_xother = True
dx_mul = 10
if decode_xother:
    tot = data.values
    minv = np.min(tot)
    maxv = np.max(tot)
    print(minv)
    print(maxv)
    minv *= dx_mul
    maxv *= dx_mul
    print(minv)
    print(maxv)
    minv = int(minv)
    maxv = int(maxv)
    dec_infeat_other = maxv - minv + 1
    print(dec_infeat_other)
else:
    dec_infeat_other = 0

x_train, x_test = train_test_split(data, test_size=0.2, random_state=RANDOM_SEED)


#pick out -> target 학습(케이스 b) 설정 요소
sparse_dup = 1 #pick out -> target 학습때 pick out된 데이터가 중복이 거의 없어 데이터를 중복 입력시킬 경우
just_pick = 0 #pick out를 소수점 일정이하 절삭하여 선형이 아닌 이산값으로 pick out -> target 학습시킬 경우

pickid = []
pick_out = 1
set_nan = 1
nan_val = 0


if pick_out: pickid_num = 15#28
else: pickid_num = 5

if small_test == 1:
    x_train = x_train.iloc[:64]#3000 이면 u0.004d, 1000 이면 u0.02d
    x_test = x_test.iloc[:64]
    sensor = 2.7
else:
    sensor = 4.24
    if small_test == 2:
        x_train = x_train.iloc[:20000]
        x_test = x_test.iloc[:20000]

if idpred_print: 
    x_train = x_train.iloc[:1000]
    x_test = x_test.iloc[:1000]


"""
df['salary'] = 0
df['salary'] = np.where(df['job'] != 'student', 'yes', 'no')
for x in range(150, 200, 10):
    start = x
    end = x+10
    temp = df[(df["height"] >= start) & (df["height"] < end)]
    print("{}이상 {}미만 : {}".format(start, end, temp["height"].mean()))
"""


if (decord_optimize and int(sys.argv[1]) >= 1 and int(sys.argv[1]) <= 6) or (decord_optimize == 0 and int(sys.argv[1]) >= 5 and int(sys.argv[1]) <= 6):
    x_train = x_train[x_train.Class == 0]
    set_nan = 0
    print('only normal case training')
else: print('mix training')

y_train = x_train['Class']
x_train = x_train.drop(['Class'], axis=1)


Y_test = x_test['Class']
x_test = x_test.drop(['Class'], axis=1)


x_train = x_train.values
x_test = x_test.values

y_train = y_train.values
y_test = Y_test.values
"""
#데이터를 양수 값을 만들기 위해 최소값을 더해 준다. 안해도 된다.
if sys.argv[1] is not '5':
    x_train = x_train - minv
    x_test = x_test - minv
"""
y_train = np.expand_dims(y_train, axis=-1)
y_test = np.expand_dims(y_test, axis=-1)

x_train = convert_row_first("f", x_train)
x_test = convert_row_first("f", x_test)
y_train = convert_row_first("f", y_train)
y_test = convert_row_first("f", y_test)

#x_train = min_max_scaling(x_train)
#x_test = min_max_scaling(x_test)

#make 31 seq
z = np.full((x_train.shape[0], 31 - x_train.shape[1]), pos, dtype = x_train.dtype)
x_train = np.concatenate((x_train, z), axis=1) 
z = np.full((x_test.shape[0], 31 - x_test.shape[1]), pos, dtype = x_test.dtype)
x_test = np.concatenate((x_test, z), axis=1) 

print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)

print(x_train)
#print(y_train)

if set_nan and nan_val == 0: #신경망 학습을 위해 난값을 0로 대체하는 경우 타겟값이 0과 1인데
    y_train = y_train + 1       #0값이 난값 처리되므로 이를 피하기위해 +1을 한다.
if test_predict:
    if set_nan and nan_val == 0: 
        y_test = y_test + 1
else:
    x_test = x_train
    y_test = y_train

"""
print('TRAIN DATA')
for i, v in enumerate(y_train):
    if v == 1: print("index : {}, value: {}".format(i,v))
print()

print('TEST DATA')
for i, v in enumerate(y_test):
    if v == 1: print("index : {}, value: {}".format(i,v))
print()
"""
#x_train = np.transpose(x_train)
#y_train = np.transpose(y_train)

#predictions = autoencoder.predict(x_test)

if auto_morphic: #뉴로모픽 타겟 연결 학습할때만 의미있고 입력을 타겟으로 설정한다.
    y_train = x_train
    decode_learn = 0

import math
def truncate(num, r) -> float:
    d = 10.0 ** r
    return math.trunc(num * d) / d

def drop_nan(arr, data_x, data_y, r):
    nmax = 0
    ra, rx, ry = [], [], []
    for a, x, y in zip(arr, data_x, data_y):
        i = 0
        for j in range(arr.shape[1]):
            if np.isnan(a[j]) == 0:
                if r > 0: a[i] = truncate(a[j], r)
                else: a[i] = a[j]
                i += 1
            a[j] = np.nan if set_nan == 0 else nan_val #pick out을 신경망에 입력으로 학습시킬려면 난값은 에러나므로
        if i:
            ra.append(a)
            rx.append(x)
            ry.append(y)
            if i > nmax: nmax = i
    ra = np.array(ra)
    rx = np.array(rx)
    ry = np.array(ry)
    return nmax, ra, rx, ry
"""
def drop_nan(arr, r):
    nmax = 0
    for row in arr:
        i = 0
        for j in range(arr.shape[1]):
            if np.isnan(row[j]) == 0:
                if r > 0: row[i] = truncate(row[j], r)
                else: row[i] = row[j]
                i += 1
            row[j] = np.nan
        if i > nmax: nmax = i
    return nmax
"""


def get_rxid(data_x, figure): #외부 입력 데이터의 차원 축소 아이디 리턴
    rv = mp.direct(stmt, "execute mempute('perception', 'anormal')")

    s = "execute mempute('discriminate', 0, {})".format(data_x.shape[1])
    rv = mp.mempute(stmt, s) #최종레벨 추출 패턴 시퀀스 길이 설정

    rv = mp.array(stmt, "execute mempute('array', 'eval_input 1 1 0 0 0 0')")
    mp.inarray(stmt, data_x, 1)

    rv = mp.array(stmt, "execute mempute('array', 'disc_output 0 1 0 0 0 0')")

    if figure:
        s = "execute mempute('discriminate', 'eval_input', 100, {}, 'disc_output')".format(figure)
        r = mp.mempute(stmt, s)
        r = r[0]
        return r[0]
    else:
        r = mp.mempute(stmt, "execute mempute('discriminate', 'eval_input', 101, 0, 'disc_output')")
        r = r.astype(np.float32)
        return r

def sizeof_rxid():
    rv = mp.direct(stmt, "execute mempute('perception', 'anormal')")
    r = mp.mempute(stmt, "execute mempute('discriminate', 102, -1)")
    
    r = r[0]
    return r[0]

def sizeof_signiden():
    rv = mp.direct(stmt, "execute mempute('perception', 'anormal')")
    r = mp.mempute(stmt, "execute mempute('discriminate', 2, -1)")
    
    #print(r)#anormal iden size print
    r = r[0]
    return r[0]

def get_signiden(data_x):

    assert pick_out == 0

    rv = mp.direct(stmt, "execute mempute('perception', 'anormal')")

    rv = mp.direct(stmt, "execute mempute('phyper', 'revise_infer 1')")

    rv = mp.mempute(stmt, f"execute mempute('discriminate', 0, {pickid_num})") #최종레벨 추출 패턴 시퀀스 길이 설정

    rv = mp.array(stmt, "execute mempute('array', 'eval_input 1 1 0 0 0 0')")
    mp.inarray(stmt, data_x, 1)

    rv = mp.array(stmt, "execute mempute('array', 'disc_output 0 1 0 0 0 0')")

    r = mp.mempute(stmt, "execute mempute('discriminate', 'eval_input', 1, 0, 'disc_output')")
    print('get anormal iden shape: ', r.shape)
    r = r.astype(np.float32)
    return r

def inference_signiden(signid, init):

    assert pick_out == 0

    if init:
        rv = mp.direct(stmt, "execute mempute('perception', 'anormal')")
    #else:#위 함수 호출후에 세션 종료없이 현 퍼셉션에서 수행할때
    mp.mempute(stmt, f"execute mempute('discriminate', 1, {pickid_num})") #입력정보 변경, 타겟 채널 정보 복원(위 함수 
                                                                          #호출후 라면, 아니면 의미 없음)
    rv = mp.array(stmt, "execute mempute('array', 'signid_input 1 1 0 0 0 0')")
    mp.inarray(stmt, signid, 1)

    rv = mp.array(stmt, "execute mempute('array', 'eval_output 0 1 0 0 0 0')")

    r = mp.mempute(stmt, "execute mempute('predict', 'signid_input', 'eval_output')")

    return r

def get_pickout(data_x, data_y): #pick out 수행

    assert pick_out

    rv = mp.direct(stmt, "execute mempute('perception', 'anormal')")

    rv = mp.direct(stmt, "execute mempute('phyper', 'regularize_rate 1000')")

    rv = mp.array(stmt, "execute mempute('array', 'anormal_input')")
    mp.inarray(stmt, data_x, 1)

    rv = mp.array(stmt, "execute mempute('array', 'anormal_target')")
    mp.inarray(stmt, data_y, 1)

    rv = mp.array(stmt, "execute mempute('array', 'anormal_output 1 1 0 0 0 0')")

    #rv = mp.mempute(stmt, "execute mempute('discriminate', 4, 2)")

    pickout = mp.mempute(stmt, "execute mempute('discriminate', 'anormal_input', 103, 1, 'anormal_output', 'anormal_target')")
    #pickout = mp.mempute(stmt, "execute mempute('discriminate', 'anormal_input', 103, 1, 'anormal_output')")
    print(pickout.shape)

    #print_array(pickout, 'bb')
    print('pick out src shape: ', data_x.shape)
    nmax_col, pickout, data_x, data_y = drop_nan(pickout, data_x, data_y, 3 if just_pick else 0)

    if nmax_col > pickid_num: nmax_col = pickid_num

    pickout = pickout[:,:nmax_col]
    #print_array(pickout, 'aa')
    print('pick out max col: ', nmax_col, 'shape: ', pickout.shape)

    return nmax_col, pickout, data_x, data_y

def inference_reinforce(pick_id, init):

    assert pick_out

    if init:
        rv = mp.direct(stmt, "execute mempute('perception', 'reinforce')")

    rv = mp.array(stmt, "execute mempute('array', 'reinforce_input 1 1 0 0 0 0')")
    mp.inarray(stmt, pick_id, 1)
    
    rv = mp.array(stmt, "execute mempute('array', 'reinforce_output 0 1 0 0 0 0')")
    #rv = mp.direct(stmt, "execute mempute('display', -3)")
    r = mp.mempute(stmt, "execute mempute('predict', 'reinforce_input', 'reinforce_output')")

    return r

def predict_present(v_right, predictions, net, present, mse=None, ppmax=0):
    threshold = 1000000
    if set_nan and nan_val == 0: 
        v_right = v_right - 1
        predictions = predictions -1

    nan_cnt = norm_norm = norm_proud = proud_norm = proud_proud = 0
    prev_v = np.zeros((predictions.shape[0], predictions.shape[1]), dtype = 'i')
    for i, (y, p) in enumerate(zip(v_right, predictions)):
        if y == 0:
            if p > 0.5 or np.isnan(p): 
                prev_v[i] = 1
                norm_proud += 1
                if np.isnan(p): nan_cnt += 1
                if present: print("index : {} normal:proud nan: {}".format(i, np.isnan(p)))
            else: norm_norm += 1
        else:
            if p > 0.5 or np.isnan(p): 
                prev_v[i] = 1
                proud_proud += 1
                if np.isnan(p): nan_cnt += 1
                if present: print("index : {} proud:proud nan: {}".format(i, np.isnan(p)))
                if mse is not None and threshold > mse[i]:
                    threshold = mse[i] - 1e-7
            else:
                proud_norm += 1
                if present: print("index : {} proud:normal".format(i))

    s = "normal-normal : {} normal-proud : {} proud:normal : {} proud:proud : {} nan cnt : {}".format(norm_norm, norm_proud, proud_norm, proud_proud, nan_cnt)
    print(s)
    if net is not None: mp.print_log(net, s)
    
    accuracy = (norm_norm + proud_proud) / (norm_norm + proud_proud + norm_proud + proud_norm)
    precision = norm_norm / (norm_norm + proud_norm + 1e-7)
    recall = norm_norm / (norm_norm + norm_proud + 1e-7)
    f1 = 2 * ((precision * recall) / (precision + recall + 1e-7))
    s = f"accuracy: {accuracy} precision: {precision} recall: {recall} f1 score: {f1}"
    print(s)
    if net is not None: mp.print_log(net, s)

    precision = proud_proud / (proud_proud + proud_norm + 1e-7)
    recall = proud_proud / (proud_proud + norm_proud + 1e-7)
    f1 = 2 * ((precision * recall) / (precision + recall + 1e-7))
    s = f"reverse precision: {precision} recall: {recall} f1 score: {f1}"
    print(s)
    if net is not None: mp.print_log(net, s)

    if mse is None or proud_proud < ppmax:
        threshold = 0

    if present < 0:
        return threshold
    #predictions = predictions.astype('int32')
    predictions = prev_v

    #for i, v in enumerate(predictions):
    #    if v > 0.5: predictions[i] = 1
    #    else: predictions[i] = 0
    conf_matrix = confusion_matrix(v_right, predictions)

    plt.figure(figsize=(12, 12))
    sns.heatmap(conf_matrix, xticklabels=LABELS, yticklabels=LABELS, annot=True, fmt="d");
    plt.title("Confusion matrix")
    plt.ylabel('True class')
    plt.xlabel('Predicted class')
    plt.show()
    return threshold

def discriminator(ytag, vright, predictions, mul, mse = None, threshold = 0):
    if mse is None:
        mse = np.mean(np.power(vright - predictions, 2), axis=1)#[batch, seq]
    if threshold == 0:
        rcount = 0
        for r in ytag:
            if r[0] == 1: rcount += 1
        rcount = int(rcount * mul)
        mse2 = sorted(mse, reverse=True)
        threshold = mse2[rcount]
            
    y_pred = [1 if e > threshold else 0 for e in mse]
    y_pred = np.array(y_pred)
    y_pred = np.expand_dims(y_pred, axis=-1)

    return y_pred, threshold, mse

def reconst_present(x_test, v_right, predictions):

    if set_nan and nan_val == 0: 
        v_right = v_right - 1
        predictions = predictions -1

    mse = np.mean(np.power(x_test - predictions, 2), axis=1)

    y_pred = [1 if e > threshold else 0 for e in mse]
    y_pred = np.array(y_pred)
    y_pred = np.expand_dims(y_pred, -1)
    conf_matrix = confusion_matrix(v_right, y_pred)

    plt.figure(figsize=(12, 12))
    sns.heatmap(conf_matrix, xticklabels=LABELS, yticklabels=LABELS, annot=True, fmt="d");
    plt.title("Confusion matrix")
    plt.ylabel('True class')
    plt.xlabel('Predicted class')
    plt.show()

xtrain_other = None
seq_len = x_train.shape[1]
if auto_regress or decode_xother is False:
    seq_len = seq_len + 1
    xtrain_other = x_train
    z = np.full((x_train.shape[0], 1), bos, dtype = x_train.dtype)
    x_train = np.concatenate((z, x_train), axis=1) #go mark
    print(x_train.shape)
    xtrain_other = np.concatenate((xtrain_other, z), axis=1) #end mark
    xtrain_other = np.expand_dims(xtrain_other, -1)
    dec_infeat_other = 1

if decode_xother:
    xtrain_other = x_train * dx_mul
    xtest_other = x_test * dx_mul
    xtrain_other = xtrain_other.astype(np.int32)
    xtest_other = xtest_other.astype(np.int32)
    xtrain_other = xtrain_other.astype(x_train.dtype)
    xtest_other = xtest_other.astype(x_train.dtype)
    #print_array(x_train, None, 10)
    #print_array(xtrain_other, None, 10)
    xtrain_other = xtrain_other - minv
    xtest_other = xtest_other - minv
    #print_array(xtrain_other, None, 10)
    if embedding:
        x_train = xtrain_other
        x_train = x_train.astype(np.int64)
        x_test = xtest_other
        x_test = x_test.astype(np.int64)
    xtrain_other = np.expand_dims(xtrain_other, -1)

inp_size = x_train.shape[-1]
gid = int(sys.argv[2])
model_name = sys.argv[4]



def make_mname(model_name):
    model_name = f"{model_name}_isz_{inp_size}_hsz_{hidden_sz}\
                _npair_{n_epair}\
                _ar_{auto_regress}_nblock_{nblock}\
                _nhead_{n_heads}\
                _dff_{ffn_hidden}_gpt_{gpt_model}\
                _posenc_{pos_enc}\
                _deatt_{deatt}_dec_infeat_other_{dec_infeat_other}\
                _precision_{precision}_cut_over_{cut_over}\
                _qlearn_lev_{qlearn_lev}_qgen_lev_{qgen_lev}\
                _qak_{qa_kernel}_ifk_{infini_kernel}"
    model_name = model_name.replace(' ', '')
    print(model_name)
    return model_name

model_name = make_mname(model_name)

if sys.argv[1] == '1':# command gid port perception-name decode_lev_opt mse, ex. 1 0 0 anormal 1 1
    
    if len(sys.argv) >= 6:
        dec_lev_opt = int(sys.argv[5]) 


    if len(sys.argv) == 7: mse = int(sys.argv[6])
    else: mse = 0

    train_params = dict(
        input_size=inp_size,
        hidden_sz=hidden_sz,
        learn_rate=1e-4,
        drop_rate=0.1,
        signid_mse = mse,
        wgt_save = 0,
        layer_norm = 1,
        n_epair = n_epair,
        residual = 1,
        on_schedule = False,
        dtype = mp.tfloat,
        levelhold = 0.7,
        tunehold = 0.98,
        seed = 777, #-1,
        decay = 0.0001,
        decode_active = 1,
        fast_once = 1,
        nblock = nblock,
        nontis = -1,
        gpt_model = gpt_model,
        pos_encoding = pos_enc,
        n_heads = n_heads,
        d_ff = ffn_hidden,
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
    #train_params['aaaa13'] = 1 
    #train_params['aaaa15'] =-1#0#-1
    if auto_regress == 0:
        train_params['aaaa16'] = 64 #차원 고정 안되게
    #train_params['aaaa18'] = 8 
    #train_params['aaaa19'] = 4 
    #train_params['aaaaa10'] = 1
    train_params['aaaaa11'] = deatt #deatten
    #train_params['aaaaa12'] = 2 #incre_lev
    train_params['aaaa17'] = 0 #oneway
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

    train_params['aaaaaa15'] = qa_kernel
    train_params['aaaaaa16'] = precision
    train_params['aaaaaa17'] = cut_over
    train_params['aaaaaa18'] = 48000
    train_params['aaaaaa19'] = 32
    train_params['aaaaa19'] = qlearn_lev 
    train_params['aaaaaa14'] = qgen_lev 
    train_params['aaaaaa11'] = 1 #monitor

    train_params['aaaaa12'] = infini_kernel

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


        #python hanormal.py 1 0 0 anormal 1 1 - 초기화
        #python hanormal.py 5 0 0 anormal 0 2  

        level = int(sys.argv[5]) #레벨 학습 단계 설정, 1부터 시작
        epoch = int(sys.argv[6])
        if len(sys.argv) >= 8: relay = int(sys.argv[7])
        else: relay = 0

        if len(sys.argv) >= 9: tune_level = int(sys.argv[8])
        else: tune_level = 1

        if len(sys.argv) == 10: repeat = int(sys.argv[9])
        else: repeat = 1

        if level: tuning = 0 
        else: 
            tuning = 1
            level = 1
        #print_array(x_train, "train")
        s = f"\nhanormal train:  EPOCH {epoch} RELAY {relay} LEVEL {level} REPEAT {repeat} TUNING {tuning} DECODE_LEV_LEARN {train_params.get('dec_lev_learn')}\n"
        print(s)
        mp.print_log(net, s)
        count = 0
        if repeat > 1: relay = 0
        while count < repeat:
            present = 10 if count + 1 == repeat else -1 #마지막 반복에서만 프린트
            reent = np.expand_dims(x_train, -1)
            for i in range(level):
                name = f'aaa_{i}'
                reent = mp.xtrain(net, reent, xtrain_other, tuning, epoch, relay, 0, name, present)

            x_test1 = np.expand_dims(x_test, -1)
            predictions = mp.xpredict(net, x_test1, 0, -1, 'bbb', present)

            print(x_test1.shape)
            print(y_test.shape)
            print(predictions.shape)
        
            if auto_morphic:
                y_pred, threshold, mse = discriminator(y_test, x_test, predictions, inner_mul)
                s = f'{repeat} REPEAT: {count} THRESHOLD: {threshold} AUTO MORHPIC'
                print(s)
                mp.print_log(net, s)
            else:
               mse = None
            threshold = predict_present(y_test, y_pred, net, present, mse, jit_count)
            if auto_morphic and threshold > 0:
                y_pred, _, _ = discriminator(y_test, x_test, predictions, inner_mul, mse, threshold)
                s = f'optim THRESHOLD: {threshold} AUTO MORHPIC'
                print(s)
                mp.print_log(net, s)
                predict_present(y_test, y_pred, net, present)

                y_pred, threshold, _ = discriminator(y_test, x_test, predictions, outer_mul, mse)
                s = f'outer THRESHOLD: {threshold} AUTO MORHPIC'
                print(s)
                mp.print_log(net, s)
                threshold = predict_present(y_test, y_pred, net, present, mse)

                y_pred, _, _ = discriminator(y_test, x_test, predictions, outer_mul, mse, threshold)
                s = f'outer optim THRESHOLD: {threshold} AUTO MORHPIC'
                print(s)
                mp.print_log(net, s)
                predict_present(y_test, y_pred, net, present)

            count += 1
            if present == 0:#반복 중간이면 최종 레벨 이상 삭제
                mp.truncate(net, tune_level, 0)

    elif sys.argv[1] == '5':#anormal.py 5 0 0 anormal 0 10
        assert train_params.get('decode_active') > 0, "decode active setting lack"
        level = int(sys.argv[5]) #레벨 학습 단계 설정, 1부터 시작
        epoch = int(sys.argv[6])
        if len(sys.argv) >= 8: repeat = int(sys.argv[7])
        else: repeat = 1

        if len(sys.argv) >= 9: relay = int(sys.argv[8])
        else: relay = 0 

        if len(sys.argv) == 10: decord_mode = int(sys.argv[9])
        else: decord_mode = 1 #현수행이 디코드 레벨학습이면 true(1,2,3) or false(0)만 의미있다

        if level: 
            tuning = 0 #레벨 단위 학습의 반복 레밸 횟수가 주어졋으면 레벨단위 학습, 1이면 한개 레벨만 학습
            if (auto_regress or decode_xother is False) and (relay == 3 or relay == 4):
                z = np.full((x_test.shape[0], 1), bos, dtype = x_test.dtype)
                x_test = np.concatenate((z, x_test), axis=1) #go mark
        else: 
            tuning = 1
            level = 1
            if auto_regress or decode_xother is False:
                z = np.full((x_test.shape[0], 1), bos, dtype = x_test.dtype)
                x_test = np.concatenate((z, x_test), axis=1) #go mark

        s = f"\nhanormal decord train:  EPOCH {epoch} RELAY {relay} DECODE {decord_mode} LEVEL {level} REPEAT {repeat} TUNING {tuning}  DECODE_LEV_LEARN {train_params.get('dec_lev_learn')}\n"
        print(s)
        mp.print_log(net, s)
        #print_array(x_train, "train")
        count = 0
        if auto_regress:
            n_input = int(x_test.shape[1] / 2)
        else: n_input = -1
        if repeat > 1: relay = 0
        while count < repeat:
            present = 10 if repeat == 1 or count + 1 == repeat else -1 #마지막 반복에서만 프린트
            reent = np.expand_dims(x_train, -1)
            for i in range(level):
                name = f'aaa_{i}'
                reent = mp.xtrain(net, reent, xtrain_other, tuning, epoch, relay, decord_mode)#, name, present)

            x_test1 = np.expand_dims(x_test, -1)
            predictions = mp.xpredict(net, x_test1, decord_mode, n_input)#, 'bbb', present)
            print(x_test.shape)
            print(predictions.shape)

            if False:#tuning and auto_regress:
                x_test2 = x_test[:,1:]
                predictions = predictions[:,:-1]
            elif decode_xother:
                x_test2 = xtest_other
            else:
                x_test2 = x_test

            y_pred, threshold, mse = discriminator(y_test, x_test2, predictions, inner_mul)
            s = f'{repeat} REPEAT: {count} THRESHOLD: {threshold} DECODE MODE: {decord_mode}'
            print(s)
            mp.print_log(net, s)
            threshold = predict_present(y_test, y_pred, net, present, mse, jit_count)

            if threshold > 0:#위 27개 이상의 정답을 맞추면 위 과정에서 획득된 최적 임계치로 다시 정답획득하여 오류개수를 줄인다.
                y_pred, _, _ = discriminator(y_test, x_test2, predictions, inner_mul, mse, threshold)
                s = f'optim THRESHOLD: {threshold} DECODE MODE: {decord_mode}'
                print(s)
                mp.print_log(net, s)
                predict_present(y_test, y_pred, net, present)
                #정답의 11배 아웃터로 임계치를 획득하고 최적임계치를 획득하여
                y_pred, threshold, _ = discriminator(y_test, x_test2, predictions, outer_mul, mse)
                s = f'outer THRESHOLD: {threshold} DECODE MODE: {decord_mode}'
                print(s)
                mp.print_log(net, s)
                threshold = predict_present(y_test, y_pred, net, present, mse)#최적임계치 획득
                #최적임계치로 오류를 줄인다.
                y_pred, _, _ = discriminator(y_test, x_test2, predictions, outer_mul, mse, threshold)
                s = f'outer optim THRESHOLD: {threshold} DECODE MODE: {decord_mode}'
                print(s)
                mp.print_log(net, s)
                predict_present(y_test, y_pred, net, present)

            count += 1
            """
            error_df = pd.DataFrame({'reconstruction_error': mse,
                            'true_class': Y_test})
            y_pred = [1 if e > threshold else 0 for e in error_df.reconstruction_error.values]
            conf_matrix = confusion_matrix(error_df.true_class, y_pred)
            plt.figure(figsize=(12, 12))
            sns.heatmap(conf_matrix, xticklabels=LABELS, yticklabels=LABELS, annot=True, fmt="d");
            plt.title("Confusion matrix")
            plt.ylabel('True class')
            plt.xlabel('Predicted class')
            plt.show()
            """
       

    elif sys.argv[1] == '6':
        if len(sys.argv) >= 6:
            decode = int(sys.argv[5])
        else:
            decode = 1
        if len(sys.argv) >= 7:
            inner_mul = int(sys.argv[6])

        if len(sys.argv) >= 8:
            jit_count = int(sys.argv[7])
    

        if auto_regress or decode_xother is False:
            z = np.full((x_test.shape[0], 1), bos, dtype = x_test.dtype)
            x_test = np.concatenate((z, x_test), axis=1) #go mark
            n_input = int(x_test.shape[1] / 2)
        else: n_input = -1
        x_test1 = np.expand_dims(x_test, -1)
        predictions = mp.xpredict(net, x_test1, decode, n_input)#, 'bbb', 100) #decord
        print(x_test.shape)
        print(predictions.shape)

        if False:
            x_test = x_test[:,1:]
            predictions = predictions[:,:-1]
        elif decode_xother:
            x_test = xtest_other

        y_pred, threshold, mse = discriminator(y_test, x_test, predictions, inner_mul)
        print('THRESHOLD: ', threshold, 'DEOCDE MODE: ', decode)
        #threshold = 0.03 #decord 2
        #threshold = 23.5 #decord 3
        threshold = predict_present(y_test, y_pred, net, 1, mse, jit_count)

        if threshold > 0:
            y_pred, _, _ = discriminator(y_test, x_test, predictions, inner_mul, mse, threshold)
            s = f'optim THRESHOLD: {threshold} DECODE MODE: {decode}'
            print(s)
            mp.print_log(net, s)
            predict_present(y_test, y_pred, net, 1)

            y_pred, threshold, _ = discriminator(y_test, x_test, predictions, outer_mul, mse)
            s = f'outer THRESHOLD: {threshold} DECODE MODE: {decode}'
            print(s)
            mp.print_log(net, s)
            threshold = predict_present(y_test, y_pred, net, 1, mse)

            y_pred, _, _ = discriminator(y_test, x_test, predictions, outer_mul, mse, threshold)
            s = f'outer optim THRESHOLD: {threshold} DECODE MODE: {decode}'
            print(s)
            mp.print_log(net, s)
            predict_present(y_test, y_pred, net, 1)

        """
        error_df = pd.DataFrame({'reconstruction_error': mse,
                        'true_class': Y_test})
        y_pred = [1 if e > threshold else 0 for e in error_df.reconstruction_error.values]
        conf_matrix = confusion_matrix(error_df.true_class, y_pred)
        plt.figure(figsize=(12, 12))
        sns.heatmap(conf_matrix, xticklabels=LABELS, yticklabels=LABELS, annot=True, fmt="d");
        plt.title("Confusion matrix")
        plt.ylabel('True class')
        plt.xlabel('Predicted class')
        plt.show()
        """
