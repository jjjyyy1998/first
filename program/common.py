'''
データ処理用の関数が入っているプログラム
'''

# coding:utf-8

### ライブラリのインポート ###
import numpy as np
from tqdm import tqdm
from scipy.io.wavfile import read
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve,roc_auc_score
import glob
import torch
import Smooth
import sys
class torch_dataset():
    def __init__(self,X,y):
        
        self.X = X
        self.y = y
        self.data_num = len(self.X)

    def __len__(self):
        return self.data_num
    
    def __getitem__(self, idx):
        out_data = self.X[idx]
        out_label = self.y[idx]

        return out_data,out_label
### フレーム化と窓掛けを行う関数 ###
def get_frames(filename, FRAME_LENGTH=128, FRAME_SHIFT_LENGTH=32):
    fs,data=read(filename)
    data = data[:,0]
    FRAME_LENGTH=FRAME_LENGTH*fs//1000
    FRAME_SHIFT_LENGTH=FRAME_SHIFT_LENGTH*fs//1000
    frame_num = 1+(len(data) - FRAME_LENGTH)//FRAME_SHIFT_LENGTH # フレーム数
    np_frames = np.zeros((frame_num, FRAME_LENGTH)) # 各フレームを入れるためのリスト(フレーム数×フレーム長)
    ## フレーム化処理を行う ##
    start_point = 0
    for i in range(frame_num):
        # np_frames[i] = data[start_point:start_point+FRAME_LENGTH,0]
        np_frames[i] = data[start_point:start_point+FRAME_LENGTH]
        start_point += FRAME_SHIFT_LENGTH

    ## 窓掛けを行う ##
    np_frames = np_frames * np.hamming(FRAME_LENGTH) # ハミング窓を用いる
   
    return np_frames



### 各フレームをFFTにより周波数領域に変換する関数 ###
def fft_data(filename,FRAME_LENGTH=128, FRAME_SHIFT_LENGTH=32,BINS=64):
    frames=get_frames(filename,FRAME_LENGTH, FRAME_SHIFT_LENGTH)
    spectrum = np.zeros([frames.shape[0],frames.shape[1]//2]) # spectrum:(対数パワー)スペクトルを格納する
    for i, frame in enumerate(frames):
        wave_fft = np.fft.fft(frame) # numpyのFFTライブラリを用いる
        
        X = abs(wave_fft) # 絶対値を取り
        X = np.log10(X+1e-6) # 常用対数を取り, -> 対数振幅スペクトル(真数条件より, 0.00001を追加)

        spectrum[i] = X[:len(X)//2] # ナイキスト周波数を考慮し, 半分までを代入
    size=len(spectrum[0])//BINS #64次元に変換するための幅情報を入手
    input_spectrum = np.zeros((len(spectrum),BINS))
    for i in range(BINS):
        for j in range(len(spectrum)):
            input_spectrum[j,i]=np.mean(spectrum[j,i*size:(i+1)*size])
            
    
    return input_spectrum


### 各フレームをFFTにより周波数領域に変換する関数 ###
def create_dataset_2d(spectrogram,BINS):
    size=int((len(spectrogram)-BINS)/4) # 周波数ビンを元に正方形にする
    data=np.zeros((size,BINS,len(spectrogram[0])))
    for i in range(size):
        data[i]=spectrogram[4*i:4*i+BINS]
    data=data.reshape((len(data),len(data[0]),len(data[0][0]),1))
    return data

### 学習データセットの作成 ###
def MyDataset(TRAIN_PATH,VALID_PATH,FRAME_LENGTH, FRAME_SHIFT_LENGTH,BINS,output_path,CHECK_POINT):
    print("train file num:{}".format(len(TRAIN_PATH)))
    for i,filename in enumerate(TRAIN_PATH):
        spectrogram = fft_data(filename, FRAME_LENGTH, FRAME_SHIFT_LENGTH,BINS) 
        data = create_dataset_2d(spectrogram,BINS) #正方形に変換
        if i == 0:
            data_list=np.zeros((len(data)*len(TRAIN_PATH),len(data[0]),len(data[0][0]),1))
            label = np.zeros((len(data)*len(TRAIN_PATH)))
        data_list[i*len(data):(i+1)*len(data)]=data #連結
        if 'id_00' in filename:
            label[i*len(data):(i+1)*len(data)]=0 
        elif 'id_02' in filename:
            label[i*len(data):(i+1)*len(data)]=1 
        elif 'id_04' in filename:
            label[i*len(data):(i+1)*len(data)]=2 
        elif 'id_06' in filename:
            label[i*len(data):(i+1)*len(data)]=3 
        # break
    data_list = np.squeeze(data_list)
    data_list = data_list.reshape((len(data_list),1,64,64))
    # np.random.seed(42)
    # np.random.shuffle(data_list) #データシャッフル
    data_list,mean_,std_ = train_Standard(data_list,output_path,CHECK_POINT,BINS) #標準化実行
    data_list = torch.from_numpy((data_list).astype(np.float32)).clone()
    label = torch.from_numpy((label)).clone()
    print('train datalist : {}'.format(data_list.shape))
    for i,filename in enumerate(VALID_PATH):
        spectrogram = fft_data(filename, FRAME_LENGTH, FRAME_SHIFT_LENGTH,BINS) 
        data = create_dataset_2d(spectrogram,BINS) #正方形に変換
        if i == 0:
            valid_list=np.zeros((len(data)*len(VALID_PATH),len(data[0]),len(data[0][0]),1))
            valid_label = np.zeros((len(data)*len(VALID_PATH)))
        valid_list[i*len(data):(i+1)*len(data)]=data #連結
        if 'id_00' in filename:
            valid_label[i*len(data):(i+1)*len(data)]=0 
        elif 'id_02' in filename:
            valid_label[i*len(data):(i+1)*len(data)]=1 
        elif 'id_04' in filename:
            valid_label[i*len(data):(i+1)*len(data)]=2 
        elif 'id_06' in filename:
            valid_label[i*len(data):(i+1)*len(data)]=3 
        # break
    valid_list = np.squeeze(valid_list)
    valid_list = valid_list.reshape((len(valid_list),1,64,64))
    np.random.seed(42)
    np.random.shuffle(valid_list) #データシャッフル
    np.random.seed(42)
    np.random.shuffle(valid_label)
    # data_list = train_Standard(data_list,output_path,CHECK_POINT,BINS) #標準化実行
    valid_list = (valid_list-mean_)/std_
    valid_list = torch.from_numpy((valid_list).astype(np.float32)).clone()
    valid_label = torch.from_numpy((valid_label)).clone()

    print('train datalist : {}'.format(valid_list.shape))

    return data_list,label,valid_list,valid_label

def testset(TRAIN_PATH,FRAME_LENGTH, FRAME_SHIFT_LENGTH,BINS,output_path,CHECK_POINT):
    print("train file num:{}".format(len(TRAIN_PATH)))
    for i,filename in enumerate(tqdm(TRAIN_PATH)):
        spectrogram = fft_data(filename, FRAME_LENGTH, FRAME_SHIFT_LENGTH,BINS) 
        data = create_dataset_2d(spectrogram,BINS) #正方形に変換
        if i == 0:
            data_list=np.zeros((len(data)*len(TRAIN_PATH),len(data[0]),len(data[0][0]),1))
        data_list[i*len(data):(i+1)*len(data)]=data #連結
        # break
    # np.random.seed(42)
    # np.random.shuffle(data_list) #データシャッフル
    data_list = np.squeeze(data_list)
    data_list = data_list.reshape((len(data_list),1,64,64))
    data_list = torch.from_numpy((data_list).astype(np.float32)).clone()
    # data_list = train_Standard(data_list,output_path,CHECK_POINT,BINS) #標準化実行
    print('train datalist : {}'.format(data_list.shape))

    return data_list

### 学習データで標準化 ###
def train_Standard(train,path,CHECK_POINT,BINS):
    # mean_=[]
    # std_=[]
    # for i in range(BINS):
    #     mean_.append(np.mean(train[:,:,i]))
    #     std_.append(np.std(train[:,:,i]))
    #     train[:,:,i]=(train[:,:,i]-mean_[i])/std_[i]
    mean_ = np.mean(train)
    std_  = np.std(train)
    train = (train - mean_)/std_
    np.save(path+'/{}/std.npy'.format(CHECK_POINT),std_)
    np.save(path+'/{}/mean.npy'.format(CHECK_POINT),mean_)
    return train,mean_,std_

### マスク位置を決めるためのコード ###
def mask_generator(BINS,PATCH_NUM):
    shift = int(BINS // (np.sqrt(PATCH_NUM))) #パッチ化に対応するピクセル数を決める
    patch1 = np.zeros((PATCH_NUM,shift**2), dtype=np.int64) #フレーム方向の配列
    patch2 = np.zeros((PATCH_NUM,shift**2), dtype=np.int64) #周波数方向の配列
    for i in range(PATCH_NUM):
        for j in range(shift**2):
            patch1[i,j] = int(shift * (i % (BINS//shift)) + j % shift)
            patch2[i,j] = int(shift * int(i//(BINS//shift)) + int(j//shift))
    index_ = np.arange((len(patch1))) #パッチ番号

    return patch1,patch2,index_

### 異常度算出 ###
def anomaly_score(mean_,std_,data_path,output_path,CHECK_POINT,result,model,FRAME_SHIFT_LENGTH,FRAME_LENGTH,BINS,PATCH_NUM,MASK_NUM,GIF,device):
    patch_1,patch_2,index_=mask_generator(BINS,PATCH_NUM) #マスク生成
    scam = Smooth.SmoothGrad(model,device)
    for i,filename in enumerate(tqdm(data_path)):
        spectrogram = fft_data(filename, FRAME_LENGTH, FRAME_SHIFT_LENGTH,BINS)
        data = create_dataset_2d(spectrogram,BINS)
        data = data.reshape((len(data),1,64,64))
        data = torch.from_numpy((data).astype(np.float32)).clone()
        data=(data-mean_)/std_
        if 'id_00' in filename:
            label = np.zeros((len(data))) 
        elif 'id_02' in filename:
            label = np.zeros((len(data))) + 1 
        elif 'id_04' in filename:
            label = np.zeros((len(data))) + 2 
        elif 'id_06' in filename:
            label = np.zeros((len(data))) + 3
        if i ==0:
            mid = np.zeros((data.shape[0]*len(data_path),496))
            # mid = np.zeros((data.shape[0]*len(data_path),992))
            
        data = data.to(device)
        # copy_ = data.clone()
        # out,hidden_half,_ = model([data,copy_])
        out,hidden_half,clf,_,f= model(data)
        # a,b = scam(data,label)
        # sys.exit()
        
        h = _.to('cpu').detach().numpy().copy()
        # h = f.to('cpu').detach().numpy().copy()
        
        out = clf[:,1].to('cpu').detach().numpy().copy()
        mse = np.mean(out)
        # mse = np.mean((out-h)**2)
        result.append(mse) #AUC算出用のMSE格納
        
        mid[i*len(data):(i+1)*len(data)] = h
        # if i ==1:
        #     break
    return mid

def anomaly_score2(mean_,std_,data_path,output_path,CHECK_POINT,result,model,FRAME_SHIFT_LENGTH,FRAME_LENGTH,BINS,PATCH_NUM,MASK_NUM,GIF,device):
    patch_1,patch_2,index_=mask_generator(BINS,PATCH_NUM) #マスク生成
    for i,filename in enumerate(tqdm(data_path)):
        spectrogram = fft_data(filename, FRAME_LENGTH, FRAME_SHIFT_LENGTH,BINS)
        data = create_dataset_2d(spectrogram,BINS)
        data = data.reshape((len(data),1,64,64))
        data = torch.from_numpy((data).astype(np.float32)).clone()
        data=(data-mean_)/std_
        if i ==0:
            mid = np.zeros((data.shape[0]*len(data_path),248))
        data = data.to(device)
        copy_ = data.clone()
        out,hidden_half,_,x1,x2 = model([data,copy_])
        h = _[0].to('cpu').detach().numpy().copy()
        out = out.to('cpu').detach().numpy().copy()
        mse = np.mean(out)
        result.append(mse) #AUC算出用のMSE格納
        
        mid[i*len(data):(i+1)*len(data)] = h
        # if i ==1:
        #     break
    return mid


### 学習 ###
def train(data_list,BATCH_SIZE,model,output_path,CHECK_POINT,EPOCHS,BINS,PATCH_NUM,MASK_NUM):
    flag = 10000000000
    patch_1,patch_2,index_=mask_generator(BINS,PATCH_NUM) #マスク生成
    data_list,valid_data_list=train_test_split(data_list, test_size=0.1,random_state=42) #データ分割
    th = int(len(index_)*0.5)
    train_loss=[]
    valid_loss=[]
    patch_size = int(0.25 * len(patch_1))
    cnt = 0
    for epoch in range(EPOCHS):
        np.random.seed(epoch)
        np.random.shuffle(data_list) #エポックごとにデータシャッフル
        train_loss_list=[]
        valid_loss_list=[]
        print("Epoch is", epoch)
        print("Number of batches", int(data_list.shape[0]/BATCH_SIZE))
        
        for index in tqdm(range(int(data_list.shape[0]/BATCH_SIZE))): # training
            input_batch = data_list[index*BATCH_SIZE:(index+1)*BATCH_SIZE].copy() #入力
            target_batch = data_list[index*BATCH_SIZE:(index+1)*BATCH_SIZE].copy() #目標出力
            np.random.shuffle(index_)
            for i,batch in enumerate(input_batch):
                if index_[i] < th:
                    # noise = 0.1 * np.random.random_sample(batch.shape) - 0.05
                    noise = np.random.normal(0,0.3,batch.shape)
                    batch = batch + noise
                    target_batch[i] = target_batch[i] + noise
                    cnt = cnt+1
            # train_loss_list.append(model.train_on_batch(input_batch,input_batch))
            for i in input_batch:
                np.random.shuffle(index_)
                i[patch_1[index_[:MASK_NUM]],patch_2[index_[:MASK_NUM]]] = 0 #マスク位置を決めています


               
            train_loss_list.append(model.train_on_batch(input_batch,target_batch)) # train_on_batchはfit関数の代わり
            # train_loss_list.append(model.train_on_batch(input_batch,input_batch))
        train_loss.append(np.mean(train_loss_list))
        for index in tqdm(range(int(valid_data_list.shape[0]/BATCH_SIZE))): # validation
            input_batch = valid_data_list[index*BATCH_SIZE:(index+1)*BATCH_SIZE].copy()
            target_batch = valid_data_list[index*BATCH_SIZE:(index+1)*BATCH_SIZE].copy()
            for i in input_batch:
                np.random.shuffle(index_)
                # i[patch_1[index_[:MASK_NUM]],patch_2[index_[:MASK_NUM]]] = 0
            valid_loss_list.append(model.test_on_batch(input_batch,target_batch))
        valid = np.mean(valid_loss_list)
        valid_loss.append(valid)
        if valid < flag:
            model.save_weights((output_path+'/{}/UNet.h5'.format(CHECK_POINT))) # 検証データの誤差が最小になったときに重みを保存
            flag = valid
            point_ = epoch
            print('\n\nmodel_save\n{}{}_UNet.h5\n\n'.format(output_path,CHECK_POINT))
            print(cnt)
            cnt = 0
        print("train_loss={}, valid_loss={}".format(np.mean(train_loss_list),valid))
        plt.plot(train_loss, label="loss") 
        plt.plot(valid_loss, label="val_loss")
        plt.scatter(point_,flag,label='CHECK_POINT',color='r')
        plt.title('learning_curve')
        plt.ylabel('MSE')
        plt.xlabel('EPOCHS')
        plt.legend()
        plt.rcParams["font.size"] = 15
        plt.tight_layout()
        plt.savefig(output_path+'/{}/learning_curve_UNet'.format(CHECK_POINT))
        plt.close()

### 可視化 ###
def visualize(normal,anomaly,output_path,CHECK_POINT,section,domain,id_):
    if section is None:
        label=np.concatenate((np.zeros((len(normal))),np.ones((len(anomaly)))))
        anomaly_score=np.concatenate((normal,anomaly))
        fpr, tpr, thresholds = roc_curve(label, anomaly_score) # ROCの計算
        score=roc_auc_score(label, anomaly_score) #AUC
        pscore=roc_auc_score(label, anomaly_score,max_fpr=0.1) #pAUC(p=0.1)

        ### ROC ###
        plt.plot(fpr, tpr, marker='o')
        plt.xlabel('FPR: False positive rate')
        plt.ylabel('TPR: True positive rate')
        plt.title('{}_AUC={:.4f}_pAUC={:.4f}'.format(id_,score,pscore))
        plt.grid()
        plt.savefig(output_path+"/{}/{}_roc".format(CHECK_POINT,id_))
        plt.close()

        ### 異常度の可視化 ###
        index=np.arange(len(anomaly_score))
        plt.scatter(index[:len(normal)],anomaly_score[:len(normal)],s=5,label='normal')
        plt.scatter(index[len(normal):],anomaly_score[len(normal):],s=5,label='anomaly')
        plt.xlabel('anomaly_score')
        plt.ylabel('MSE')
        plt.title('{}_AUC={:.4f}_pAUC={:.4f}'.format(id_,score,pscore))
        plt.legend()
        plt.savefig(output_path+"/{}/{}_anomaly_score".format(CHECK_POINT,id_))
        plt.close()

        ### AUC保存 ###
        with open(output_path+'/{}/result.txt'.format(CHECK_POINT), 'a', newline="\n") as f: # txtに書き込み
            f.write('\n{}_AUC={:.4f},'.format(id_,score))  
            # f.write('\n{}_pAUC={:.4f},'.format(id_,pscore)) 
    
    else:
        label=np.concatenate((np.zeros((len(normal))),np.ones((len(anomaly)))))
        anomaly_score=np.concatenate((normal,anomaly))
        fpr, tpr, thresholds = roc_curve(label, anomaly_score) # ROCの計算
        score=roc_auc_score(label, anomaly_score) #AUC
        pscore=roc_auc_score(label, anomaly_score,max_fpr=0.1) #pAUC(p=0.1)

        ### ROC ###
        plt.plot(fpr, tpr, marker='o')
        plt.xlabel('FPR: False positive rate')
        plt.ylabel('TPR: True positive rate')
        plt.title('AUC={:.4f}_pAUC={:.4f}'.format(score,pscore))
        plt.grid()
        plt.savefig(output_path+"/{}/roc_{}_domain={}".format(CHECK_POINT,section,domain))
        plt.close()

        ### 異常度の可視化 ###
        index=np.arange(len(anomaly_score))
        plt.scatter(index[:len(normal)],anomaly_score[:len(normal)],s=5,label='normal')
        plt.scatter(index[len(normal):],anomaly_score[len(normal):],s=5,label='anomaly')
        plt.xlabel('anomaly_score')
        plt.ylabel('Abnormality')
        plt.title('AUC={:.4f}_pAUC={:.4f}'.format(score,pscore))
        plt.legend()
        plt.savefig(output_path+"/{}/anomaly_score_{}_domain={}".format(CHECK_POINT,section,domain))
        plt.close()

        ### AUC保存 ###
        with open(output_path+'/{}/result.txt'.format(CHECK_POINT), 'a', newline="\n") as f: # txtに書き込み
            f.write('\n{}_{}_AUC={:.4f},'.format(section,domain,score))  
            # f.write('\n{}_{}_pAUC={:.4f},'.format(section,domain,pscore))  

        
def make_path_list(root_,ID_,SEED):
    train_list=[]
    valid_list=[]
    normal_list=[]
    anomaly_list=[]
    if type(ID_) != list:
        root = root_+'/'+ID_+'/'
        print(root)
        normal=glob.glob(root+'normal/*wav')
        anomaly=glob.glob(root+'abnormal/*wav')
        np.random.seed(SEED)
        np.random.shuffle(normal)
        print(normal[:10])
        for i in range(len(normal[len(anomaly):])):
            train_list.append(normal[len(anomaly)+i])
        for i in range(len(anomaly)):
            normal_list.append(normal[i])
            anomaly_list.append(anomaly[i])
            
        print(len(train_list),len(normal_list),len(anomaly_list))
    else:
        for id in ID_:
            root = root_+'/'+id+'/'
            print(root)
            normal=glob.glob(root+'normal/*wav')
            anomaly=glob.glob(root+'abnormal/*wav')
            np.random.seed(SEED)
            np.random.shuffle(normal)
            print(normal[:10])
            for i in range(len(normal[len(anomaly):])):
                if i % 10 == 0:
                    valid_list.append(normal[len(anomaly)+i])
                else:
                    train_list.append(normal[len(anomaly)+i])
            for i in range(len(anomaly)):
                normal_list.append(normal[i])
                anomaly_list.append(anomaly[i])
            
        print(len(train_list),len(normal_list),len(anomaly_list))

    return train_list,valid_list,normal_list,anomaly_list


        

