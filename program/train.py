import glob
import sys
import os
import torch_model as models
import common as com
import torch
import torch.nn.functional as F
from optim import *
from sklearn.model_selection import train_test_split
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from pytorch_metric_learning import losses, miners, distances, reducers, samplers

# def pair_cos_loss(arr):
#     output=torch.zeros(int(len(arr)/2))
#     for i in range(len(output)):
#         idx = torch.randperm(arr.shape[0])
#         arr = arr[idx].view(arr.size())
#         output[i] = - F.cosine_similarity(arr[2*i], arr[2*1+1],dim=0) + 1
#     return torch.mean(output**2) 
def pair_cos_loss(arr1,arr2):
    output=torch.zeros(int(len(arr1)))
    for i in range(len(output)):
        # idx = torch.randperm(arr.shape[0])
        # arr = arr[idx].view(arr.size())
        output[i] = - F.cosine_similarity(arr1[i].view(-1), arr2[i].view(-1),dim=0) + 1
    return torch.mean(output**2) 

def cent(half_hidden,mi):
    c = torch.zeros(len(half_hidden))
    for i in range(len(half_hidden)):
        c[i] = torch.mean((half_hidden[i]-mi)**2)

    return torch.mean(c)

def pair_cos_loss(arr):
    output=torch.zeros(int(len(arr[0])))
    for i in range(len(output)):
        output[i] = - F.cosine_similarity(arr[0][i], arr[1][i],dim=0) + 1
    return torch.mean(output) 

def pair_cos_far_loss(arr):
    size = int(len(arr)/2)
    output=torch.zeros(size)
    for i in range(len(output)):
        output[i] = F.cosine_similarity(arr[i], arr[size+i],dim=0) + 1
    return torch.mean(output**2)

# def pair_cos_far_loss(arr):
#     output=torch.zeros(int(len(arr[0])))
#     for i in range(len(output)):
#         output[i] = F.cosine_similarity(arr[0][i], arr[1][i],dim=0) + 1
#     return torch.mean(output**2) 

def pair_cos_far_loss_MN(arr1,arr2):
    output=torch.zeros(int(len(arr1)))
    for i in range(len(output)):
        idx = torch.randperm(arr1.shape[0])
        arr1 = arr1[idx].view(arr1.size())
        output[i] = F.cosine_similarity(arr1[i], arr2[i],dim=0) + 1
    return torch.mean(output) 


def main():
    # sys.exit()
    print('\n\n\n\nparameter\n\nBINS : {}\nFIL : {}\nBATCH_SIZE : {}\nFRAME_LENGTH : {}[ms]'.format(args.BINS,args.FIL,args.BATCH_SIZE,args.FRAME_LENGTH))
    print('FRAME_SHIFT_LENGTH : {}[ms]\nPERCENTAGE : {}\nPATCH_NUM : {}\nMASK_NUM : {}\nPATCH_TYPE : {}\n'.format(args.FRAME_SHIFT_LENGTH,args.PERCENTAGE,args.PATCH_NUM,MASK_NUM,args.MASK_TYPE))
    if args.MACHINE is None:
        train_root = args.ROOT+'/train/' #trainデータのrootを取得
        output_path = args.OUTPUT_PATH +'/' #出力パス
    else:
        # train_root = args.ROOT+args.MACHINE+'/train/' #trainデータのrootを取得
        file_root = args.ROOT+args.SNR+args.MACHINE #trainデータのrootを取得
        output_path = args.OUTPUT_PATH+args.MACHINE+'/' #出力パス

    print(file_root)
    
    if not os.path.isdir(output_path): os.makedirs(output_path)
    if not os.path.isdir(output_path+'/'+CHECK_POINT): os.makedirs(output_path+'/'+CHECK_POINT)
    train_path,valid_path,normal_file,anomaly_file = com.make_path_list(file_root,ID_,args.SEED)
    # train_path = glob.glob(train_root+'*') #ファイル取得
    train_path=sorted(train_path)

    train_data,train_label,valid_data,valid_label = com.MyDataset(train_path,valid_path,args.FRAME_LENGTH,args.FRAME_SHIFT_LENGTH,args.BINS,output_path,CHECK_POINT) #データセット作成
    # train_data,valid_data,train_label,valid_label = train_test_split(data_list,label,test_size=0.1, shuffle=True)
    # del data_list
    train_list = com.torch_dataset(train_data,train_label)
    valid_list = com.torch_dataset(valid_data,valid_label)
    train_dataloader = torch.utils.data.DataLoader(train_list, 
                                                   batch_size=64, 
                                                   shuffle=True,
                                                   drop_last=True)#データローダーへ
    valid_dataloader = torch.utils.data.DataLoader(valid_list, 
                                                   batch_size=64, 
                                                   shuffle=True,
                                                   drop_last=True)
    dataloaders_dict = {"train": train_dataloader, "validation": valid_dataloader}#辞書にまとめる
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("使用デバイス：", device)
    model = models.UNet() #モデル作成
    # model = models.AE() #モデル作成
    # model = models.Dual_model()
    model.to(device)
    criterion = torch.nn.MSELoss()
    ce = torch.nn.CrossEntropyLoss()
    cs = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    epochs = args.EPOCHS
    history = {"train_loss":[], "val_loss":[]} # 学習曲線用
    torch.backends.cudnn.benchmark = True
    min_ = 10100001000
    distance = distances.CosineSimilarity()
    reducer = reducers.ThresholdReducer(low=0)
    loss_func = losses.TripletMarginLoss(margin=0.2, distance=distance, reducer=reducer)
    miner = miners.TripletMarginMiner(margin=0.2, distance=distance, type_of_triplets="hard")
    alpha = 1.0
    patch_1,patch_2,index_=com.mask_generator(args.BINS,args.PATCH_NUM)
    random_index = np.arange(64)
    random_1 = np.arange(56)
    random_2 = np.arange(56)
    for epoch in range(args.EPOCHS):
        # if epoch == 600:
        #     optimizer = torch.optim.Adam(model.parameters(), lr=0.00001)
        # elif epoch == 800:
        #     optimizer = torch.optim.Adam(model.parameters(), lr=0.000001)
        # if epoch ==20:
        #     for name,param in model.named_parameters():
        #         if 'UNet_1' in name :
        #             param.requires_grad=False
                    
        print('Epoch：{}/{}'.format(epoch+1, epochs))
        
        for phase in ['train', 'validation']:
            # cnt = 0
            if phase == 'train':
                model.train()
            else:
                model.eval()
                
            epoch_loss = 0.0
            discriminator_loss = 0.0
            
            for cnt,(inputs,labels) in enumerate(dataloaders_dict[phase]):
                # print(1)
                # cnt +=1
                inp_ = inputs.clone()
                inp    = inputs.clone()
                labels_ = labels.clone()
                for i in inp:
                    a = np.random.rand()
                    

                    if a < 0.2:
                        np.random.shuffle(random_1)
                        np.random.shuffle(random_2)
                        i[:,random_1[0]:random_1[0]+8,random_2[0]:random_2[0]+8] += alpha 
                    elif a < 0.4:
                        np.random.shuffle(random_index)
                        i[:,random_index[0]] += alpha 
                    elif a < 0.6:
                        np.random.shuffle(random_index)
                        i[:,:,random_index[0]] += alpha 
                    elif a < 0.8:
                        np.random.shuffle(index_)
                        masking = int(MASK_NUM*alpha)
                        i[:,patch_1[index_[:masking]],patch_2[index_[:masking]]] += 1 
                    else:
                        np.random.shuffle(random_index)
                        rand = random_index[:int(64*alpha)+2]
                        sort_ = sorted(rand)
                        i[:,sort_] = i[:,rand]

                    
                    # if a < 0.2:
                    #     np.random.shuffle(random_index)
                    #     rand = random_index[:int(64*alpha)+2]
                    #     sort_ = sorted(rand)
                    #     rand = sorted(sort_,reverse=True)
                    #     i[:,sort_] = i[:,rand]
                    # a = np.random.rand()
                    # if a < 0.2:
                    #     np.random.shuffle(random_index)
                    #     rand = random_index[:int(64*alpha)+2]
                    #     sort_ = sorted(rand)
                    #     rand = sorted(sort_,reverse=True)
                    #     i[:,:,sort_] = i[:,rand]












                    # if a < 0.25:
                    #     np.random.shuffle(random_1)
                    #     np.random.shuffle(random_2)
                    #     i[:,random_1[0]:random_1[0]+8,random_2[0]:random_2[0]+8] += alpha 
                    # elif a < 0.5:
                    #     np.random.shuffle(random_index)
                    #     i[:,random_index[0]] += alpha 
                    # elif a < 0.75:
                    #     np.random.shuffle(random_index)
                    #     i[:,:,random_index[0]] += alpha 
                    # else:
                    #     np.random.shuffle(index_)
                    #     masking = int(MASK_NUM*alpha)
                    #     i[:,patch_1[index_[:masking]],patch_2[index_[:masking]]] += 1 
                    
                    # if a < 0.1:
                    #     np.random.shuffle(random_1)
                    #     np.random.shuffle(random_2)
                    #     i[:,random_1[0]:random_1[0]+8,random_2[0]:random_2[0]+8] += alpha 
                    # elif a < 0.2:
                    #     np.random.shuffle(random_1)
                    #     np.random.shuffle(random_2)
                    #     i[:,random_1[0]:random_1[0]+8,random_2[0]:random_2[0]+8] -= alpha 
                    # elif a < 0.3:
                    #     np.random.shuffle(random_index)
                    #     i[:,random_index[0]] += alpha 
                    # elif a < 0.4:
                    #     np.random.shuffle(random_index)
                    #     i[:,random_index[0]] -= alpha 
                    # elif a < 0.5:
                    #     np.random.shuffle(random_index)
                    #     i[:,:,random_index[0]] += alpha 
                    # elif a < 0.6:
                    #     np.random.shuffle(random_index)
                    #     i[:,:,random_index[0]] -= alpha 
                    # elif a < 0.8:
                    #     np.random.shuffle(index_)
                    #     masking = int(MASK_NUM*alpha)
                    #     i[:,patch_1[index_[:masking]],patch_2[index_[:masking]]] += 1 
                    # else:
                    #     np.random.shuffle(random_index)
                    #     rand = random_index[:int(64*alpha)+2]
                    #     sort_ = sorted(rand)
                    #     i[:,sort_] = i[:,rand]
                    
                # GPUが使えるならGPUにデータを送る
                
                
                inp = torch.cat((inp_,inp),0)
                lab = labels.clone() + 4
                # lab = labels.clone() 
                labels = torch.cat((labels,lab),0)
                # w = int(len(labels)//2)
                # g = torch.cat((torch.zeros(w),torch.ones(w)),0)
                inp = inp.float().to(device)
                targets = inp.clone()
                # if phase == 'train':
                #     for i in inp_:
                #         np.random.shuffle(index_)
                #         i[:,patch_1[index_[:MASK_NUM]],patch_2[index_[:MASK_NUM]]] = 0 #マスク位置を決めています
                #     inp = inp_.float().to(device)
                # if phase == 'train':
                #     for i in inp:
                #         np.random.shuffle(index_)
                #         i[:,patch_1[index_[:MASK_NUM]],patch_2[index_[:MASK_NUM]]] = 0 #マスク位置を決めています
                    
                
                # inp = torch.reshape(inp,(2,-1,1,64,64))
                # inp = inp.float().to(device)
                
                labels = labels.long().to(device)
                # g = g.long().to(device)
                
                # targets = torch.reshape(targets,(2,-1,1,64,64))
                targets = targets.float().to(device)
                # print(1)
                # print(1)
                # 順伝播
                with torch.set_grad_enabled(phase == 'train'):
                
                    optimizer.zero_grad()
                    # outputs,half_hidden,far_hidden,x1,x2 = model(inputs)
                    # outputs,half_hidden,out,_ = model(inp)
                    # outputs,embbed,clf,eg,_ = model(inp)
                    outputs,clf,eg,_,f = model(inp)
                    # triplets = miner(embbed, labels)
                    triplets = miner(_, labels)
                    # if epoch == 10 and cnt == 0 and phase == 'train':
                    #     mi = np.zeros((len(dataloaders_dict[phase].dataset),half_hidden.shape[1]))
                    # # outputs_1 = model(inputs[:,0],inputs[:,1],inputs[:,2],inputs[:,3])
                    # print(len(triplets))
                    # loss = criterion(outputs, targets) + ce(clf,labels)  + loss_func(embbed,labels,triplets)
                    # d_loss = pair_cos_far_loss(eg)

                    # loss = criterion(outputs, targets)
                    loss = criterion(outputs, targets) + loss_func(_,labels,triplets) + ce(clf,labels) #+ loss_func(_,labels,triplets) #+ 0.1 * cs(eg,g) + loss_func(_,label,triplets)
                    
                    
                    
                    # print(loss)
                    # if epoch == 10 and phase == 'train':
                    #     mi[(cnt)*inputs.shape[0]:(cnt+1)*inputs.shape[0]] = half_hidden.to('cpu').detach().numpy().copy()
                    # loss = criterion(outputs, targets) #+ pair_cos_loss(x1,x2)#+ criterion(x1,x2) * 100#+ pair_cos_loss(half_hidden) + pair_cos_far_loss(far_hidden) # + pair_cos_far_loss_MN(half_hidden,far_hidden)
                    # if epoch > 10:
                    #     loss = cent(half_hidden,mi) #criterion(outputs, targets) + cent(half_hidden,mi) 
                    # else:
                    #     loss = criterion(outputs, targets)
                    # mse += criterion(outputs, targets) * inputs.size(0)
                    # train時は学習
                    # print(cnt)
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                    # sys.exit()
                        

                epoch_loss += loss.item() * inputs.size(0)        # lossの合計を更新
                # discriminator_loss += d_loss.item() * inputs.size(0) 
                    
            epoch_loss = epoch_loss / len(dataloaders_dict[phase].dataset)
            # discriminator_loss = discriminator_loss / len(dataloaders_dict[phase].dataset)
            # print(discriminator_loss)

            # if epoch ==10 and phase == 'train':
            #     print(mi.shape)
            #     mi = np.mean(mi,axis=0)
            #     mi = torch.from_numpy(mi.astype(np.float32)).clone()
            #     mi = mi.float().to(device)



            # historyにlossとaccuracyを保存
            if phase == "train":
                history["train_loss"].append(epoch_loss)
                # print(torch.mean(mse))
            else:
                history["val_loss"].append(epoch_loss)
                if epoch_loss < min_: #and epoch > 10:
                    print("\n\n##################         model save         ##################\n\n")
                    torch.save(model.state_dict(),'./{}/{}/model.pth'.format(output_path,CHECK_POINT)) #検証データが最もも小さくなったところでモデル保存
                    min_ = epoch_loss
                    point_ = epoch
                    # if epoch_loss < alpha:
                    #     alpha = epoch_loss

            print('{} loss: {:.4f}'.format(phase, epoch_loss))
    
        plt.plot(history["train_loss"],label='train_loss')
        plt.plot(history["val_loss"],label='validation_loss')
        plt.scatter(point_,min_,label='check_point',color='r')
        plt.xlabel('Epoch')
        plt.ylabel('MSE')    
        plt.title('Learning Curve')
        plt.legend()
        plt.savefig('./{}/{}/Leaning_Curve.png'.format(output_path,CHECK_POINT))
        plt.close()
    # for param in model.parameters():
    #     param.requires_grad = False 
    # model = models.DeepSVDD(model,half_hidden.shape[1])
    # model.to(device)
    # min_ = 10100001000
    
    # patch_1,patch_2,index_=com.mask_generator(args.BINS,args.PATCH_NUM)
    # for epoch in range(epochs):
    #     print('Epoch：{}/{}'.format(epoch+1, epochs))
        
    #     for phase in ['train', 'validation']:
    #         # cnt = 0
    #         if phase == 'train':
    #             model.train()
    #         else:
    #             model.eval()
                
    #         epoch_loss = 0.0
            
    #         for cnt,inputs in enumerate(tqdm(dataloaders_dict[phase])):
    #             # cnt +=1
    #             if phase == 'train' and np.random.rand > 0.4:
    #                 noise = torch.normal(mean=torch.ones(inputs.shape), std=torch.ones(inputs.shape)*0.1)
    #                 inputs = inputs + noise
    #             # GPUが使えるならGPUにデータを送る
    #             inputs = inputs.float().to(device)
    #             # inputs = torch.reshape(inputs,(2,-1,1,64,64))
    #             targets = inputs.clone()
    #             for i in inputs:
    #                 np.random.shuffle(index_)
    #                 i[:,patch_1[index_[:MASK_NUM]],patch_2[index_[:MASK_NUM]]] = 0 #マスク位置を決めています
    #             # 順伝播
    #             with torch.set_grad_enabled(phase == 'train'):
                
    #                 optimizer.zero_grad()
    #                 # outputs,half_hidden,far_hidden,x1,x2 = model(inputs)
    #                 outputs,half_hidden = model(inputs)
    #                 # if epoch == 10 and cnt == 0 and phase == 'train':
    #                 #     mi = np.zeros((len(dataloaders_dict[phase].dataset),half_hidden.shape[1]))
    #                 # # outputs_1 = model(inputs[:,0],inputs[:,1],inputs[:,2],inputs[:,3])
    #                 loss = criterion(outputs, half_hidden)
    #                 # if epoch == 10 and phase == 'train':
    #                 #     mi[(cnt)*inputs.shape[0]:(cnt+1)*inputs.shape[0]] = half_hidden.to('cpu').detach().numpy().copy()
    #                 # loss = criterion(outputs, targets) #+ pair_cos_loss(x1,x2)#+ criterion(x1,x2) * 100#+ pair_cos_loss(half_hidden) + pair_cos_far_loss(far_hidden) # + pair_cos_far_loss_MN(half_hidden,far_hidden)
    #                 # if epoch > 10:
    #                 #     loss = cent(half_hidden,mi) #criterion(outputs, targets) + cent(half_hidden,mi) 
    #                 # else:
    #                 #     loss = criterion(outputs, targets)
    #                 # mse += criterion(outputs, targets) * inputs.size(0)
    #                 # train時は学習
    #                 # print(cnt)
    #                 if phase == 'train':
    #                     loss.backward()
    #                     optimizer.step()
                        

    #             epoch_loss += loss.item() * inputs.size(0)        # lossの合計を更新
                    
    #         epoch_loss = epoch_loss / len(dataloaders_dict[phase].dataset)
            

    #         # if epoch ==10 and phase == 'train':
    #         #     print(mi.shape)
    #         #     mi = np.mean(mi,axis=0)
    #         #     mi = torch.from_numpy(mi.astype(np.float32)).clone()
    #         #     mi = mi.float().to(device)



    #         # historyにlossとaccuracyを保存
    #         if phase == "train":
    #             history["train_loss"].append(epoch_loss)
    #             # print(torch.mean(mse))
    #         else:
    #             history["val_loss"].append(epoch_loss)
    #             if epoch_loss < min_: #and epoch > 10:
    #                 print("\n\n##################         model save         ##################\n\n")
    #                 torch.save(model.state_dict(),'./{}/{}/model.pth'.format(output_path,CHECK_POINT)) #検証データが最もも小さくなったところでモデル保存
    #                 min_ = epoch_loss
    #                 point_ = epoch

    #         print('{} loss: {:.4f}'.format(phase, epoch_loss))

    
  
   
    
    # com.train(data_list,args.BATCH_SIZE,model,output_path,CHECK_POINT,args.EPOCHS,args.BINS,args.PATCH_NUM,MASK_NUM) #学習

if __name__ == "__main__":
    main()




