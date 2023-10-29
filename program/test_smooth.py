import numpy as np
import glob
import sys
import torch_model as models
import common as com
import torch
import torch.nn.functional as F
from optim import * #optim.pyにある変数をすべて読み込む
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.covariance import LedoitWolf
from sklearn.covariance import MinCovDet
import Smooth


def main(args):
    
    if args.MACHINE is None:
        output_path = args.OUTPUT_PATH+'/' 
        test_root = args.ROOT+'/test/' #trainデータのrootを取得
        # model_weights=output_path+'/{}/UNet.h5'.format(CHECK_POINT)
        model = models.UNet() #モデル作成
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model.to(device)
        model_weight = './{}/{}/model.pth'.format(output_path,CHECK_POINT)
        model.load_state_dict(torch.load(model_weight))
        normal_path = glob.glob(test_root + '*{}*'.format(args.NORMAL)) #正常データ
        anomaly_path = glob.glob(test_root + '*{}*'.format(args.ANOMALY)) #異常データ
        normal_testfiles=sorted(normal_path)
        anomaly_testfiles=sorted(anomaly_path)
        mean_=np.load(output_path+'/{}/mean.npy'.format(CHECK_POINT))
        std_=np.load(output_path+'/{}/std.npy'.format(CHECK_POINT))
        section = None
        domain = None
        normal=[]
        anomaly=[]
        com.anomaly_score(mean_,std_,normal_testfiles,output_path,CHECK_POINT,normal,model,args.FRAME_SHIFT_LENGTH,args.FRAME_LENGTH,args.BINS,args.PATCH_NUM,MASK_NUM,args.GIF) #正常データの異常度算出
        com.anomaly_score(mean_,std_,anomaly_testfiles,output_path,CHECK_POINT,anomaly,model,args.FRAME_SHIFT_LENGTH,args.FRAME_LENGTH,args.BINS,args.PATCH_NUM,MASK_NUM,args.GIF) #異常データの異常度算出
        com.visualize(normal,anomaly,output_path,CHECK_POINT,section,domain) #AUC,異常度の可視化

    else:
        file_root = args.ROOT+args.SNR+args.MACHINE #trainデータのrootを取得
        # test_root = args.ROOT+args.MACHINE+'/test/' #trainデータのrootを取得
        output_path = args.OUTPUT_PATH+args.MACHINE +'/' + args.OPTIM 
        # model_weights=output_path+'/{}/UNet.h5'.format(CHECK_POINT)
        model = models.UNet() #モデル作成
        # model = models.AE() #モデル作成
        # model = models.Dual_model()
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model.to(device)
        model_weight = './{}/{}/model.pth'.format(output_path,CHECK_POINT)
        model.load_state_dict(torch.load(model_weight))
        section_list = ['section_00','section_01','section_02'] #DCASEにはセクションというものが存在するため、用意していますが実際に動かすときは関係ないです
        mean_=np.load(output_path+'/{}/mean.npy'.format(CHECK_POINT))
        std_=np.load(output_path+'/{}/std.npy'.format(CHECK_POINT))
        domain_list = ['source','target']
        model.eval()
        # for section in section_list:
        #     for domain in domain_list:
        #         normal_path = glob.glob(test_root + '*{}*{}*{}*'.format(section,domain,args.NORMAL)) #正常データ
        #         anomaly_path = glob.glob(test_root + '*{}*{}*{}*'.format(section,domain,args.ANOMALY)) #異常データ
        #         normal_testfiles=sorted(normal_path)
        #         anomaly_testfiles=sorted(anomaly_path)

        #         normal=[]
        #         anomaly=[]
        #         com.anomaly_score(mean_,std_,normal_testfiles,output_path,CHECK_POINT,normal,model,args.FRAME_SHIFT_LENGTH,args.FRAME_LENGTH,args.BINS,args.PATCH_NUM,MASK_NUM,args.GIF) #正常データの異常度算出
        #         com.anomaly_score(mean_,std_,anomaly_testfiles,output_path,CHECK_POINT,anomaly,model,args.FRAME_SHIFT_LENGTH,args.FRAME_LENGTH,args.BINS,args.PATCH_NUM,MASK_NUM,args.GIF) #異常データの異常度算出
        #         com.visualize(normal,anomaly,output_path,CHECK_POINT,section,domain) #AUC,異常度の可視化
        for id_ in ID_:
            train_path,normal_path,anomaly_path = com.make_path_list(file_root,id_,args.SEED)
            test = com.testset(anomaly_path,args.FRAME_LENGTH,args.FRAME_SHIFT_LENGTH,args.BINS,output_path,CHECK_POINT)
            if id_ == 'id_00':
                mean_ = torch.from_numpy((mean_).astype(np.float32)).clone()
                std_ = torch.from_numpy((std_).astype(np.float32)).clone()
            test = (test - mean_) / std_
            # np.random.shuffle(test)
            # test = test[:int(len(test)//10)]
            length = int(len(test)//len(anomaly_path))
            mi = np.zeros((test.shape[0],248))
            mi = np.zeros((test.shape[0],496))
            # mi = np.zeros((test.shape[0],496*2))
            # mi = np.zeros((test.shape[0],128))]
            scam = Smooth.SmoothGrad(model,device)
            if 'id_00' == id_:
                label = 0
            elif 'id_02' == id_:
                label = 1 
            elif 'id_04' == id_:
                label = 2 
            elif 'id_06' == id_:
                label = 3
          
            # sys.exit()
            for i in tqdm(range(len(train_path))):
                inputs = test[i*length:(i+1)*length].to(device)
                # copy_ = inputs.clone()
                # out,hidden_half = model([data,copy_])
                output,hidden_half,clf,_,f = model(inputs)
                cam,cnt = scam(inputs,label)
                inputs = inputs[0].cpu().numpy()
                plt.imshow(np.squeeze(inputs)[:,64::-1].T,extent=[0,64,0,8000],aspect='auto')
                plt.colorbar()
                plt.ylabel('Frequency (Hz)')
                plt.xlabel('Frames')
                plt.savefig('./{}/{}.png'.format(output_path,id_))
                plt.close()
                span = abs(np.percentile(cam, 95))
                vmin = 0
                vmax = span
                cam = np.clip((cam - vmin) / (vmax - vmin), 0, 1)
                plt.imshow(np.squeeze(cam)[:,64::-1].T,extent=[0,64,0,8000],aspect='auto')
                plt.colorbar()
                plt.ylabel('Frequency (Hz)')
                plt.xlabel('Frames')
                plt.savefig('./{}/{}_heat.png'.format(output_path,id_))
                plt.close()
                # mi[i*length:(i+1)*length] = hidden_half.to('cpu').detach().numpy().copy()
                mi[i*length:(i+1)*length] = _.to('cpu').detach().numpy().copy()
                # mi[i*length:(i+1)*length] = clf.to('cpu').detach().numpy().copy()
                # mi[i*length:(i+1)*length] = clf[:,248:].to('cpu').detach().numpy().copy()
                break
            continue
            mu_mat = np.mean(mi,axis=0)
            # cov = np.cov(mi.T)
            # cov = np.linalg.pinv(cov)
            cov = LedoitWolf().fit(mi)
            cov = np.linalg.pinv(cov.covariance_) 
           
            print(mu_mat.shape)
            print(mu_mat)

            print(cov.shape)
            print(cov)

            normal=[]
            anomaly=[]
            normal_middle = com.anomaly_score(mean_,std_,normal_path,output_path,CHECK_POINT,normal,model,args.FRAME_SHIFT_LENGTH,args.FRAME_LENGTH,args.BINS,args.PATCH_NUM,MASK_NUM,args.GIF,device) #正常データの異常度算出
            anomaly_middle = com.anomaly_score(mean_,std_,anomaly_path,output_path,CHECK_POINT,anomaly,model,args.FRAME_SHIFT_LENGTH,args.FRAME_LENGTH,args.BINS,args.PATCH_NUM,MASK_NUM,args.GIF,device) #異常データの異常度算出
            section = None
            domain = None
            com.visualize(normal,anomaly,output_path,CHECK_POINT,section,domain,id_) #AUC,異常度の可視化
            abnormality = np.concatenate([normal_middle,anomaly_middle],axis=0)
            mahala_result = []
            from scipy.spatial import distance
            # for i in abnormality:
            #     # mahala_result.append(distance.mahalanobis(i,mu_mat,cov))
            #     mahala_result.append(distance.euclidean(i,mu_mat))
            #     # print(i)
            # score =[]
            # for i in tqdm(range(len(normal_path)*2)):
            #     score.append(np.mean(mahala_result[length*i:(i+1)*length]))
            # plt.plot(mahala_result)
            # plt.savefig(output_path+"/{}/{}_middle".format(CHECK_POINT,id_))
            # plt.close()
            # com.visualize(score[:len(normal_path)],score[len(normal_path):],output_path,CHECK_POINT,section,domain,id_)
            # from sklearn.mixture import GaussianMixture
            # gmm = GaussianMixture(n_components=10, random_state=0).fit(mi)
            # mahala_result=-gmm.score_samples(abnormality)
            # score =[]
            # for i in tqdm(range(len(normal_path)*2)):
            #     score.append(np.mean(mahala_result[length*i:(i+1)*length]))
            # plt.plot(mahala_result)
            # plt.savefig(output_path+"/{}/{}_middle".format(CHECK_POINT,id_))
            # plt.close()
            # com.visualize(score[:len(normal_path)],score[len(normal_path):],output_path,CHECK_POINT,section,domain,id_)
            
            mahala_result = []
            for i in abnormality:
                mahala_result.append(distance.mahalanobis(i,mu_mat,cov))
                # mahala_result.append(distance.euclidean(i,mu_mat))
                # print(i)
            score =[]
            for i in tqdm(range(len(normal_path)*2)):
                score.append(np.mean(mahala_result[length*i:(i+1)*length]))
            plt.plot(mahala_result)
            plt.savefig(output_path+"/{}/{}_middle".format(CHECK_POINT,id_))
            plt.close()
            com.visualize(score[:len(normal_path)],score[len(normal_path):],output_path,CHECK_POINT,section,domain,id_)
            
            # cov = MinCovDet(support_fraction=1).fit(mi)
            # cov = np.linalg.pinv(cov.covariance_) 
            # from scipy.spatial import distance
            # mahala_result = []
            # for i in abnormality:
            #     mahala_result.append(distance.mahalanobis(i,mu_mat,cov))
            #     # mahala_result.append(distance.euclidean(i,mu_mat))
            #     # print(i)
            # score =[]
            # for i in tqdm(range(len(normal_path)*2)):
            #     score.append(np.mean(mahala_result[length*i:(i+1)*length]))
            # plt.plot(mahala_result)
            # plt.savefig(output_path+"/{}/{}_middle".format(CHECK_POINT,id_))
            # plt.close()
            # com.visualize(score[:len(normal_path)],score[len(normal_path):],output_path,CHECK_POINT,section,domain,id_)
            # cov = np.cov(mi.T)
            # cov = np.linalg.pinv(cov)
            
            # mahala_result = []
            # for i in abnormality:
            #     mahala_result.append(distance.mahalanobis(i,mu_mat,cov))
            #     # mahala_result.append(distance.euclidean(i,mu_mat))
            #     # print(i)
            # score =[]
            # for i in tqdm(range(len(normal_path)*2)):
            #     score.append(np.mean(mahala_result[length*i:(i+1)*length]))
            # plt.plot(mahala_result)
            # plt.savefig(output_path+"/{}/{}_middle".format(CHECK_POINT,id_))
            # plt.close()
            # com.visualize(score[:len(normal_path)],score[len(normal_path):],output_path,CHECK_POINT,section,domain,id_)
            # print(type(abnormality))
            # print(abnormality.shape)
            # from sklearn.manifold import TSNE
            # X = TSNE(n_components=2,init='random').fit_transform(abnormality)
            # plt.scatter(X[:len(normal_middle),0],X[:len(normal_middle),1],s=0.5,label='normal',c='b',alpha=0.5)
            # plt.scatter(X[len(normal_middle):,0],X[len(normal_middle):,1],s=0.5,label='anomaly',c='r',alpha=0.5)
            # plt.legend()
            # plt.savefig(output_path+"/{}/{}_middle".format(CHECK_POINT,id_))
            # plt.close()
    

def memo():
    mem = psutil.virtual_memory() 
    print(mem.total)
    #[結果] 8492281856

    # メモリ使用量を取得 
    print(mem.used)
    #[結果] 4748627968

    # メモリ空き容量を取得 
    print(mem.available)
    #[結果] 3743653888

if __name__ == "__main__":
    main(args)
