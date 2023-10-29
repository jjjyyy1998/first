import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--EPOCHS', type=int, default=20) # エポック数
parser.add_argument('--FIL', type=int, default=16) # フィルターの初期値
parser.add_argument('--BINS', type=int, default=64) # 周波数ビンの数
parser.add_argument('--BATCH_SIZE', type=int, default=64) # バッチサイズ 
parser.add_argument('--FRAME_LENGTH','-FL', type=int, default=128) # フレーム長
parser.add_argument('--FRAME_SHIFT_LENGTH', '-FSL', type=int, default=64) # フレームシフト長
parser.add_argument('--PERCENTAGE', '-P', type=float, default=0.75) # マスクの割合 0<p<1の範囲で
parser.add_argument('--MASK_TYPE', '-MT', type=str, default='pixel') # パッチ数、[4,16,64,256] pixelのときは次の行で勝手に書き換えられます
parser.add_argument('--PATCH_NUM', '-PN', type=int, default=64) # 入力画像の縦サイズ
parser.add_argument('--NORMAL', type=str, default='normal') # テストデータのうち正常データを判断するキーワード
parser.add_argument('--ANOMALY', type=str, default='anomaly') # テストデータのうち異常データを判断するキーワード
parser.add_argument('--OUTPUT_PATH', type=str, default='../result/') # 結果保存パス
# parser.add_argument('--ROOT', type=str, default='../../DCASE/dataset/') # データセットのパス
parser.add_argument('--ROOT', type=str, default='../../MIMII') # データセットのパス
parser.add_argument('--MACHINE', type=str, default=None) # DCASEのデータの場合、どの機械を使うか指定する
parser.add_argument('--GIF', type=bool, default=False) # GIF画像
parser.add_argument('--SNR', type=str, default='/6db/') # GIF画像
parser.add_argument('--SEED',type=int,default=42)
parser.add_argument('--ID',type=str,default=None)
parser.add_argument('--OPTIM','-O',type=str,default='')




args = parser.parse_args()
print(args)
if args.MASK_TYPE in 'pixel':
    args.PATCH_NUM = args.BINS**2
MASK_NUM = int(args.PATCH_NUM*args.PERCENTAGE) # いくつのパッチ(ピクセル)にマスクをしたか

CHECK_POINT = "model_epochs={}_filter={}_bins={}_frame_length={}_shift={}_mask_num={}_patch_num={}_type={}".format(args.EPOCHS,args.FIL,args.BINS,args.FRAME_LENGTH,args.FRAME_SHIFT_LENGTH,MASK_NUM,args.PATCH_NUM,args.MASK_TYPE) # モデルの保存名等に使われます
# ID_ =['id_00','id_02','id_04','id_06']
if args.ID == None: 
    ID_ =['id_00','id_02','id_04','id_06']
else:
    ID_ =[args.ID]