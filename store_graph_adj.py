import torch
import argparse
import numpy as np
import pandas as pd
import gc

#torch.cuda.set_device(1)
parser = argparse.ArgumentParser(description='Argument parser for the program.')

parser.add_argument('--dataset', type=str, default='yelp', help='dataset')
parser.add_argument('--layer', type=int, default='3', help='the number layers')

args = parser.parse_args()
 
dataset = args.dataset
layer= args.layer



if dataset=='citeulike':
	user,item=5551,16980

elif dataset=='yelp':
	user,item=25677,25815

elif dataset=='Gowalla':
	user,item=29858,40981
else:
	 raise NotImplementedError(f"Dataset Error!")


df_train=pd.read_csv('./'+dataset+r'/train.csv')




rate_matrix=torch.zeros(user+item,user+item).cuda()

for row in df_train.itertuples():
	rate_matrix[row[1],user+row[2]]=1
	rate_matrix[user+row[2],row[1]]=1


D=1/rate_matrix.sum(1).pow(0.5)
D[D==float('inf')]=0




L=(D.unsqueeze(1)*(rate_matrix)*D.unsqueeze(0))




del rate_matrix, D 
gc.collect()
torch.cuda.empty_cache()



exp_L=torch.eye(L.shape[0]).cuda()

L1=L

#start=time.time()
for i in range(layer):
	exp_L=exp_L+L1
	L1=L1.mm(L)

#end=time.time()



np.save('./'+dataset+r'/adj_store.npy',exp_L.cpu().numpy())
