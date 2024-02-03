import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import pandas as pd
import torch.nn.functional as F
import time
import gc

#citeulike
user,item=5551,16980
#yelp
#user,item=25677,25815
#gowalla
#user,item=29858,40981




dataset='./citelikeu'


result=[]
#load train test data

df_train=pd.read_csv(dataset+r'/train.csv')
df_test=pd.read_csv(dataset+r'/test.csv')

#load the  data
train_samples=0
#train_data=[[] for i in range(user)]
test_data=[[] for i in range(user)]
for row in df_train.itertuples():
	train_samples+=1
for row in df_test.itertuples():
	test_data[row[1]].append(row[2])





class DirectSpec(nn.Module):

	def __init__(self, user_size, item_size, latent_size=64, alpha=1.0, tau=3,shrink_norm=0.1, batch_size=256,learning_rate=180,reg=0.01):
		super(DirectSpec, self).__init__()
		

		self.user_size=user_size
		self.item_size=item_size
		self.latent_size=latent_size
		self.alpha=alpha
		self.tau=tau
		self.shrink_norm=shrink_norm
		self.batch_size=batch_size
		self.lr=learning_rate
		self.reg=reg
		self.rate_matrix=torch.Tensor(np.load(dataset+r'/rate.npy')).cuda()
		self.L=torch.Tensor(np.load(dataset+r'/adjacency.npy')).cuda()

		self.embed=Variable(torch.nn.init.uniform_(torch.randn(user_size+item_size,latent_size),-np.sqrt(6. / (user_size+item_size+latent_size) ) ,np.sqrt(6. / (user_size+item_size+latent_size) )).cuda(),requires_grad=True)



	def sampling(self):

		u=np.random.randint(0,self.user_size,self.batch_size)
		p=torch.multinomial(self.rate_matrix[u],1,True).squeeze(1)

		return u, p


	def forward(self,u,p):

		final_user,final_pos=self.L[u].mm(self.embed),self.L[self.user_size+p].mm(self.embed)

		final_user=final_user /final_user.norm(2,dim=1).unsqueeze(1)
		final_pos=final_pos/final_pos.norm(2,dim=1).unsqueeze(1)
		
		
		final_user = final_user - self.alpha* F.softmax(final_user.mm(final_user.t())*self.tau,1).mm(final_user)
		final_pos = final_pos - self.alpha* F.softmax(final_pos.mm(final_pos.t())*self.tau,1).mm(final_pos)
			

		final_user=final_user*(final_user.norm(2,dim=1).unsqueeze(1)*self.shrink_norm)
		final_pos=final_pos*(final_pos.norm(2,dim=1).unsqueeze(1)*self.shrink_norm)
		
		
		out=((final_user*final_pos).sum(1)).sigmoid()
		
		reg=self.reg*(final_user**2+final_pos**2).sum()

		return (-torch.log(out).sum()+reg)/self.batch_size

	def update(self):
		self.embed-=self.lr*self.embed.grad
		self.embed.grad.zero_()


	def predict(self):

		out=self.L.mm(self.embed)
		return out[:user].mm(out[user:].t()).sigmoid()-self.rate_matrix*1000




	def test(self):
		#calculate idcg@k(k={1,...,20})
		def cal_idcg(k=20):
			idcg_set=[0]
			scores=0.0
			for i in range(1,k+1):
				scores+=1/np.log2(1+i)
				idcg_set.append(scores)

			return idcg_set

		def cal_score(topn,now_user,trunc=20):
			dcg10,dcg20,hit10,hit20=0.0,0.0,0.0,0.0
			for k in range(trunc):
				max_item=topn[k]
				if test_data[now_user].count(max_item)!=0:
					if k<=10:
						dcg10+=1/np.log2(2+k)
						hit10+=1
					dcg20+=1/np.log2(2+k)
					hit20+=1

			return dcg10,dcg20,hit10,hit20



		#accuracy on test data
		ndcg10,ndcg20,recall10,recall20=0.0,0.0,0.0,0.0
		predict=self.predict()
		

		idcg_set=cal_idcg()
		for now_user in range(user):
			test_lens=len(test_data[now_user])

			#number of test items truncated at k
			all10=10 if(test_lens>10) else test_lens
			all20=20 if(test_lens>20) else test_lens
		
			#calculate dcg
			topn=predict[now_user].topk(20)[1]

			dcg10,dcg20,hit10,hit20=cal_score(topn,now_user)


			ndcg10+=(dcg10/idcg_set[all10])
			ndcg20+=(dcg20/idcg_set[all20])
			recall10+=(hit10/all10)
			recall20+=(hit20/all20)			

		ndcg10,ndcg20,recall10,recall20=round(ndcg10/user,4),round(ndcg20/user,4),round(recall10/user,4),round(recall20/user,4)
		print(ndcg10,ndcg20,recall10,recall20)

		result.append([ndcg10,ndcg20,recall10,recall20])


model=DirectSpec(user,item)

epoch=train_samples//model.batch_size

for i in range(390):
	total_loss=0.0
	#start=time.time()
	for j in range(0,epoch):

		u,p=model.sampling()

		loss=model(u,p)

		loss.backward()
		with torch.no_grad():
			model.update()

		total_loss+=loss.item()
		

	#end=time.time()
	#print(end-start)
	print('epoch %d training loss:%f' %(i,total_loss/epoch))
	model.lr*=0.9995
	
	if (i+1)%10==0 and (i+1)>=50 :
		model.test()
	


output=pd.DataFrame(result)
output.to_csv(r'./directspec.csv')

