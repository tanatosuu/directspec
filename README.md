# DirectSpec
Codes for the follow papers:<br/>
Balancing Embedding Spectrum for Recommendation<br/>

# Environment
* Pytorch+GPU==1.8.0<br/>
* Numpy==1.19.2<br/>
* Pandas==1.1.4<br/>

#Run the Algorithm
- Yelp
```bash
 python 'file_name.py' --dataset='yelp' --embedding_size=64 --lr=10 --reg=0.01 --batch_size=512 --alpha=0.8 --tau=3.0 --shrink_norm=0.0
```

- CiteULike
```bash
 python 'file_name.py' --dataset='citeulike' --embedding_size=64 --lr=180 --reg=0.01 --batch_size=512 --alpha=1.0 --tau=3.0 --shrink_norm=0.03
```

- Gowalla
```bash
 python 'file_name.py' --dataset='gowalla' --embedding_size=64 --lr=200 --reg=0.01 --batch_size=512 --alpha=0.7 --tau=4.0 --shrink_norm=0.02
```
