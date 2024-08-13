# node2vec

### 半监督运行命令和结果

数据集

cora，Test accuarcy: 0.6760

```
python train_node2vec.py --data_path "cora"  --device 'cuda' --embed_size 128 --walk_length 8 --num_walks 50 --p 0.55 --q 0.55 --test_model 'last'
```


citeseer，Test accuarcy: 0.4460

```
python train_node2vec.py --data_path "citeseer"  --device 'cuda' --embed_size 128 --walk_length 10 --num_walks 50 --p 0.25 --q 0.25 --test_model 'best'
```


pubmed，Test accuarcy: 0.6860

```
python train_node2vec.py --data_path "pubmed"  --device 'cuda' --embed_size 128 --walk_length 10 --num_walks 100 --p 0.25 --q 0.25 --test_model 'best'
```


### 监督学习

cora，Test accuarcy: 0.7601

```
python train_node2vec.py --data_path "cora"  --device 'cuda' --embed_size 128 --walk_length 8 --num_walks 50 --p 0.55 --q 0.55 --test_model 'last' --use_semi_supervised 'False'
```


citeseer，Test accuarcy: 0.5495

```
python train_node2vec.py --data_path "citeseer"  --device 'cuda' --embed_size 128 --walk_length 10 --num_walks 50 --p 0.25 --q 0.25 --test_model 'best' --use_semi_supervised 'False'
```


pubmed，Test accuarcy: 0.7928

```
python train_node2vec.py --data_path "pubmed"  --device 'cuda' --embed_size 128 --walk_length 10 --num_walks 100 --p 0.25 --q 0.25 --test_model 'best' --use_semi_supervised 'False'
```
