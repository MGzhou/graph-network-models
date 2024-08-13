# 训练

### 半监督运行命令和结果

cora，Test accuarcy:0.6770

```
python train_deepwalk.py --data_path "cora"  --device 'cuda' --embed_size 128 --walk_length 10 --num_walks 80 --test_model 'last'
```


citeseer，Test accuarcy:0.4420

```
python train_deepwalk.py --data_path "citeseer"  --device 'cuda' --embed_size 128 --walk_length 10 --num_walks 50 --test_model 'best'
```


pubmed，Test accuarcy:0.6620

```
python train_deepwalk.py --data_path "pubmed"  --device 'cuda' --embed_size 128 --walk_length 16 --num_walks 120 --test_model 'best'
```


### 监督结果

cora，Test accuarcy: 0.7952

```
python train_deepwalk.py --data_path "cora"  --device 'cuda' --embed_size 128 --walk_length 10 --num_walks 80 --test_model 'last' --use_semi_supervised 'False'
```


citeseer，Test accuarcy: 0.5405

```
python train_deepwalk.py --data_path "citeseer"  --device 'cuda' --embed_size 128 --walk_length 10 --num_walks 50 --test_model 'best' --use_semi_supervised 'False'
```


pubmed，Test accuarcy: 0.7959

```
python train_deepwalk.py --data_path "pubmed"  --device 'cuda' --embed_size 128 --walk_length 16 --num_walks 120 --test_model 'best' --use_semi_supervised 'False'
```
