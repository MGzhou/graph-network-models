# GCN

### 半监督运行命令和结果

数据集

cora， Test accuarcy: 0.8150

```
python train_gcn.py --data_path "cora"  --device 'cuda' --layers 2 --hidden_dim 32 --test_model 'best'
```

citeseer， Test accuarcy: 0.7180

```
python train_gcn.py --data_path "citeseer"  --device 'cuda' --layers 2 --hidden_dim 64 --test_model 'best'
```

pubmed，Test accuarcy: 0.7910

```
python train_gcn.py --data_path "pubmed"  --device 'cuda' --layers 2 --hidden_dim 32 --test_model 'best'
```

### 监督学习

cora，Test accuarcy: 0.8875

```
python train_gcn.py --data_path "cora"  --device 'cuda' --layers 2 --hidden_dim 32 --test_model 'best' --use_semi_supervised 'False'
```

citeseer，Test accuarcy: 0.7673

```
python train_gcn.py --data_path "citeseer"  --device 'cuda' --layers 2 --hidden_dim 64 --test_model 'best' --use_semi_supervised 'False'
```

pubmed，Test accuarcy: 0.8651

```
python train_gcn.py --data_path "pubmed"  --device 'cuda' --layers 2 --hidden_dim 32 --test_model 'best'  --use_semi_supervised 'False'
```
