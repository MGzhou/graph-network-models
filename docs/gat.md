## GAT-model

### 半监督运行命令与结果

数据集

cora，Test accuarcy: 0.8280

```
python train_gat.py --data_path "cora"  --device 'cuda' --num_heads 8 --hidden_dim 16 --dropout 0.6 --alpha 0.2 --residual "False" --bias "True" --test_model "best"
```


citeseer，Test accuarcy: 0.7130

```
python train_gat.py --data_path "citeseer" --device 'cuda' --num_heads 6 --hidden_dim 32 --dropout 0.6 --alpha 0.2 --residual "False" --bias "True" --test_model "best"
```


pubmed

```
python train_gat.py --data_path "pubmed"  --device 'cuda' --num_heads 6 --hidden_dim 16 --dropout 0.6 --alpha 0.2 --residual "False" --bias "True" --test_model "best"
```

超出显存

### 监督学习

cora，Test accuarcy: 0.8616

```
python train_gat.py --data_path "cora"  --device 'cuda' --num_heads 8 --hidden_dim 16 --dropout 0.6 --alpha 0.2 --residual "False" --bias "True" --test_model "best" --use_semi_supervised 'False'
```


citeseer，Test accuarcy: 0.7447

```
python train_gat.py --data_path "citeseer" --device 'cuda' --num_heads 8 --hidden_dim 32 --dropout 0.6 --alpha 0.2 --residual "False" --bias "True" --test_model "best" --use_semi_supervised 'False'
```
