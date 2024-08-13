# GraphSAGE

### 半监督运行命令和结果


cora，Test accuarcy: 0.8653

```
python train_graphsage.py --data_path "cora"  --device 'cuda' --batch_size 128 --hidden_dim 32 --k 2 --num_sample 8 --dropout 0.6 --agg "mean" --concat "False" --activation "True" --bias "True" --test_model "best"
```


citeseer，Test accuarcy: 0.7523

```
python train_graphsage.py --data_path "citeseer"  --device 'cuda' --batch_size 512 --hidden_dim 64 --k 1 --num_sample 10 --dropout 0.6 --agg "gcn" --concat "False" --activation "True" --bias "False" --test_model "best" --patience 50
```


pubmed，Test accuarcy: 0.8806

```
python train_graphsage.py --data_path "pubmed"  --device 'cuda' --batch_size 64 --hidden_dim 64 --k 2 --num_sample 8 --dropout 0.6 --agg "mean" --concat "False" --activation "True" --bias "False" --test_model "best"
```


### 半监督

cora，Test accuarcy: 0.7660

```
python train_graphsage.py --data_path "cora"  --device 'cuda' --batch_size 128 --hidden_dim 32 --k 2 --num_sample 8 --dropout 0.6 --agg "mean" --concat "False" --activation "True" --bias "True" --test_model "best" --use_semi_supervised "True"
```


citeseer，Test accuarcy: 0.6660

```
python train_graphsage.py --data_path "citeseer"  --device 'cuda' --batch_size 512 --hidden_dim 64 --k 1 --num_sample 10 --dropout 0.6 --agg "gcn" --concat "False" --activation "True" --bias "False" --test_model "best" --patience 50 --use_semi_supervised "True"
```


pubmed，Test accuarcy: 0.7480

```
python train_graphsage.py --data_path "pubmed"  --device 'cuda' --batch_size 64 --hidden_dim 64 --k 2 --num_sample 8 --dropout 0.6 --agg "mean" --concat "False" --activation "True" --bias "False" --test_model "best" --use_semi_supervised "True"
```
