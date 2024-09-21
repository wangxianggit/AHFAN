## AHFAN

This project implements the Graph Anomaly Detection Based on Hybrid Node Representation Learning, which focuses on graph anomaly detection with GNNs addressing class inconsistency and semantic inconsistency.


## Model Usage

### Dependencies 

This project is tested on cuda 11.6 with several dependencies listed below:

```markdown
pytorch=1.11.0
torch-geometric=2.0.4
```


### Dataset 

Two public datasets Elliptic and Weibo of graph anomaly detection are available for evaluation. `Elliptic` was first proposed in [this paper](https://arxiv.org/pdf/2008.08692.pdf), of which goal is to detect money-laundering users in bitcoin network.
### Usage
```
python main.py --dataset weibo/elliptic
```

Tuned hyper-parameters could be found in `config.py`

### Run on your own dataset

You could organize your dataset into a `torch_geometric.data.Data` then add profile of your own dataset on `config.py`
