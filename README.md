## AHFAN

This project implements the paper "Graph Anomaly Detection Based on Hybrid Node Representation Learning" published in Neural Networks (https://doi.org/10.1016/j.neunet.2025.107169).


## Model Usage

### Dependencies 

This project is tested on cuda 11.6 with several dependencies listed below:

```markdown
pytorch=1.11.0
torch-geometric=2.0.4
```


### Dataset 

Public datasets Elliptic, Yelp and Weibo used for graph anomaly detection are available for evaluation. `Elliptic` was first proposed in [this paper](https://arxiv.org/pdf/2008.08692.pdf), of which goal is to detect money-laundering users in bitcoin network.
### Usage
```
python train.py --dataset weibo/yelp/elliptic
```

Tuned hyper-parameters could be found in `config.py`

### Run on your own dataset

You could organize your dataset into a `torch_geometric.data.Data` then add profile of your own dataset on `config.py`
