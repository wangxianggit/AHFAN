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

Three public datasets Elliptic, Weibo and Yelp of graph anomaly detection are available for evaluation. `Elliptic` was first proposed in [this paper](https://arxiv.org/pdf/2008.08692.pdf), of which goal is to detect money-laundering users in bitcoin network.
### Usage
python main.py --dataset weibo/yelp/elliptic

