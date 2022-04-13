# A Template Implementation of GraphSage model

This repo contains a template of GraphSage model, in which the nodes do not have to have an initial feature. To achieve this, we use an Embedding network provided by PyTorch to get the initial features of nodes, which can be tuned during the training process. 

## graph.py

The file `graph.py` contains the dataset loader. The dataset loader reads the edge file and transforms it into a node map and adjacent list. Also, it will split the whole dataset into three parts, namely training, validation and test. As a template, we just use the whole dataset as the training set.

## model.py

The file `model.py` contains the GraphSage model. We make some modifications based on the [PyTorch implementation of GraphSAGE](https://github.com/twjiang/graphSAGE-pytorch/tree/e9a05cafec31b51a23679dbe7fa2baeea95ee35d) such that the current version can handle the graphs without raw features of nodes. To achieve this, we add an Embedding layer before the first Sage layer. 

## utils.py

The file `utils.py` contains the training part of the GraphSage model. As a template, we drop the optimization part. In order to complete the downstream tasks, after we last Sage layer, we should feed the nodes' embeddings as input of the downstream networks, which depends on different applications.

## main.py

To run the codes, we just tap `python3 main.py` in the terminal. You can change the arguments based on your preference. 
