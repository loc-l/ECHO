# ECHO+

- [src](https://github.com/loc-l/ECHO/blob/echoplus/src) contains the code of  *ECHO[+]*
- [src/AES.py](https://github.com/loc-l/ECHO/blob/echoplus/src/AES.py) contains the implementation of *AESTrainer* and *AESTrainerPlus*.

## Prerequisites
- torch 1.10.0
- torch_geometric 2.0.4
- ogb 1.3.5
- fast_sampler from [SALIENT](https://github.com/MITIBMxGraph/SALIENT)

## Example Usage
[train_arxiv_sage_with_echo.ipynb](https://github.com/loc-l/ECHO/blob/echoplus/src/train_arxiv_sage_with_echo.ipynb) is an example for training SAGE on ogbn-arxiv for *ECHO*.
[train_products_sage_with_echoplus.ipynb](https://github.com/loc-l/ECHO/blob/echoplus/src/train_products_sage_with_echoplus.ipynb) is an example for training SAGE on ogbn-products for *ECHO+*.

- Use ```from config.${dataset}_${model} import *``` to include different configurations. We have put all configurations in [config](https://github.com/loc-l/ECHO/blob/echoplus/src/config).
- For small datasets on SAGE, you can use ```test_loader=None```. 
- For other cases, to avoid OOM please use:
```
test_loader = NeighborSampler(data.adj_t, node_idx=None, sizes=[-1],
                              batch_size=4096, shuffle=False,
                              num_workers=12, return_e_id=False)
```


## Reference
- https://github.com/MITIBMxGraph/SALIENT
- https://github.com/pyg-team/pytorch_geometric
