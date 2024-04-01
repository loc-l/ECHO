# ECHO

- [supplyment.pdf](https://github.com/loc-l/ECHO/blob/main/supplyment.pdf) includes the proof of theorems and additional experimental results of *ECHO*.
- [src](https://github.com/loc-l/ECHO/blob/main/src) contains the code of  *ECHO*.

## Prerequisites
- torch 1.10.0
- torch_geometric 2.0.4
- ogb 1.3.5
- fast_sampler from [SALIENT](https://github.com/MITIBMxGraph/SALIENT)

## Example Usage
[train_arxiv_sage.ipynb](https://github.com/loc-l/ECHO/blob/main/src/train_arxiv_sage.ipynb) is an example for training SAGE on ogbn-arxiv.

- Use ```from config.${dataset}_${model} import *``` to include different configurations. We have put all configurations in [config](https://github.com/loc-l/ECHO/blob/main/src/config).
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
