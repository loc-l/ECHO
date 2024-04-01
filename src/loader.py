import torch
import numpy as np
import time

class SubgraphSampler(torch.utils.data.DataLoader):
    def __init__(self, data, num_parts, batch_size, shuffle, num_workers):
        self.data = data
        partcount = np.array([data.num_nodes // num_parts] * num_parts)
        partcount[:data.num_nodes - (data.num_nodes // num_parts)*num_parts] += 1
        partptr = [0]*(num_parts+1)
        for i in range(1, num_parts+1):
            partptr[i] = partptr[i-1]+partcount[i-1]


        self.partptr = torch.tensor(partptr)

        if batch_size > 1:
            super().__init__(range(self.partptr.numel() - 1), collate_fn=self.__collate__,
                            batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
        else:
            t1 = time.time()
            data_list = list(torch.utils.data.DataLoader(range(self.partptr.numel() - 1), collate_fn=self.__collate__, 
                            batch_size=batch_size, shuffle=True, num_workers=num_workers))
            t2 = time.time()
            print(f'Processing time for $s=1$: {t2-t1}s')
            super().__init__(data_list, batch_size=1,
                             collate_fn=lambda x: x[0], shuffle=shuffle, num_workers=num_workers)

    def __collate__(self, batch):
        if not isinstance(batch, torch.Tensor):
            batch = torch.tensor(batch)
        batch, _ = torch.sort(batch)

        start = self.partptr[batch].tolist()
        end = self.partptr[batch + 1].tolist()
        node_idx = torch.cat([torch.arange(s, e) for s, e in zip(start, end)])
        train_mask = self.data['train_mask'][node_idx]
        
        if torch.count_nonzero(train_mask) == 0:
            return None

        adj, _ = self.data.adj_t.saint_subgraph(node_idx)
        xs = self.data.x[node_idx]
        ys = self.data.y[node_idx][train_mask]
        
        return (train_mask,adj,xs,ys)