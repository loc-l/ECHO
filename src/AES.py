import math
import torch
from tools import *


class AESTrainer:
    def __init__(self, args, cc_loader, ss_loader, test_loader=None):
        self.args = args
        self.cc_loader = cc_loader
        self.ss_loader = ss_loader
        self.test_loader = test_loader
        self.results = []


    def cc(self, model, device, optimizer):
        model.train()
        total_loss = total_nodes = 0
        for batch_size, n_id, adjs in self.cc_loader:
            xs = self.ss_loader.data.x[n_id]
            ys = self.ss_loader.data.y[n_id[:batch_size]]

            xs = xs.to(device, non_blocking=True)
            ys = ys.to(device, non_blocking=True)
            adjs = [adj.to(device, non_blocking=True) for adj in adjs]

            optimizer.zero_grad()
            out = model(xs, adjs)
            loss = F.nll_loss(out, ys)
            loss.backward()
            optimizer.step()

            total_loss += float(loss) * batch_size
            total_nodes += batch_size
        return total_loss / total_nodes
    

    def ss(self, model, device, optimizer):
        model.train()
        total_loss = total_nodes = 0
        for batch in self.ss_loader:
            if batch is None:
                continue
            train_mask, adj, xs, ys = batch

            xs = xs.to(device, non_blocking=True)
            ys = ys.to(device, non_blocking=True)
            adj = adj.to(device, non_blocking=True)

            optimizer.zero_grad()
            out = model(xs, adj)
            loss = F.nll_loss(out[train_mask], ys)
            loss.backward()
            optimizer.step()

            nodes = train_mask.sum().item()
            total_loss += loss.item() * nodes
            total_nodes += nodes
        return total_loss / total_nodes
    
    
    @torch.no_grad()
    def test(self, model, device, e):
        model.eval()
        if self.test_loader is None:
            out = model(self.ss_loader.data.x.to(device), self.ss_loader.data.adj_t.to(device))
        else:
            out = model.inference(self.ss_loader.data.x, self.test_loader, device)
        pred = out.argmax(dim=-1).cpu()
        accs = get_acc(pred, self.ss_loader.data)
        losses = get_loss(out.cpu(), self.ss_loader.data)
        accs.append(e)
        self.results.append(accs)
        print(f'Epoch {e:3d}, Loss:train {losses[0]:.4f}, Acc:train/val/test {accs[0]:.4f} {accs[1]:.4f} {accs[2]:.4f}')


    def run(self, model, device, optimizer):
        e = 1
        last_loss, tmp_loss = 1000000, 1000000

        while(1):
            tmp_loss = self.cc(model, device, optimizer)
            if self.args.save_dir=='':
                self.test(model, device, e)
            else:
                torch.save(model, f'{self.args.save_dir}/model-{e}.pt')
                print(f'Epoch {e:3d}, Approx. Loss {tmp_loss:.4f}')
            e+=1
            if last_loss-tmp_loss < self.args.xi:
                break
            last_loss = tmp_loss

        tmp_loss = self.ss(model, device, optimizer)

        k = math.ceil(tmp_loss / last_loss)
        k = e - 1 + k
        for e in range(e+1, 1+self.args.epochs):
            if e==k or e==self.args.epochs:
                last_loss = self.cc(model, device, optimizer)
                if self.args.save_dir=='':
                    self.test(model, device, e)
                else:
                    torch.save(model, f'{self.args.save_dir}/model-{e}.pt')
                    print(f'Epoch {e:3d}, Approx. Loss {last_loss:.4f}')
            else:
                tmp_loss = self.ss(model, device, optimizer)  
                if e>k:
                    k = math.ceil(tmp_loss/last_loss)
                    k = e-1+k  
        
        self.final_results()

    
    def final_results(self):
        if self.args.save_dir=='':
            results = torch.tensor(self.results)
            argmax = results[:, 1].argmax().item()
            print(f'Final(highest valid):\nEpoch {int(results[argmax, 3])}, Train {results[argmax, 0]:.4f}, Valid {results[argmax, 1]:.4f}, Test {results[argmax, 2]:.4f}')
        else:
            print(f'Please use test fuction to evaluate saved models.')


    def clear(self):
        self.results.clear()