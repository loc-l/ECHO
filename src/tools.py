import torch.nn.functional as F
import torch


def get_acc(pred, data):
    accs = []
    correct = pred.eq(data.y)
    for _, mask in data('train_mask', 'val_mask', 'test_mask'):
        accs.append(correct[mask].sum().item() / mask.sum().item())

    return accs

def get_loss(out, data):
    loss = []
    for _, mask in data('train_mask', 'val_mask', 'test_mask'):
        loss.append(F.nll_loss(out[mask].log_softmax(dim=-1), data.y[mask]))
    return loss


@torch.no_grad()
def test(model, data, device, test_loader=None):
    model.eval()
    if test_loader is None:
        out = model(data.x.to(device), data.adj_t.to(device))
    else:
        out = model.inference(data.x, test_loader, device)
    pred = out.argmax(dim=-1).cpu()
    accs = get_acc(pred, data)
    losses = get_loss(out.cpu(), data)
    
    print(f'Loss:train {losses[0]:.4f}, Acc:train/val/test {accs[0]:.4f} {accs[1]:.4f} {accs[2]:.4f}')