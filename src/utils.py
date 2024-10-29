import torch
import tqdm
from unimol_tools import utils
from unimol_tools.data import DataHub
from sklearn.metrics import roc_auc_score, auc, precision_recall_curve


def train(model, dataloader, optimizer, criterion, device):
    model.train()
    preds = []
    labels = []
    losses = 0
    
    for batch in tqdm.tqdm(dataloader):
        src_tokens, src_distance, src_coord, src_edge_type, esm1, esm2, auxiliary, num_resi, label = [x.to(device) for x in batch]
        pred = model(src_tokens, src_distance, src_coord, src_edge_type, esm1, esm2, auxiliary, num_resi).squeeze()
        
        preds.append(pred.detach().cpu())
        labels.append(label.detach().cpu())
        
        loss = criterion(pred, label)
        losses += loss.detach().item()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    
    print('Loss:\t{}'.format(losses/len(dataloader)))
    
    labels = torch.cat(labels, dim=0).numpy()
    preds = torch.cat(preds, dim=0).numpy()
    return labels, preds, losses/len(dataloader)


def predict(model, dataloader, device):
    model.eval()
    preds = []
    labels = []
    
    with torch.no_grad():
        for batch in tqdm.tqdm(dataloader):
            src_tokens, src_distance, src_coord, src_edge_type, esm1, esm2, auxiliary, num_resi, label = [x.to(device) for x in batch]
            pred = model(src_tokens, src_distance, src_coord, src_edge_type, esm1, esm2, auxiliary, num_resi).squeeze()
            
            preds.append(pred.detach().cpu())
            labels.append(label.detach().cpu())
            
    labels = torch.cat(labels).numpy()
    preds = torch.cat(preds).numpy() 
    return labels, preds


def performance_evaluation(label, pred):
    pred_scores = torch.sigmoid(torch.from_numpy(pred))
    roc_auc = roc_auc_score(label, pred_scores)
    prec, reca, _ = precision_recall_curve(label, pred_scores)
    aupr = auc(reca, prec)
    return roc_auc, aupr


def batch_collate_fn(samples):    
    src_tokens = torch.stack(tuple(utils.pad_1d_tokens([s[0] for s in samples], pad_idx=0.0)))
    src_distance = torch.stack(tuple(utils.pad_2d([s[1] for s in samples], pad_idx=0.0)))
    src_coord = torch.stack(tuple(utils.pad_coords([s[2] for s in samples], pad_idx=0.0)))
    src_edge_type = torch.stack(tuple(utils.pad_2d([s[3] for s in samples], pad_idx=0.0)))

    esm1 = torch.stack([s[4] for s in samples])
    esm2 = torch.stack([s[5] for s in samples])
    auxiliary = torch.stack([s[6] for s in samples])
    num_resi = torch.stack([s[7] for s in samples])
    
    label = torch.FloatTensor([s[8] for s in samples])
        
    return src_tokens, src_distance, src_coord, src_edge_type, esm1, esm2, auxiliary, num_resi, label