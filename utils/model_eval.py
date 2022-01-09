import torch
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def mae(y_true, y_pred):
    
    return np.mean(np.abs(y_true-y_pred))

def q2(y_true, y_pred):
    
    press = np.sum((y_pred-y_true)**2)
    tss=np.sum((y_true-y_true.mean())**2)
    return 1-(press/tss)

def train(model, loader, loss_f, optimizer, dummy=None):
    
    for batch in loader:
        batch.to(device)
        optimizer.zero_grad()

        pred, embedding = model(batch.x.float(), batch.edge_index, 
                                batch.edge_attr.float(), batch.batch,
                               batch.desc.float())

        loss = torch.sqrt(loss_f(pred.flatten(), batch.y))

        loss.backward()

        optimizer.step()
    return loss

def dro_train(model, loader, loss_f, optimizer, lambda_min):
    """
    Training via Distributionally Robust Optimization.
    See: https://arxiv.org/pdf/2003.00688.pdf for more details.

    """
    optimizer.zero_grad()
    y_true = []
    preds = []
    distribution_ids = []
    lambda_min = lambda_min

    for batch in loader:
        batch.to(device)

        pred, embedding = model(batch.x.float(), batch.edge_index, 
                                batch.edge_attr.float(), batch.batch,
                                batch.desc.float())
        preds.append(pred)
        y_true.append(batch.y.float())
        distribution_ids.append(batch.cluster.float())

    preds = torch.cat(preds)
    y_true = torch.cat(y_true)
    distribution_ids = torch.cat(distribution_ids)

    partial_losses = []
    for dist_id in torch.unique(distribution_ids):
        mask = dist_id==distribution_ids
        loss = torch.sqrt(loss_f(preds.flatten()[mask], y_true[mask]))
        partial_losses.append(loss)

    partial_losses = torch.stack(partial_losses)

    loss = (1 - lambda_min) * partial_losses.max() + lambda_min * partial_losses.sum()
    loss.backward()

    optimizer.step()
    return loss

def predict(model, loader):
    
    y_pred_mean, y_pred_std, y_true = [], [], []
        
    with torch.no_grad():
        for batch in loader:
            batch.to(device)
            
            mc_dropout_tries = []
            for _ in range(6):
                pred, embedding = model(batch.x.float(), batch.edge_index, 
                                        batch.edge_attr.float(), batch.batch,
                                        batch.desc.float())
                mc_dropout_tries.append(pred)
            
            tries_mean = torch.mean(torch.cat(mc_dropout_tries, axis=-1), axis=1)
            tries_std = torch.std(torch.cat(mc_dropout_tries, axis=-1), axis=1)
            
            y_pred_mean.append(tries_mean)
            y_pred_std.append(tries_std)
            
            y_true.append(batch.y)

        y_pred_mean = torch.cat(y_pred_mean).float().numpy().ravel()
        y_pred_std = torch.cat(y_pred_std).float().numpy().ravel()
        y_true = torch.cat(y_true).float().numpy().ravel()
        
    return y_true, y_pred_mean, y_pred_std

def metric(model, loader):
    
    y, y_pred, y_pred_std = predict(model, loader)
    q2_score = q2(y, y_pred)
    
    confidence_mask = y_pred_std <= np.quantile(y_pred_std, 0.25)
    q2_confident = q2(y[confidence_mask], y_pred[confidence_mask])
    
    return q2_score, q2_confident

