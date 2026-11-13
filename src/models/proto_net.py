import torch
import torch.nn.functional as F

def compute_prototypes(support_emb, n_way, k_shot):

    D = support_emb.shape[-1] # support_emb = (n_way * k_shot, 128). D = 128
    
    return support_emb.view(n_way, k_shot, D).mean(dim=1) # view() is used to reshape according to n_way & k_shot. ex: (20, 128) -> (4, 5, 128). compute mean across dim = 1, i.e k_shot dimension.

def prototypical_loss(model, support, query, labels, n_way, k_shot):
    C, H, W = support.shape[2], support.shape[3], support.shape[4] # shape of support = (n_way, k_shot, C, H, W). C = 3, H = 224, W = 224.
    support_flattened = support.view(n_way * k_shot, C, H, W) # shape = (4, 5, 3, 224, 224) -> (20, 224, 224)
    support_embeddings = model(support_flattened) # generate embeddings. (20, 3, 224, 224) -> (20, 128)

    prototypes = compute_prototypes(support_embeddings, n_way, k_shot)

    query_embeddings = model(query) # shape = (20, 128)

    dists = torch.cdist(query_embeddings, prototypes, p = 2) # compute pairwise euclidean distances (p = 2) between query embeddings (n_way * k_shot, 128) & prototypes (n_way, 128)

    log_probs = F.log_softmax(-dists, dim = 1) # smaller distance = better, so negate the distances. softmax converts scores to probabilities.

    loss = F.nll_loss(log_probs, labels) # labels contain correct local class for each query. if low probability is assigned to correct class -> loss becomes high.

    predictions = log_probs.argmax(dim = 1) # choose the class with highest log probability 
    accuracy = (predictions == labels).float().mean().item() # check wherever prediction matches label, convert bool -> float, and compute mean. item() converts one element torch tensor -> float.

    return loss, accuracy, prototypes.detach()