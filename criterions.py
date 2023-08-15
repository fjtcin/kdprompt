import torch
from torch import nn

class CosineSimilarityLoss(nn.Module):
    def __init__(self):
        super(CosineSimilarityLoss, self).__init__()

    def forward(self, logits, prompts, labels):
        logits_n = nn.functional.normalize(logits)
        prompts_n = nn.functional.normalize(prompts)
        labels_n = nn.functional.normalize(torch.full_like(labels, 0.5) - labels)
        res = logits_n @ prompts_n.t() * labels_n
        return res.mean()

class CustomKLDivLoss(nn.Module):
    def __init__(self):
        super(CustomKLDivLoss, self).__init__()
        # self.kl_div_loss = torch.nn.KLDivLoss(reduction="batchmean", log_target=True)

    def forward(self, logits, prompts, labels):
        #TODO use GNN's promts to teach MLP's promts?
        # return self.kl_div_loss(logits.log_softmax(dim=1), labels.log_softmax(dim=1))
        logits_n = nn.functional.normalize(logits)
        labels_n = nn.functional.normalize(labels)
        return (logits_n-labels_n).abs().mean()