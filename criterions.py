import torch
from torch import nn

class CosineSimilarityLoss(nn.Module):
    def __init__(self):
        super(CosineSimilarityLoss, self).__init__()
        self.kl_div_loss = nn.KLDivLoss(reduction="batchmean", log_target=True)
        self.mse_loss = nn.MSELoss(reduction="sum")
        self.cos_loss = nn.CosineEmbeddingLoss()

    def js_divergence(self, log_p, log_q):
        m = 0.5 * (torch.exp(log_p) + torch.exp(log_q))
        log_m = m.log()
        return 0.5 * (self.kl_div_loss(log_m, log_p) + self.kl_div_loss(log_m, log_q))

    def forward(self, logits, prompts, labels):
        logits_n = nn.functional.normalize(logits)
        prompts_n = nn.functional.normalize(prompts)
        return self.kl_div_loss((logits_n @ prompts_n.mT).log_softmax(dim=1), labels.log_softmax(dim=1))
        # return self.mse_loss((logits_n @ prompts_n.mT), labels) / labels.size(0)
        # return self.cos_loss((logits_n @ prompts_n.mT), labels, torch.tensor([1], device='cuda:0'))
        return self.js_divergence((logits_n @ prompts_n.mT).log_softmax(dim=1), labels.log_softmax(dim=1))


class CustomKLDivLoss(nn.Module):
    def __init__(self):
        super(CustomKLDivLoss, self).__init__()
        self.kl_div_loss = nn.KLDivLoss(reduction="batchmean", log_target=True)
        self.mse_loss = nn.MSELoss(reduction="sum")
        self.cos_loss = nn.CosineEmbeddingLoss()

    def js_divergence(self, log_p, log_q):
        m = 0.5 * (torch.exp(log_p) + torch.exp(log_q))
        log_m = m.log()
        return 0.5 * (self.kl_div_loss(log_m, log_p) + self.kl_div_loss(log_m, log_q))

    def forward(self, logits, prompts, labels):
        # return self.kl_div_loss(logits.log_softmax(dim=1), labels.log_softmax(dim=1))
        # return self.mse_loss(logits, labels) / logits.size(0)
        # return self.cos_loss(logits, labels, torch.tensor([1], device='cuda:0'))
        return self.js_divergence(logits.log_softmax(dim=1), labels.log_softmax(dim=1))
