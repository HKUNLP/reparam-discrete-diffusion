import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
logger = logging.getLogger(__name__)

def topk_masking(scores, cutoff_len, stochastic=False, temp=1.0):
    """
    scores: [b, n]
    cutoff_len: [b, 1]
    stochastic: bool, whether to add noise to select top_k or not
    returns:
        mask: [b, n], with 1 if the token is in top-k lowest scores, 0 otherwise
    """
    if stochastic:
        gumbel_noise = -torch.log(-torch.log(torch.rand_like(scores) + 1e-8) + 1e-8)
        _scores = scores + temp * gumbel_noise
    else:
        _scores = scores
    sorted_index = _scores.sort(-1)[0]
    cutoff = sorted_index.gather(dim=-1, index=cutoff_len) + 1e-10
    # cutoff_len = k -> select k + 1 tokens
    masking = _scores < cutoff
    return masking

def new_arange(x, *size):
    """
    Return a Tensor of `size` filled with a range function on the device of x.
    If size is empty, using the size of the variable x.
    """
    if len(size) == 0:
        size = x.size()
    return torch.arange(size[-1], device=x.device).expand(*size).contiguous()

def sample_bernoulli(log_prob, u=None):
    """
        return boolean tensor b with the same shape as log prob.
    """
    if u is None:
        u = torch.rand_like(log_prob)
    b = torch.log(u.clamp(min=1e-30)) < log_prob
    return b

def log_sample_categorical(logits, uniform_noise=None, eps=1e-10, return_log=True, num_classes=None):
    if uniform_noise is None:
        uniform_noise = torch.rand_like(logits)
    else:
        assert uniform_noise.shape == logits.shape
    gumbel_noise = -torch.log(-torch.log(uniform_noise + eps) + eps)
    x = (gumbel_noise + logits).argmax(dim=-1)
    if return_log:
        assert num_classes is not None
        log_x = index_to_log_onehot(x, num_classes)
        return log_x
    else:
        return x

def multinomial_kl(log_prob1, log_prob2):
    kl = (log_prob1.exp() * (log_prob1 - log_prob2)).sum(dim=-1)
    return kl

def log_categorical(log_x_0, log_prob):
    return (log_x_0.exp() * log_prob).sum(dim=-1)

def index_to_log_onehot(x, num_classes):
    assert x.max().item() < num_classes, \
        f'Error: {x.max().item()} >= {num_classes}'
    x_onehot = F.one_hot(x, num_classes) # [b, n, c]
    log_x = torch.log(x_onehot.float().clamp(min=1e-30))
    return log_x

def log_onehot_to_index(log_x):
    return log_x.argmax(-1)

def mean_ds(x, dim=None):
    return (
        x.float().mean().type_as(x)
        if dim is None
        else x.float().mean(dim).type_as(x)
    )

def expand_gather(a, t):
    b, *_ = t.shape
    non_batch_shape = a.shape[1:]
    t = t.reshape(b, *((1,) * len(non_batch_shape))).repeat((1,) + non_batch_shape)
    out = torch.gather(a, 0, t)
    return out

def log_sub_exp(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
        this requires a > b
        computes log (exp(a) - exp(b))
        = log (exp(a)(1 - exp(b - a)))
        = log (exp(a)(1 - exp(b - a)))
        = a + log (1 - exp(b - a))
    """
    return a + log1mexp(b - a)

def log1mexp(a: torch.Tensor) -> torch.Tensor:
    '''
        compute log(1 - exp(a)) adapted from 
        https://github.com/HEmile/storchastic/blob/e7d8f64a3316c20973e002fc8beaf228b45297d7/storch/sampling/swor.py#L228
        we must have 1 - e^a > 0, implying a < 0.
    '''
    c = -0.693
    a1 = -a.abs()
    eps = 1e-6
    return torch.where(a1 > c, torch.log(-a1.expm1() + eps), torch.log1p(-a1.exp() + eps))

def log_matmul_exp(log_A: torch.Tensor, log_B: torch.Tensor) -> torch.Tensor:
    """
        adapted from https://stackoverflow.com/a/60731666
        computes (log_A.exp() @ log_B.exp()).log() in a numerically stable way.
        this requires N x M x K time/space complexity.
    """
    N, M = log_A.shape
    M_, K = log_B.shape
    assert M == M_
    log_A_expanded = log_A.unsqueeze(-1).expand((N, M, K))
    log_B_expanded = log_B.unsqueeze(0).expand((N, M, K))
    log_pairwise_products = log_A_expanded + log_B_expanded
    return torch.logsumexp(log_pairwise_products, dim=-2)
