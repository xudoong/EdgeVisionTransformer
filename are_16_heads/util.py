import torch


def head_entropy(p):
    plogp = p * torch.log(p)
    plogp[p == 0] = 0
    return - plogp.sum(dim=-1)


def head_pairwise_kl(p):
    # p has shape bsz x nheads x L x L and is normalized in the last
    # dim
    logp = torch.log(p)
    logp[p == 0] = -40
    H_pq = -torch.einsum("bilk,bjlk->bijl", [p, logp])
    H_p = head_entropy(p).unsqueeze(-2)
    KL = H_pq - H_p
    KL.masked_fill_(p.sum(-1).eq(0).unsqueeze(1), 0.0)
    KL.masked_fill_(p.sum(-1).eq(0).unsqueeze(2), 0.0)
    return KL


def attn_disagreement(p):
    # p has shape bsz x nheads x L x L and is normalized in the last
    # dim
    n_heads = p.size(1)
    return torch.einsum("bilk,bjlk->b", [p, p]) / n_heads ** 2


def out_disagreement(out):
    # out has shape bsz x nheads x L x d
    n_heads = out.size(1)
    # Normalize
    out /= torch.sqrt((out ** 2).sum(-1)).unsqueeze(-1) + 1e-20
    cosine = torch.einsum("bild,bjld->b", [out, out])
    return cosine / n_heads ** 2


def print_1d_tensor(tensor):
    print("\t".join(f"{x:.5f}" for x in tensor.cpu().data))


def print_2d_tensor(tensor):
    for row in range(len(tensor)):
        print_1d_tensor(tensor[row])


def none_if_empty(string):
    return string if string != "" else None
