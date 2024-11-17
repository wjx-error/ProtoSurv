try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper
from collections import OrderedDict
import torch
from torch import Tensor
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# c 0确认死亡 1存活
def nll_loss(hazards, S, Y, c, alpha=0.4, eps=1e-7):
    batch_size = len(Y)
    Y = Y.view(batch_size, 1)  # ground truth bin, 1,2,...,k
    c = c.view(batch_size, 1).float()  # censorship status, 0 or 1
    if S is None:
        S = torch.cumprod(1 - hazards, dim=1)  # surival is cumulative product of 1 - hazards
    # without padding, S(0) = S[0], h(0) = h[0]
    S_padded = torch.cat([torch.ones_like(c), S], 1)  # S(-1) = 0, all patients are alive from (-inf, 0) by definition
    # after padding, S(0) = S[1], S(1) = S[2], etc, h(0) = h[0]

    # uncensored_loss 确认死亡事件发生前面不为0+
    uncensored_loss = -(1 - c) * (torch.log(torch.gather(S_padded, 1, Y).clamp(min=eps)) + torch.log(
        torch.gather(hazards, 1, Y).clamp(min=eps)))

    # censored_loss 存活时不为0
    censored_loss = - c * torch.log(torch.gather(S_padded, 1, Y + 1).clamp(min=eps))

    neg_l = censored_loss + uncensored_loss
    loss = (1 - alpha) * neg_l + alpha * uncensored_loss  # alpha一般是0
    loss = loss.mean()
    return loss


def ce_loss(hazards, S, Y, c, alpha=0.4, eps=1e-7):
    batch_size = len(Y)
    Y = Y.view(batch_size, 1)  # ground truth bin, 1,2,...,k
    c = c.view(batch_size, 1).float()  # censorship status, 0 or 1
    if S is None:
        S = torch.cumprod(1 - hazards, dim=1)  # surival is cumulative product of 1 - hazards
    # without padding, S(0) = S[0], h(0) = h[0]
    # after padding, S(0) = S[1], S(1) = S[2], etc, h(0) = h[0]
    # h[y] = h(1)
    # S[1] = S(1)
    S_padded = torch.cat([torch.ones_like(c), S], 1)
    reg = -(1 - c) * (
            torch.log(torch.gather(S_padded, 1, Y) + eps) + torch.log(torch.gather(hazards, 1, Y).clamp(min=eps)))
    ce_l = - c * torch.log(torch.gather(S, 1, Y).clamp(min=eps)) - (1 - c) * torch.log(
        1 - torch.gather(S, 1, Y).clamp(min=eps))
    loss = (1 - alpha) * ce_l + alpha * reg
    loss = loss.mean()
    return loss


class CrossEntropySurvLoss(object):
    def __init__(self, alpha=0.15):
        self.alpha = alpha

    def __call__(self, hazards, S, Y, c, alpha=None, **kwargs):
        if alpha is None:
            return ce_loss(hazards, S, Y, c, alpha=self.alpha)
        else:
            return ce_loss(hazards, S, Y, c, alpha=alpha)


# loss_fn(hazards=hazards, S=S, Y=Y_hat, c=c, alpha=0)
class NLLSurvLoss(object):
    def __init__(self, alpha=0.15):
        self.alpha = alpha

    def __call__(self, hazards, S, Y, c, alpha=None, **kwargs):
        if alpha is None:
            return nll_loss(hazards, S, Y, c, alpha=self.alpha)
        else:
            return nll_loss(hazards, S, Y, c, alpha=alpha)
    # h_padded = torch.cat([torch.zeros_like(c), hazards], 1)
    # reg = - (1 - c) * (torch.log(torch.gather(hazards, 1, Y)) + torch.gather(torch.cumsum(torch.log(1-h_padded), dim=1), 1, Y))


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class CoxPHLoss(torch.nn.Module):
    """Loss for CoxPH model. If data is sorted by descending duration, see `cox_ph_loss_sorted`.

    We calculate the negative log of $(\frac{h_i}{\sum_{j \in R_i} h_j})^d$,
    where h = exp(log_h) are the hazards and R is the risk set, and d is event.

    We just compute a cumulative sum, and not the true Risk sets. This is a
    limitation, but simple and fast.
    """

    def __init__(self, alpha):
        super().__init__()
        self.alpha = alpha
        print('cox loss alpha:', self.alpha)

    # event_time:T c:E
    def forward(self, S: Tensor, c: Tensor, event_time: Tensor, **kwargs) -> Tensor:
        return self.cox_ph_loss(S, event_time, c)

    def cox_ph_loss(self, log_h: Tensor, durations: Tensor, events: Tensor, eps: float = 1e-5) -> Tensor:
        """Loss for CoxPH model. If data is sorted by descending duration, see `cox_ph_loss_sorted`.

        We calculate the negative log of $(\frac{h_i}{\sum_{j \in R_i} h_j})^d$,
        where h = exp(log_h) are the hazards and R is the risk set, and d is event.

        We just compute a cumulative sum, and not the true Risk sets. This is a
        limitation, but simple and fast.
        """
        idx = durations.sort(descending=True)[1]

        events = events[idx]
        log_h = log_h[idx]

        return self.cox_ph_loss_sorted(log_h, events, eps)

    def cox_ph_loss_sorted(self, log_h: Tensor, events: Tensor, eps: float = 1e-5) -> Tensor:
        """Requires the input to be sorted by descending duration time.
        See DatasetDurationSorted.

        We calculate the negative log of $(\frac{h_i}{\sum_{j \in R_i} h_j})^d$,
        where h = exp(log_h) are the hazards and R is the risk set, and d is event.

        We just compute a cumulative sum, and not the true Risk sets. This is a
        limitation, but simple and fast.
        """

        if events.dtype is not torch.float32:
            events = events.float()

        events = 1 - events  ### censorship的值和event是反过来的
        events[events == 0] = self.alpha

        events = events.view(-1)
        log_h = log_h.view(-1)
        gamma = log_h.max()
        log_cumsum_h = log_h.sub(gamma).exp().cumsum(0).add(eps).log().add(gamma)  # .float() 转float32防nan

        return - log_h.sub(log_cumsum_h).mul(events).sum().div(events.sum())


def l1_reg_all(model, reg_type=None):
    l1_reg = None

    for W in model.parameters():
        if l1_reg is None:
            l1_reg = torch.abs(W).sum()
        else:
            l1_reg = l1_reg + torch.abs(W).sum()  # torch.abs(W).sum() is equivalent to W.norm(1)
    return l1_reg


# prototype loss
# prototypes (b,Class*K,d)
def get_compatibility_loss_batch(data_WSI, prototypes, num_classes):  # (N,dim), (Class*K,d), train_labels, Class
    loss = []
    for idx, data in enumerate(data_WSI):
        pp = get_compatibility_loss(data.x, prototypes[idx], data.patch_classify_type, num_classes)
        loss.append(pp)
    loss = sum(loss) / len(loss)
    return loss


def get_compatibility_loss(x, prototypes, labels, num_classes):  # (N,dim), (Class*K,d), train_labels, Class

    x = x.to(prototypes.device)
    labels = labels.to(prototypes.device)

    # print('prototypes', prototypes.shape)
    # print('x',x.shape)
    # print('labels',labels.shape)
    # print()

    # labels = torch.argmax(labels, dim=1)
    labels = labels.to(torch.int64)
    dim = x.size(-1)  # d
    dots = []
    prototypes = prototypes.reshape(num_classes, -1, dim)  # .max(dim=1)[0] # (Class, K, d)
    for i in range(prototypes.size(1)):  # K
        prototypes_i = prototypes[:, i]  # (Class, d)
        dots_i = torch.cdist(x, prototypes_i, p=2)  # (N,Class)
        dots.append(dots_i.unsqueeze(1))  # (N,1,Class)

    # dots = torch.cat(dots, dim=1).max(dim=1)[0]  # (N,K,Class) -> (N,Class)
    dots = torch.cat(dots, dim=1).mean(dim=1)  # (N,K,Class) -> (N,Class)

    attn = dots  # .softmax(dim=1) # n x c
    positives = torch.gather(input=attn[:, :], dim=1, index=labels[:, None]).squeeze()  # (N)
    negatives = attn[F.one_hot(labels) == 0]
    comp_loss = torch.sum(positives) + torch.logsumexp(-negatives, dim=0)
    comp_loss = comp_loss / (x.size(0) * prototypes.size(0))
    return comp_loss


def get_orthogonal_regularization_loss(s):  # (Class*K,d)
    # Orthogonality regularization.
    s = s.unsqueeze(0) if s.dim() == 2 else s  # (1,Class*K,d)
    k = s.size(-1)  # d
    ss = torch.matmul(s.transpose(1, 2), s)  # (1,d,d)
    i_s = torch.eye(k).type_as(ss)  # (d,d)对角1
    ortho_loss = torch.norm(ss / torch.norm(ss, dim=(-1, -2), keepdim=True) - i_s / torch.norm(i_s), dim=(-1, -2))
    ortho_loss = torch.mean(ortho_loss)
    return ortho_loss


def get_loss(args):
    print('\nInit loss function...', end=' ')
    if args.bag_loss == 'ce':
        loss_fn = CrossEntropySurvLoss(alpha=args.alpha_surv)
    elif args.bag_loss == 'nll':
        loss_fn = NLLSurvLoss(alpha=args.alpha_surv)
    elif args.bag_loss == 'cox':
        loss_fn = CoxPHLoss(alpha=args.alpha_surv)
    else:
        raise NotImplementedError

    print('Done')
    return loss_fn
