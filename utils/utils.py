import numpy as np
import tensorboardX

import torch
from torch import nn
from PIL import Image

from torch.nn.modules import loss
from torch.nn.modules import activation

from torch.autograd import Variable
import torch.optim.lr_scheduler as sched

import scipy.misc

from io import BytesIO


class Identity(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, *input):
        return input

class TensorBoard:
    def __init__(self, model_dir):
        self.model_dir = model_dir
        self.summary_writer = tensorboardX.SummaryWriter(model_dir)

    def image_summary(self, tag, values, step):
        if not isinstance(values, list):
            values = [values]

        for idx, img in enumerate(values):

            if type(img) == str:
                img = np.array(Image.open(img))
            elif isinstance(img, np.ndarray):
                pass
            else:
                img = scipy.misc.toimage(img)

            self.summary_writer.add_image(f"{tag}/{idx}", img, global_step=step)

    def scalar_summary(self, tag, value, step):
        self.summary_writer.add_scalar(tag, value, global_step=step)


def isnan(tensor):
    return np.isnan(tensor.cpu().data.numpy()).sum() > 0


def size(p): return np.prod(p.size())


def where(cond, one, another):
    """
    Improvised where expression for pytorch.
    Please note: it's up to user, to satisfy tensor datatypes equality condition.

    :param cond: tensor with boolean values
    :param one: tensor
    :param another: tensor
    :return: tensor with values of one tensor on indices
             where cond is 1 and values of another everywhere else.
    """
    return (cond * one) + ((1 - cond) * another)


def embedding_dropout(embedding, input, dropout=0.1, scale=None):
    if 0 < dropout < 1:
        mask = embedding.weight.data.new().resize_((embedding.weight.size(0), 1)) \
                .bernoulli_(1 - dropout).expand_as(embedding.weight) / (1 - dropout)
        mask = Variable(mask)
        masked_embed_weight = mask * embedding.weight

        if scale:
            masked_embed_weight = scale.expand_as(masked_embed_weight) * masked_embed_weight

        padding_idx = embedding.padding_idx or -1
        return embedding._backend.Embedding.apply(input, masked_embed_weight,
                                               padding_idx, embedding.max_norm, embedding.norm_type,
                                               embedding.scale_grad_by_freq, embedding.sparse)

    return embedding(input)


def get_optimizer(name):
    """
    Returns optimizer by name:

    :param name: optimizer name (str)
    :return: corresponding optimizer from torch.optim module.
    """
    return {
        'asgd': torch.optim.ASGD,
        'adadelta': torch.optim.Adadelta,
        'adagrad': torch.optim.Adagrad,
        'adam': torch.optim.Adam,
        'adamax': torch.optim.Adamax,
        'lbfgs': torch.optim.LBFGS,
        'rmsprop': torch.optim.RMSprop,
        'rprop': torch.optim.Rprop,
        'sgd': torch.optim.SGD,
        'sparseadam': torch.optim.SparseAdam
    }[name.strip().lower()]


def get_activation(name, **kwargs):
    """
    Returns activation by name:

    :param name: activation name (str)
    :param kwargs: kwargs passed to activation constructor.
    :return: corresponding activation from torch.nn module.
    """
    return {
        'elu': activation.ELU(**kwargs),
        'glu': activation.GLU(**kwargs),
        'hardshrink': activation.Hardshrink(**kwargs),
        'hardtanh': activation.Hardtanh(**kwargs),
        'leakyrelu': activation.LeakyReLU(**kwargs),
        'logsigmoid': activation.LogSigmoid(),
        'logsoftmax': activation.LogSoftmax(**kwargs),
        'prelu': activation.PReLU(**kwargs),
        'parameter': activation.Parameter(**kwargs),
        'rrelu': activation.RReLU(**kwargs),
        'relu': activation.ReLU(**kwargs),
        'relu6': activation.ReLU6(**kwargs),
        'selu': activation.SELU(**kwargs),
        'sigmoid': activation.Sigmoid(),
        'softmax': activation.Softmax(**kwargs),
        'softmax2d': activation.Softmax2d(),
        'softmin': activation.Softmin(**kwargs),
        'softplus': activation.Softplus(**kwargs),
        'softshrink': activation.Softshrink(**kwargs),
        'softsign': activation.Softsign(),
        'tanh': activation.Tanh(),
        'tanhshrink': activation.Tanhshrink(),
        #'threshold': activation.Threshold(**kwargs),
        'identity': Identity()
    }[name.strip().lower()]


def get_criterion(name, **kwargs):
    """
    Returns criterion by name.

    :param name: criterion name (str)
    :param kwargs: kwargs passed to criterion constructor.
    :return: corresponding criterion from torch.nn module.
    """
    return {
        'bce': loss.BCELoss(),
        'bcewithlogits': loss.BCEWithLogitsLoss(),
        'cosineembedding': loss.CosineEmbeddingLoss(),
        'crossentropy': loss.CrossEntropyLoss(),
        'hingeembedding': loss.HingeEmbeddingLoss(),
        'kldiv': loss.KLDivLoss(),
        'l1': loss.L1Loss(),
        'mse': loss.MSELoss(),
        'marginranking': loss.MarginRankingLoss(),
        'multilabelmargin': loss.MultiLabelMarginLoss(),
        'multilabelsoftmargin': loss.MultiLabelSoftMarginLoss(),
        'multimargin': loss.MultiMarginLoss(),
        'nll': loss.NLLLoss(),
        'nll2d': loss.NLLLoss2d(),
        'poissonnll': loss.PoissonNLLLoss(),
        'smoothl1': loss.SmoothL1Loss(),
        'softmargin': loss.SoftMarginLoss(),
        'tripletmargin': loss.TripletMarginLoss()
    }[name.strip().lower()]


def get_scheduler(name):
    """
    Returns learning rate scheduler by name.

    :param name: name of the scheduler (without 'LR').
    :return: scheduler.
    """
    return {
        'cosineannealing': CosineAnnealingLR,
        'exponential': sched.ExponentialLR,
        'lambda': sched.LambdaLR,
        'multistep': sched.MultiStepLR,
        'reduceonplateau': sched.ReduceLROnPlateau,
        'step': sched.StepLR
    }[name.strip().lower()]


def get_variable(inputs, cuda=False, tensor_type=torch.Tensor, **kwargs):
    """
    Wraps Tensors, lists and numpy.ndarrays into a torch.autograd.Variable.

    :param inputs: torch.Tensor, list or numpy.ndarrays
    :param cuda: wheter to use cuda (bool), or device to place on (int).
    :param kwargs: optional kwargs passed to Variable constructor.
    :return: Variable with inputs as data.
    """
    if isinstance(inputs, Variable):
        return get_variable(inputs.data.cpu(), cuda, **kwargs)

    if type(inputs) in [list, np.ndarray]:
        inputs = tensor_type(inputs)

    out = Variable(inputs, **kwargs)
    if isinstance(cuda, bool) and cuda:
        out = out.cuda()
    elif isinstance(cuda, int) and cuda >= 0:
        out = out.cuda(cuda)

    return out


def place(tensor, device=-1):
    """
    Places tensor on `device`.

    :param tensor: tensor.
    :param device: device id (-1 for CPU).
    :return: placed tensor.
    """

    if device < 0:
        return tensor.cpu()
    else:
        return tensor.cuda(device)


def linear_schedule(starting, final, epochs):
    delta = (final-starting) / epochs
    return lambda epoch: starting + epoch*delta


def exp_schedule(starting, final, epochs):
    mult = np.exp(np.log(final/starting)/epochs)
    return lambda epoch: starting * np.power(mult, epoch)


def get_np(inputs):
    if inputs.is_cuda:
        inputs = inputs.cpu()
    if isinstance(inputs, Variable):
        inputs = inputs.data
    if torch.is_tensor(inputs):
        inputs = inputs.numpy()
    return inputs