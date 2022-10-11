import torch
import torch.nn.functional as F


def binary_classification_loss(concat_true, concat_pred):
    t_true = concat_true[:, 1]

    t_pred = concat_pred[:, 2]
    t_pred = (t_pred + 0.001) / 1.002

    losst = torch.sum(F.binary_cross_entropy(t_pred, t_true,reduction='none'))

    return losst



def regression_loss(concat_true, concat_pred):
    y_true = concat_true[:, 0]
    t_true = concat_true[:, 1]

    y0_pred = concat_pred[:, 0]
    y1_pred = concat_pred[:, 1]

    loss0 = torch.sum((1. - t_true) * torch.square((y_true - y0_pred)))
    loss1 = torch.sum(t_true * torch.square((y_true - y1_pred)))

    return loss0 + loss1


def ned_loss(concat_true, concat_pred):
    t_true = concat_true[:, 1]

    t_pred = concat_pred[:, 1]
    return torch.sum(F.binary_cross_entropy(t_pred, t_true, reduction='none'))


def dead_loss(concat_true, concat_pred):
    return regression_loss(concat_true, concat_pred)


def dragonnet_loss_binarycross(concat_true, concat_pred):
    return regression_loss(concat_true, concat_pred) + binary_classification_loss(concat_true, concat_pred)


def treatment_accuracy(concat_true, concat_pred):
    t_true = concat_true[:, 1]
    t_pred = concat_pred[:, 2]
    return t_pred.argmax(dim=1).eq(t_true).sum().item()


def track_epsilon(concat_true, concat_pred):
    epsilons = concat_pred[:, 3]
    return torch.abs(torch.mean(epsilons))


def make_tarreg_loss(ratio=1., dragonnet_loss=dragonnet_loss_binarycross):
    def tarreg_ATE_unbounded_domain_loss(concat_true, concat_pred):
        vanilla_loss = dragonnet_loss(concat_true, concat_pred)

        y_true = concat_true[:, 0]
        t_true = concat_true[:, 1]

        y0_pred = concat_pred[:, 0]
        y1_pred = concat_pred[:, 1]
        t_pred = concat_pred[:, 2]

        epsilons = concat_pred[:, 3]
        t_pred = (t_pred + 0.01) / 1.02
        # t_pred = tf.clip_by_value(t_pred,0.01, 0.99,name='t_pred')

        y_pred = t_true * y1_pred + (1 - t_true) * y0_pred

        h = t_true / t_pred - (1 - t_true) / (1 - t_pred)

        y_pert = y_pred + epsilons * h
        targeted_regularization = torch.sum(torch.square(y_true - y_pert))

        # final
        loss = vanilla_loss + ratio * targeted_regularization
        return loss

    return tarreg_ATE_unbounded_domain_loss
