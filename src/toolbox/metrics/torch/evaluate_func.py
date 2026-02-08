# All evaluate functions used in evaluate_on_batch().
import torchmetrics
from einops import rearrange

import torch


def microf1(input_data, target, single_mask, num_classes, **kwargs):
    masked_input_data = input_data.masked_fill(~single_mask, num_classes)
    masked_target = target.masked_fill(~single_mask, num_classes)

    return torchmetrics.functional.classification.multiclass_f1_score(
        masked_input_data,
        masked_target,
        num_classes=num_classes + 1,
        multidim_average="samplewise",
        average="micro",
        ignore_index=num_classes,
    )


def macrof1(input_data, target, single_mask, num_classes, **kwargs):
    masked_input_data = input_data.masked_fill(~single_mask, num_classes)
    masked_target = target.masked_fill(~single_mask, num_classes)

    return torchmetrics.functional.classification.multiclass_f1_score(
        masked_input_data,
        masked_target,
        num_classes=num_classes + 1,
        multidim_average="samplewise",
        average="macro",
        ignore_index=num_classes,
    )


def acc(input_data, target, single_mask, num_classes, **kwargs):
    masked_input_data = input_data.masked_fill(~single_mask, num_classes)
    masked_target = target.masked_fill(~single_mask, num_classes)

    return torchmetrics.functional.classification.multiclass_accuracy(
        masked_input_data,
        masked_target,
        num_classes=num_classes + 1,
        multidim_average="samplewise",
        average="macro",
        ignore_index=num_classes,
    )


def top_k(input_data, target, single_mask, num_classes, **kwargs):
    masked_input_data = input_data.masked_fill(~single_mask.unsqueeze(dim=-1), num_classes)
    masked_target = target.masked_fill(~single_mask, num_classes)
    masked_input_data = rearrange(masked_input_data, 'b s ne -> b ne s')

    top_k_acc_single_event_seq = []
    for k in range(num_classes - 1):
        top_k_acc_single_event_seq.append(
            torchmetrics.functional.classification.multiclass_accuracy(
                masked_input_data,
                masked_target,
                num_classes=num_classes,
                multidim_average="samplewise",
                average="micro",
                ignore_index=num_classes,
                top_k=k + 1,
            )
        )

    return torch.stack(top_k_acc_single_event_seq).transpose(0, 1)


def l1(input_data, target, mask, timestamp, **kwargs):
    from .l1 import l1

    return l1(input_data, target, mask, timestamp)


def l1_self(input_data, target, mask, timestamp, **kwargs):
    from .l1 import l1_self

    return l1_self(input_data, mask, timestamp)


def spearman(input_data, target, single_mask, *args, **kwargs):
    results = []
    for input_data_per_seq, target_per_seq, single_mask_per_seq in zip(input_data, target, single_mask):
        results.append(
            torchmetrics.functional.spearman_corrcoef(
                input_data_per_seq[single_mask_per_seq].flatten(), target_per_seq[single_mask_per_seq].flatten()
            )
        )
    return torch.stack(results)


def spearman_self(input_data, target, single_mask, *args, **kwargs):
    results = []
    num_marks = input_data.shape[-1]
    for input_data_per_seq, single_mask_per_seq in zip(input_data, single_mask):
        picked_input_data_per_seq = input_data_per_seq[single_mask_per_seq]
        picked_input_data_per_seq = rearrange(picked_input_data_per_seq, "sl r ne -> (sl r) ne")

        spearman_matrix_per_seq = torch.zeros(num_marks, num_marks, device=input_data.device)

        for i in range(num_marks):
            for j in range(num_marks):
                spearman_matrix_per_seq[i][j] = torchmetrics.functional.spearman_corrcoef(
                    picked_input_data_per_seq[:, i], picked_input_data_per_seq[:, j]
                )
        results.append(spearman_matrix_per_seq)
    return torch.stack(results)


def pearson(input_data, target, single_mask, *args, **kwargs):
    results = []
    for input_data_per_seq, target_per_seq, single_mask_per_seq in zip(input_data, target, single_mask):
        results.append(
            torchmetrics.functional.pearson_corrcoef(
                input_data_per_seq[single_mask_per_seq].flatten(), target_per_seq[single_mask_per_seq].flatten()
            )
        )
    return torch.stack(results)


def pearson_self(input_data, target, single_mask, *args, **kwargs):
    results = []
    num_marks = input_data.shape[-1]
    for input_data_per_seq, single_mask_per_seq in zip(input_data, single_mask):
        picked_input_data_per_seq = input_data_per_seq[single_mask_per_seq]
        picked_input_data_per_seq = rearrange(picked_input_data_per_seq, "sl r ne -> (sl r) ne")

        spearman_matrix_per_seq = torch.zeros(num_marks, num_marks, device=input_data.device)

        for i in range(num_marks):
            for j in range(num_marks):
                spearman_matrix_per_seq[i][j] = torchmetrics.functional.pearson_corrcoef(
                    picked_input_data_per_seq[:, i], picked_input_data_per_seq[:, j]
                )
        results.append(spearman_matrix_per_seq)
    return torch.stack(results)


evaluate_func_dict = {
    "acc": acc,
    "macro-f1": macrof1,
    "micro-f1": microf1,
    "top_k": top_k,
    "l1": l1,
    "pearson": pearson,
    "spearman": spearman,
    "l1_self": l1_self,
    "pearson_self": pearson_self,
    "spearman_self": spearman_self,
}


def evaluate_func(name):
    return evaluate_func_dict[name]
