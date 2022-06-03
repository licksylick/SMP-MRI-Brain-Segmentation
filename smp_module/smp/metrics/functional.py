import torch
import warnings
from typing import Optional, List, Tuple, Union


__all__ = [
    "get_stats",
    "fbeta_score",
    "f1_score",
    "iou_score",
    "accuracy",
    "precision",
    "recall",
    "sensitivity",
    "specificity",
    "balanced_accuracy",
    "positive_predictive_value",
    "negative_predictive_value",
    "false_negative_rate",
    "false_positive_rate",
    "false_discovery_rate",
    "false_omission_rate",
    "positive_likelihood_ratio",
    "negative_likelihood_ratio",
]


###################################################################################################
# Statistics computation (true positives, false positives, false negatives, false positives)
###################################################################################################


def get_stats(
    output: Union[torch.LongTensor, torch.FloatTensor],
    target: torch.LongTensor,
    mode: str,
    ignore_index: Optional[int] = None,
    threshold: Optional[Union[float, List[float]]] = None,
    num_classes: Optional[int] = None,
) -> Tuple[torch.LongTensor]:
    if torch.is_floating_point(target):
        raise ValueError(f"Target should be one of the integer types, got {target.dtype}.")

    if torch.is_floating_point(output) and threshold is None:
        raise ValueError(
            f"Output should be one of the integer types if ``threshold`` is not None, got {output.dtype}."
        )

    if torch.is_floating_point(output) and mode == "multiclass":
        raise ValueError(f"For ``multiclass`` mode ``target`` should be one of the integer types, got {output.dtype}.")

    if mode not in {"binary", "multiclass", "multilabel"}:
        raise ValueError(f"``mode`` should be in ['binary', 'multiclass', 'multilabel'], got mode={mode}.")

    if mode == "multiclass" and threshold is not None:
        raise ValueError("``threshold`` parameter does not supported for this 'multiclass' mode")

    if output.shape != target.shape:
        raise ValueError(
            "Dimensions should match, but ``output`` shape is not equal to ``target`` "
            + f"shape, {output.shape} != {target.shape}"
        )

    if mode != "multiclass" and ignore_index is not None:
        raise ValueError(f"``ignore_index`` parameter is not supproted for '{mode}' mode")

    if mode == "multiclass" and num_classes is None:
        raise ValueError("``num_classes`` attribute should be not ``None`` for 'multiclass' mode.")

    if ignore_index is not None and 0 <= ignore_index <= num_classes - 1:
        raise ValueError(
            f"``ignore_index`` should be outside the class values range, but got class values in range "
            f"0..{num_classes - 1} and ``ignore_index={ignore_index}``. Hint: if you have ``ignore_index = 0``"
            f"consirder subtracting ``1`` from your target and model output to make ``ignore_index = -1``"
            f"and relevant class values started from ``0``."
        )

    if mode == "multiclass":
        tp, fp, fn, tn = _get_stats_multiclass(output, target, num_classes, ignore_index)
    else:
        if threshold is not None:
            output = torch.where(output >= threshold, 1, 0)
            target = torch.where(target >= threshold, 1, 0)
        tp, fp, fn, tn = _get_stats_multilabel(output, target)

    return tp, fp, fn, tn


@torch.no_grad()
def _get_stats_multiclass(
    output: torch.LongTensor,
    target: torch.LongTensor,
    num_classes: int,
    ignore_index: Optional[int],
) -> Tuple[torch.LongTensor, torch.LongTensor, torch.LongTensor, torch.LongTensor]:

    batch_size, *dims = output.shape
    num_elements = torch.prod(torch.tensor(dims)).long()

    if ignore_index is not None:
        ignore = target == ignore_index
        output = torch.where(ignore, -1, output)
        target = torch.where(ignore, -1, target)
        ignore_per_sample = ignore.view(batch_size, -1).sum(1)

    tp_count = torch.zeros(batch_size, num_classes, dtype=torch.long)
    fp_count = torch.zeros(batch_size, num_classes, dtype=torch.long)
    fn_count = torch.zeros(batch_size, num_classes, dtype=torch.long)
    tn_count = torch.zeros(batch_size, num_classes, dtype=torch.long)

    for i in range(batch_size):
        target_i = target[i]
        output_i = output[i]
        mask = output_i == target_i
        matched = torch.where(mask, target_i, -1)
        tp = torch.histc(matched.float(), bins=num_classes, min=0, max=num_classes - 1)
        fp = torch.histc(output_i.float(), bins=num_classes, min=0, max=num_classes - 1) - tp
        fn = torch.histc(target_i.float(), bins=num_classes, min=0, max=num_classes - 1) - tp
        tn = num_elements - tp - fp - fn
        if ignore_index is not None:
            tn = tn - ignore_per_sample[i]
        tp_count[i] = tp.long()
        fp_count[i] = fp.long()
        fn_count[i] = fn.long()
        tn_count[i] = tn.long()

    return tp_count, fp_count, fn_count, tn_count


@torch.no_grad()
def _get_stats_multilabel(
    output: torch.LongTensor,
    target: torch.LongTensor,
) -> Tuple[torch.LongTensor, torch.LongTensor, torch.LongTensor, torch.LongTensor]:

    batch_size, num_classes, *dims = target.shape
    output = output.view(batch_size, num_classes, -1)
    target = target.view(batch_size, num_classes, -1)

    tp = (output * target).sum(2)
    fp = output.sum(2) - tp
    fn = target.sum(2) - tp
    tn = torch.prod(torch.tensor(dims)) - (tp + fp + fn)

    return tp, fp, fn, tn


###################################################################################################
# Metrics computation
###################################################################################################


def _handle_zero_division(x, zero_division):
    nans = torch.isnan(x)
    if torch.any(nans) and zero_division == "warn":
        warnings.warn("Zero division in metric calculation!")
    value = zero_division if zero_division != "warn" else 0
    value = torch.tensor(value, dtype=x.dtype).to(x.device)
    x = torch.where(nans, value, x)
    return x


def _compute_metric(
    metric_fn,
    tp,
    fp,
    fn,
    tn,
    reduction: Optional[str] = None,
    class_weights: Optional[List[float]] = None,
    zero_division="warn",
    **metric_kwargs,
) -> float:

    if class_weights is None and reduction is not None and "weighted" in reduction:
        raise ValueError(f"Class weights should be provided for `{reduction}` reduction")

    class_weights = class_weights if class_weights is not None else 1.0
    class_weights = torch.tensor(class_weights).to(tp.device)
    class_weights = class_weights / class_weights.sum()

    if reduction == "micro":
        tp = tp.sum()
        fp = fp.sum()
        fn = fn.sum()
        tn = tn.sum()
        score = metric_fn(tp, fp, fn, tn, **metric_kwargs)

    elif reduction == "macro" or reduction == "weighted":
        tp = tp.sum(0)
        fp = fp.sum(0)
        fn = fn.sum(0)
        tn = tn.sum(0)
        score = metric_fn(tp, fp, fn, tn, **metric_kwargs)
        score = _handle_zero_division(score, zero_division)
        score = (score * class_weights).mean()

    elif reduction == "micro-imagewise":
        tp = tp.sum(1)
        fp = fp.sum(1)
        fn = fn.sum(1)
        tn = tn.sum(1)
        score = metric_fn(tp, fp, fn, tn, **metric_kwargs)
        score = _handle_zero_division(score, zero_division)
        score = score.mean()

    elif reduction == "macro-imagewise" or reduction == "weighted-imagewise":
        score = metric_fn(tp, fp, fn, tn, **metric_kwargs)
        score = _handle_zero_division(score, zero_division)
        score = (score.mean(0) * class_weights).mean()

    elif reduction == "none" or reduction is None:
        score = metric_fn(tp, fp, fn, tn, **metric_kwargs)
        score = _handle_zero_division(score, zero_division)

    else:
        raise ValueError(
            "`reduction` should be in [micro, macro, weighted, micro-imagewise,"
            + "macro-imagesize, weighted-imagewise, none, None]"
        )

    return score


# Logic for metric computation, all metrics are with the same interface


def _fbeta_score(tp, fp, fn, tn, beta=1):
    beta_tp = (1 + beta ** 2) * tp
    beta_fn = (beta ** 2) * fn
    score = beta_tp / (beta_tp + beta_fn + fp)
    return score


def _iou_score(tp, fp, fn, tn):
    return tp / (tp + fp + fn)


def _accuracy(tp, fp, fn, tn):
    return (tp + tn) / (tp + fp + fn + tn)


def _sensitivity(tp, fp, fn, tn):
    return tp / (tp + fn)


def _specificity(tp, fp, fn, tn):
    return tn / (tn + fp)


def _balanced_accuracy(tp, fp, fn, tn):
    return (_sensitivity(tp, fp, fn, tn) + _specificity(tp, fp, fn, tn)) / 2


def _positive_predictive_value(tp, fp, fn, tn):
    return tp / (tp + fp)


def _negative_predictive_value(tp, fp, fn, tn):
    return tn / (tn + fn)


def _false_negative_rate(tp, fp, fn, tn):
    return fn / (fn + tp)


def _false_positive_rate(tp, fp, fn, tn):
    return fp / (fp + tn)


def _false_discovery_rate(tp, fp, fn, tn):
    return 1 - _positive_predictive_value(tp, fp, fn, tn)


def _false_omission_rate(tp, fp, fn, tn):
    return 1 - _negative_predictive_value(tp, fp, fn, tn)


def _positive_likelihood_ratio(tp, fp, fn, tn):
    return _sensitivity(tp, fp, fn, tn) / _false_positive_rate(tp, fp, fn, tn)


def _negative_likelihood_ratio(tp, fp, fn, tn):
    return _false_negative_rate(tp, fp, fn, tn) / _specificity(tp, fp, fn, tn)


def fbeta_score(
    tp: torch.LongTensor,
    fp: torch.LongTensor,
    fn: torch.LongTensor,
    tn: torch.LongTensor,
    beta: float = 1.0,
    reduction: Optional[str] = None,
    class_weights: Optional[List[float]] = None,
    zero_division: Union[str, float] = 1.0,
) -> torch.Tensor:
    """F beta score"""
    return _compute_metric(
        _fbeta_score,
        tp,
        fp,
        fn,
        tn,
        beta=beta,
        reduction=reduction,
        class_weights=class_weights,
        zero_division=zero_division,
    )


def f1_score(
    tp: torch.LongTensor,
    fp: torch.LongTensor,
    fn: torch.LongTensor,
    tn: torch.LongTensor,
    reduction: Optional[str] = None,
    class_weights: Optional[List[float]] = None,
    zero_division: Union[str, float] = 1.0,
) -> torch.Tensor:
    """F1 score"""
    return _compute_metric(
        _fbeta_score,
        tp,
        fp,
        fn,
        tn,
        beta=1.0,
        reduction=reduction,
        class_weights=class_weights,
        zero_division=zero_division,
    )


def iou_score(
    tp: torch.LongTensor,
    fp: torch.LongTensor,
    fn: torch.LongTensor,
    tn: torch.LongTensor,
    reduction: Optional[str] = None,
    class_weights: Optional[List[float]] = None,
    zero_division: Union[str, float] = 1.0,
) -> torch.Tensor:
    """IoU score or Jaccard index"""  # noqa
    return _compute_metric(
        _iou_score,
        tp,
        fp,
        fn,
        tn,
        reduction=reduction,
        class_weights=class_weights,
        zero_division=zero_division,
    )


def accuracy(
    tp: torch.LongTensor,
    fp: torch.LongTensor,
    fn: torch.LongTensor,
    tn: torch.LongTensor,
    reduction: Optional[str] = None,
    class_weights: Optional[List[float]] = None,
    zero_division: Union[str, float] = 1.0,
) -> torch.Tensor:
    """Accuracy"""
    return _compute_metric(
        _accuracy,
        tp,
        fp,
        fn,
        tn,
        reduction=reduction,
        class_weights=class_weights,
        zero_division=zero_division,
    )


def sensitivity(
    tp: torch.LongTensor,
    fp: torch.LongTensor,
    fn: torch.LongTensor,
    tn: torch.LongTensor,
    reduction: Optional[str] = None,
    class_weights: Optional[List[float]] = None,
    zero_division: Union[str, float] = 1.0,
) -> torch.Tensor:
    """Sensitivity, recall, hit rate, or true positive rate (TPR)"""
    return _compute_metric(
        _sensitivity,
        tp,
        fp,
        fn,
        tn,
        reduction=reduction,
        class_weights=class_weights,
        zero_division=zero_division,
    )


def specificity(
    tp: torch.LongTensor,
    fp: torch.LongTensor,
    fn: torch.LongTensor,
    tn: torch.LongTensor,
    reduction: Optional[str] = None,
    class_weights: Optional[List[float]] = None,
    zero_division: Union[str, float] = 1.0,
) -> torch.Tensor:
    """Specificity, selectivity or true negative rate (TNR)"""
    return _compute_metric(
        _specificity,
        tp,
        fp,
        fn,
        tn,
        reduction=reduction,
        class_weights=class_weights,
        zero_division=zero_division,
    )


def balanced_accuracy(
    tp: torch.LongTensor,
    fp: torch.LongTensor,
    fn: torch.LongTensor,
    tn: torch.LongTensor,
    reduction: Optional[str] = None,
    class_weights: Optional[List[float]] = None,
    zero_division: Union[str, float] = 1.0,
) -> torch.Tensor:
    """Balanced accuracy"""
    return _compute_metric(
        _balanced_accuracy,
        tp,
        fp,
        fn,
        tn,
        reduction=reduction,
        class_weights=class_weights,
        zero_division=zero_division,
    )


def positive_predictive_value(
    tp: torch.LongTensor,
    fp: torch.LongTensor,
    fn: torch.LongTensor,
    tn: torch.LongTensor,
    reduction: Optional[str] = None,
    class_weights: Optional[List[float]] = None,
    zero_division: Union[str, float] = 1.0,
) -> torch.Tensor:
    return _compute_metric(
        _positive_predictive_value,
        tp,
        fp,
        fn,
        tn,
        reduction=reduction,
        class_weights=class_weights,
        zero_division=zero_division,
    )


def negative_predictive_value(
    tp: torch.LongTensor,
    fp: torch.LongTensor,
    fn: torch.LongTensor,
    tn: torch.LongTensor,
    reduction: Optional[str] = None,
    class_weights: Optional[List[float]] = None,
    zero_division: Union[str, float] = 1.0,
) -> torch.Tensor:
    """Negative predictive value (NPV)"""
    return _compute_metric(
        _negative_predictive_value,
        tp,
        fp,
        fn,
        tn,
        reduction=reduction,
        class_weights=class_weights,
        zero_division=zero_division,
    )


def false_negative_rate(
    tp: torch.LongTensor,
    fp: torch.LongTensor,
    fn: torch.LongTensor,
    tn: torch.LongTensor,
    reduction: Optional[str] = None,
    class_weights: Optional[List[float]] = None,
    zero_division: Union[str, float] = 1.0,
) -> torch.Tensor:
    """Miss rate or false negative rate (FNR)"""
    return _compute_metric(
        _false_negative_rate,
        tp,
        fp,
        fn,
        tn,
        reduction=reduction,
        class_weights=class_weights,
        zero_division=zero_division,
    )


def false_positive_rate(
    tp: torch.LongTensor,
    fp: torch.LongTensor,
    fn: torch.LongTensor,
    tn: torch.LongTensor,
    reduction: Optional[str] = None,
    class_weights: Optional[List[float]] = None,
    zero_division: Union[str, float] = 1.0,
) -> torch.Tensor:
    """Fall-out or false positive rate (FPR)"""
    return _compute_metric(
        _false_positive_rate,
        tp,
        fp,
        fn,
        tn,
        reduction=reduction,
        class_weights=class_weights,
        zero_division=zero_division,
    )


def false_discovery_rate(
    tp: torch.LongTensor,
    fp: torch.LongTensor,
    fn: torch.LongTensor,
    tn: torch.LongTensor,
    reduction: Optional[str] = None,
    class_weights: Optional[List[float]] = None,
    zero_division: Union[str, float] = 1.0,
) -> torch.Tensor:
    """False discovery rate (FDR)"""  # noqa
    return _compute_metric(
        _false_discovery_rate,
        tp,
        fp,
        fn,
        tn,
        reduction=reduction,
        class_weights=class_weights,
        zero_division=zero_division,
    )


def false_omission_rate(
    tp: torch.LongTensor,
    fp: torch.LongTensor,
    fn: torch.LongTensor,
    tn: torch.LongTensor,
    reduction: Optional[str] = None,
    class_weights: Optional[List[float]] = None,
    zero_division: Union[str, float] = 1.0,
) -> torch.Tensor:
    """False omission rate (FOR)"""  # noqa
    return _compute_metric(
        _false_omission_rate,
        tp,
        fp,
        fn,
        tn,
        reduction=reduction,
        class_weights=class_weights,
        zero_division=zero_division,
    )


def positive_likelihood_ratio(
    tp: torch.LongTensor,
    fp: torch.LongTensor,
    fn: torch.LongTensor,
    tn: torch.LongTensor,
    reduction: Optional[str] = None,
    class_weights: Optional[List[float]] = None,
    zero_division: Union[str, float] = 1.0,
) -> torch.Tensor:
    """Positive likelihood ratio (LR+)"""
    return _compute_metric(
        _positive_likelihood_ratio,
        tp,
        fp,
        fn,
        tn,
        reduction=reduction,
        class_weights=class_weights,
        zero_division=zero_division,
    )


def negative_likelihood_ratio(
    tp: torch.LongTensor,
    fp: torch.LongTensor,
    fn: torch.LongTensor,
    tn: torch.LongTensor,
    reduction: Optional[str] = None,
    class_weights: Optional[List[float]] = None,
    zero_division: Union[str, float] = 1.0,
) -> torch.Tensor:
    """Negative likelihood ratio (LR-)"""
    return _compute_metric(
        _negative_likelihood_ratio,
        tp,
        fp,
        fn,
        tn,
        reduction=reduction,
        class_weights=class_weights,
        zero_division=zero_division,
    )