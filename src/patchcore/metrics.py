"""Anomaly metrics."""
import numpy as np
from sklearn import metrics


def compute_imagewise_retrieval_metrics(
    anomaly_prediction_weights, anomaly_ground_truth_labels
):
    """
    Computes retrieval statistics (AUROC, FPR, TPR).

    Args:
        anomaly_prediction_weights: [np.array or list] [N] Assignment weights
                                    per image. Higher indicates higher
                                    probability of being an anomaly.
        anomaly_ground_truth_labels: [np.array or list] [N] Binary labels - 1
                                    if image is an anomaly, 0 if not.
    """
    fpr, tpr, thresholds = metrics.roc_curve(
        anomaly_ground_truth_labels, anomaly_prediction_weights
    )

    auroc = metrics.roc_auc_score(
        anomaly_ground_truth_labels, anomaly_prediction_weights
    )

    precision, recall, pr_thresholds = metrics.precision_recall_curve(
        anomaly_ground_truth_labels, anomaly_prediction_weights
    )
    F1_scores = np.divide(
        2 * precision * recall,
        precision + recall,
        out=np.zeros_like(precision),
        where=(precision + recall) != 0,
    )
    optimal_threshold = pr_thresholds[np.argmax(F1_scores)]
    predictions = (anomaly_prediction_weights >= optimal_threshold).astype(int)

    prauc = metrics.auc(recall, precision)

    mcc = metrics.matthews_corrcoef(
        anomaly_ground_truth_labels, predictions
    )
    mcc = (mcc+1)/2

    return {"auroc": auroc, "prauc": prauc, "mcc": mcc, "fpr": fpr, "tpr": tpr, "threshold": thresholds}


def compute_pixelwise_retrieval_metrics(anomaly_segmentations, ground_truth_masks):
    """
    Computes pixel-wise statistics (AUROC, FPR, TPR) for anomaly segmentations
    and ground truth segmentation masks.

    Args:
        anomaly_segmentations: [list of np.arrays or np.array] [NxHxW] Contains
                                generated segmentation masks.
        ground_truth_masks: [list of np.arrays or np.array] [NxHxW] Contains
                            predefined ground truth segmentation masks
    """
    if isinstance(anomaly_segmentations, list):
        anomaly_segmentations = np.stack(anomaly_segmentations)
    if isinstance(ground_truth_masks, list):
        ground_truth_masks = np.stack(ground_truth_masks)

    flat_anomaly_segmentations = anomaly_segmentations.ravel()
    flat_ground_truth_masks = ground_truth_masks.ravel()

    fpr, tpr, thresholds = metrics.roc_curve(
        flat_ground_truth_masks.astype(int), flat_anomaly_segmentations
    )
    auroc = metrics.roc_auc_score(
        flat_ground_truth_masks.astype(int), flat_anomaly_segmentations
    )

    precision, recall, thresholds = metrics.precision_recall_curve(
        flat_ground_truth_masks.astype(int), flat_anomaly_segmentations
    )
    F1_scores = np.divide(
        2 * precision * recall,
        precision + recall,
        out=np.zeros_like(precision),
        where=(precision + recall) != 0,
    )

    prauc = metrics.auc(recall, precision)


    optimal_threshold = thresholds[np.argmax(F1_scores)]
    predictions = (flat_anomaly_segmentations >= optimal_threshold).astype(int)
    fpr_optim = np.mean(predictions > flat_ground_truth_masks)
    fnr_optim = np.mean(predictions < flat_ground_truth_masks)

    mcc = metrics.matthews_corrcoef(
        flat_ground_truth_masks.astype(int), predictions
    )
    mcc = (mcc+1)/2

    unique_regions = np.unique(ground_truth_masks)
    pro_scores = []
    for region in unique_regions:
        pred_region = predictions[flat_ground_truth_masks == region]
        gt_region = flat_ground_truth_masks[flat_ground_truth_masks == region].astype(int)
        pro_score = metrics.jaccard_score(gt_region, pred_region)
        pro_scores.append(pro_score)
    pro_average = np.mean(pro_scores)

    pro_score = 0
    for i in range(anomaly_segmentations.shape[0]):
        anomaly_mask = (anomaly_segmentations[i] >= optimal_threshold).astype(int)
        ground_truth_mask = ground_truth_masks[i].astype(int)
        pro_score += metrics.jaccard_score(anomaly_mask.ravel(), ground_truth_mask.ravel(), zero_division=1.0)
    pro_score /= anomaly_segmentations.shape[0]

    return {
        "auroc": auroc,
        "prauc": prauc,
        "mcc": mcc,
        "fpr": fpr,
        "tpr": tpr,
        "optimal_threshold": optimal_threshold,
        "optimal_fpr": fpr_optim,
        "optimal_fnr": fnr_optim,
        "pro": pro_score,
    }
