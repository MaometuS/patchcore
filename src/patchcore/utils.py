import csv
import logging
import os
import random
import math

import matplotlib.pyplot as plt
import numpy as np
import PIL
import torch
import tqdm
import warnings
from scipy.spatial import distance

LOGGER = logging.getLogger(__name__)


def plot_segmentation_images(
    savefolder,
    image_paths,
    segmentations,
    anomaly_scores=None,
    mask_paths=None,
    image_transform=lambda x: x,
    mask_transform=lambda x: x,
    save_depth=4,
):
    """Generate anomaly segmentation images.

    Args:
        image_paths: List[str] List of paths to images.
        segmentations: [List[np.ndarray]] Generated anomaly segmentations.
        anomaly_scores: [List[float]] Anomaly scores for each image.
        mask_paths: [List[str]] List of paths to ground truth masks.
        image_transform: [function or lambda] Optional transformation of images.
        mask_transform: [function or lambda] Optional transformation of masks.
        save_depth: [int] Number of path-strings to use for image savenames.
    """
    if mask_paths is None:
        mask_paths = ["-1" for _ in range(len(image_paths))]
    masks_provided = mask_paths[0] != "-1"
    if anomaly_scores is None:
        anomaly_scores = ["-1" for _ in range(len(image_paths))]

    os.makedirs(savefolder, exist_ok=True)

    for image_path, mask_path, anomaly_score, segmentation in tqdm.tqdm(
        zip(image_paths, mask_paths, anomaly_scores, segmentations),
        total=len(image_paths),
        desc="Generating Segmentation Images...",
        leave=False,
    ):
        image = PIL.Image.open(image_path).convert("RGB")
        image = image_transform(image)
        if not isinstance(image, np.ndarray):
            image = image.numpy()

        if masks_provided:
            if mask_path is not None:
                mask = PIL.Image.open(mask_path).convert("RGB")
                mask = mask_transform(mask)
                if not isinstance(mask, np.ndarray):
                    mask = mask.numpy()
            else:
                mask = np.zeros_like(image)

        savename = image_path.split("/")
        savename = "_".join(savename[-save_depth:])
        savename = os.path.join(savefolder, savename)
        f, axes = plt.subplots(1, 2 + int(masks_provided))
        axes[0].imshow(image.transpose(1, 2, 0))
        axes[1].imshow(mask.transpose(1, 2, 0))
        axes[2].imshow(segmentation)
        f.set_size_inches(3 * (2 + int(masks_provided)), 3)
        f.tight_layout()
        f.savefig(savename)
        plt.close()


def create_storage_folder(
    main_folder_path, project_folder, group_folder, mode="iterate"
):
    os.makedirs(main_folder_path, exist_ok=True)
    project_path = os.path.join(main_folder_path, project_folder)
    os.makedirs(project_path, exist_ok=True)
    save_path = os.path.join(project_path, group_folder)
    if mode == "iterate":
        counter = 0
        while os.path.exists(save_path):
            save_path = os.path.join(project_path, group_folder + "_" + str(counter))
            counter += 1
        os.makedirs(save_path)
    elif mode == "overwrite":
        os.makedirs(save_path, exist_ok=True)

    return save_path


def set_torch_device(gpu_ids):
    """Returns correct torch.device.

    Args:
        gpu_ids: [list] list of gpu ids. If empty, cpu is used.
    """
    if len(gpu_ids):
        # os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        # os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_ids[0])
        return torch.device("cuda:{}".format(gpu_ids[0]))
    return torch.device("cpu")


def fix_seeds(seed, with_torch=True, with_cuda=True):
    """Fixed available seeds for reproducibility.

    Args:
        seed: [int] Seed value.
        with_torch: Flag. If true, torch-related seeds are fixed.
        with_cuda: Flag. If true, torch+cuda-related seeds are fixed
    """
    random.seed(seed)
    np.random.seed(seed)
    if with_torch:
        torch.manual_seed(seed)
    if with_cuda:
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True


def compute_and_store_final_results(
    results_path,
    results,
    row_names=None,
    column_names=[
        "Instance AUROC",
        "Full Pixel AUROC",
        "Full PRO",
        "Anomaly Pixel AUROC",
        "Anomaly PRO",
    ],
):
    """Store computed results as CSV file.

    Args:
        results_path: [str] Where to store result csv.
        results: [List[List]] List of lists containing results per dataset,
                 with results[i][0] == 'dataset_name' and results[i][1:6] =
                 [instance_auroc, full_pixelwisew_auroc, full_pro,
                 anomaly-only_pw_auroc, anomaly-only_pro]
    """
    if row_names is not None:
        assert len(row_names) == len(results), "#Rownames != #Result-rows."

    mean_metrics = {}
    for i, result_key in enumerate(column_names):
        mean_metrics[result_key] = np.mean([x[i] for x in results])
        LOGGER.info("{0}: {1:3.3f}".format(result_key, mean_metrics[result_key]))

    savename = os.path.join(results_path, "results.csv")
    with open(savename, "w") as csv_file:
        csv_writer = csv.writer(csv_file, delimiter=",")
        header = column_names
        if row_names is not None:
            header = ["Row Names"] + header

        csv_writer.writerow(header)
        for i, result_list in enumerate(results):
            csv_row = result_list
            if row_names is not None:
                csv_row = [row_names[i]] + result_list
            csv_writer.writerow(csv_row)
        mean_scores = list(mean_metrics.values())
        if row_names is not None:
            mean_scores = ["Mean"] + mean_scores
        csv_writer.writerow(mean_scores)

    mean_metrics = {"mean_{0}".format(key): item for key, item in mean_metrics.items()}
    return mean_metrics


def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    # Cut & paste from PyTorch official master until it's in a few official releases - RW
    # Method based on https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    def norm_cdf(x):
        # Computes standard normal cumulative distribution function
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
                      "The distribution of values may be incorrect.",
                      stacklevel=2)

    with torch.no_grad():
        # Values are generated by using a truncated uniform distribution and
        # then using the inverse CDF for the normal distribution.
        # Get upper and lower cdf values
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)

        # Uniformly fill tensor with values from [l, u], then translate to
        # [2l-1, 2u-1].
        tensor.uniform_(2 * l - 1, 2 * u - 1)

        # Use inverse cdf transform for normal distribution to get truncated
        # standard normal
        tensor.erfinv_()

        # Transform to proper mean, std
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)

        # Clamp to ensure it's in the proper range
        tensor.clamp_(min=a, max=b)
        return tensor


def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    # type: (Tensor, float, float, float, float) -> Tensor
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)


def Weight_Distance_Unsupervised(Z, i, device):
    matrix_min = torch.tensor([]).to(device)
    for j in range(Z.shape[0]):
        if j != i:
            matrix_min = torch.cat((matrix_min, torch.min(torch.cdist(Z[i], Z[j]), 1)[0].unsqueeze(1)), dim=1)
    return torch.mean(matrix_min, dim=1)


def Weight_Distance_Supervised(Z, Z_train, i, device):
    matrix_min = torch.tensor([]).to(device)
    for j in range(Z_train.shape[0]):
        dist_matrix = torch.cdist(Z[i], Z_train[j])
        dist_matrix = torch.min(dist_matrix, dim=1)[0].unsqueeze(1)
        matrix_min = torch.cat((matrix_min, dist_matrix), dim=1)
    matrix_min = torch.min(matrix_min, dim=1)[0]
    return matrix_min


def Matrix_Alpha_Unsupervised(tau, k, Z, device):
    print("{:-^80}".format("Calculating Unsupervised Alpha Matrix"))

    matrix_alpha = torch.tensor([]).double().to(device)
    with tqdm.tqdm(total=int(Z.shape[0])) as progress:
        for i in range(Z.shape[0]):
            weight_distance_matrix = Weight_Distance_Unsupervised(Z, i, device).unsqueeze(0)
            weight_distance_matrix = weight_distance_matrix.double()
            if math.isclose(tau, 0):
                alpha_i = (weight_distance_matrix == weight_distance_matrix.max(axis=1, keepdims=True)[0])\
                    .to(dtype=torch.float64)
            else:
                tau_1 = 1 / tau
                alpha_i = k * torch.exp(tau_1 * weight_distance_matrix)
            alpha_i = alpha_i / alpha_i.sum()
            matrix_alpha = torch.cat((matrix_alpha, alpha_i), dim=0)
            progress.update(1)
    return matrix_alpha


def Matrix_Alpha_Supervised(tau, k, Z, Z_train, device):
    print("{:-^80}".format("Calculating Supervised Alpha Matrix"))

    matrix_alpha = torch.tensor([]).double().to(device)
    with tqdm.tqdm(total=int(Z.shape[0])) as progress:
        for i in range(Z.shape[0]):
            weight_distance_matrix = Weight_Distance_Supervised(Z, Z_train, i, device).unsqueeze(0)
            weight_distance_matrix = weight_distance_matrix.double()
            if math.isclose(tau, 0):
                alpha_i = (weight_distance_matrix == weight_distance_matrix.max(axis=1, keepdims=True)[0])\
                            .to(dtype=torch.float64)
            else:
                tau_1 = 1 / tau
                alpha_i = k * torch.exp(tau_1 * weight_distance_matrix)
            alpha_i = alpha_i / alpha_i.sum()
            matrix_alpha = torch.cat((matrix_alpha, alpha_i), dim=0)
            progress.update(1)
    return matrix_alpha


def Max_Hausdorff_Distance(i, j):
    return max(distance.directed_hausdorff(i, j), distance.directed_hausdorff(i, j))[0]