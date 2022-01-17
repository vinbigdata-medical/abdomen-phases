from sklearn import preprocessing
from tqdm import tqdm
import time
import pandas as pd
import numpy as np
import torch
import torch.nn as nn

from sklearn.metrics import accuracy_score, recall_score
from sklearn.metrics import precision_score, f1_score
from sklearn.metrics import classification_report

from core.utils import AverageMeter
from itertools import cycle
from core.utils import save_checkpoint
from sklearn.metrics import roc_auc_score, roc_curve, auc
# from .grad_cam_log import record_output_gradcam

import matplotlib.pyplot as plt

target_names_dict = {"Non": 0, "Venous": 1, "Aterial": 2, "Others": 3}
map_id_name = {0: "Non Contrast", 1: "Venous", 2: "Aterial", 3: "Others"}

def valid_model(
    cfg,
    mode,
    epoch,
    model,
    dataloader,
    criterion,
    writer=None,
    save_prediction=True,
    best_metric=None,
    visual=False
):
    """Evaluate model performance on Validating dataset

    Args:
        cfg (CfgNode): Config object containing running configuration
        mode (str): Model running mode (valid/test)
        model (nn.Module): Model that need to have performance evaluated
        dataloader (data.DataLoader): Dataloader object to load data batch-wise
        criterion: Loss function
        writer (Summarywriter): Logger that log validation loss and plot it on Tensorboard
        save_prediction (Boolean): Whether to save prediction output or not (for bootstraping)
        best_metric (float, optional): Best performance result of loaded model. Defaults to None.
    """

    # Declare variables
    gpu = cfg.SYSTEM.GPU
    output_log_dir = cfg.DIRS.OUTPUTS
    model.eval()
    losses = AverageMeter()
    tbar = tqdm(dataloader)
    targets, preds, filenames, study_IDs, seriesNumbers = (
        list(),
        list(),
        list(),
        list(),
        list(),
    )
    data = dict()

    total_time = 0
    all_probs = []
    for i, (filename, study_ID, seriesNumber, image, target) in enumerate(tbar):
        with torch.no_grad():
            image = image.float()
            if gpu:
                image, target = image.cuda(), target.cuda()
                start = time.time()
                output = model(image)
                end = time.time()

            # Output prediction
            sigmoid = nn.Sigmoid()
            probs = sigmoid(output)
            pred = torch.argmax(probs, 1)
            probs = probs.cpu().numpy()
            all_probs.append(probs)
            # print(probs.shape)
            # print(pred.shape)
            # print("_--------------_")
            total_time += end - start
 
            # Compute loss
            loss = criterion(output, target)

            # Record loss
            losses.update(loss.item() * cfg.SOLVER.GD_STEPS, target.size(0))
            tbar.set_description("Valid loss: %.9f" % (losses.avg))

            # Convert target, prediction to numpy
            target = list(target.detach().cpu().numpy())
            pred = list(pred.detach().cpu().numpy())
            # print(pred)
            filename = list(filename)
            targets += target
            preds += pred
            filenames += filename
            study_IDs += study_ID
            seriesNumbers += list(np.array(seriesNumber))
    # print(f"Inference time =", (total_time / len(tbar)) / 100)
    all_targets = []
    for idx in range(len(targets)):
        cur = [0] * 4
        cur[targets[idx]] = 1
        all_targets.append([cur])
    all_probs = np.concatenate(all_probs, axis=0)
    all_target = np.concatenate(all_targets, axis=0)
    # print(all_target.shape)
    # print(all_probs.shape)
    np.save("target.npy", all_target)
    np.save("probs.npy", all_probs)
    # print(type(targets), len(targets))
    # print(all_probs.shape)
    if visual == True:
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        for i in range(4):
            fpr[i], tpr[i], _ = roc_curve(all_target[:, i], all_probs[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
        colors = cycle(["aqua", "darkorange", "cornflowerblue", "red"])
        lw = 2
        plt.figure()
        for i, color in zip(range(4), colors):
            plt.plot(
                fpr[i],
                tpr[i],
                color=color,
                lw=lw,
                label=f"ROC curve of class {map_id_name[i]} (area = {roc_auc[i]})"
            )
            
        plt.plot([0, 1], [0, 1], "k--", lw=lw)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("Some extension of Receiver operating characteristic to multiclass")
        plt.legend(loc="lower right")
        plt.show()
    # Record wrongly predicted sample and save confusion matrix
    # record_output(cfg, mode, output_log_dir, study_IDs, seriesNumbers,
    #                 targets, preds, filenames)
    # record_output_gradcam(cfg, mode, output_log_dir, targets, preds, filenames, model)

    # Calculate Metrics
    accuracy = accuracy_score(targets, preds)
    recall = recall_score(targets, preds, average="macro")
    precision = precision_score(targets, preds, average="macro")
    f1 = f1_score(targets, preds, average="macro")
    print(
        "ACCURACY: %.9f, RECALL: %.9f, PRECISION: %.9f, F1: %.9f"
        % (accuracy, recall, precision, f1)
    )

    if len(np.unique(preds)) == cfg.MODEL.NUM_CLASSES:
        report = classification_report(
            targets,
            preds,
            target_names=["Non", "Venous", "Aterial", "Others"],
            digits=4,
        )
        print(report)
    # else:
    #     from core.utils import print_report, classification_report_
    #     report = classification_report_(targets, preds, target_names_dict)
    #     print_report(report)
    data["Study_ID"] = study_IDs
    data["Filename"] = filenames
    data["SeriesNumber"] = seriesNumbers
    data["Prediction"] = preds
    data["Label"] = targets
    data = pd.DataFrame(data)
    all_series = []
    for (studyuid, seriesuid), tmp_df in data.groupby(['Study_ID', 'SeriesNumber']):
        preds = tmp_df['Prediction'].tolist()
        labels = tmp_df['Label'].tolist()
        f1_series = f1_score(labels, preds, average='macro')
        all_series.append(f1_series)
    all_series = np.array(all_series)
    f1_series = np.mean(all_series)
    print("series", f1_series)
    save_dict = {
            "epoch": epoch + 1,
            "arch": cfg.NAME,
            "state_dict": model.state_dict(),
            "best_metric": best_metric,
        }
    save_filename = f"{cfg.NAME}_{str(f1)}_{str(f1_series)}.pth"
    
    save_checkpoint(save_dict, root=cfg.DIRS.WEIGHTS, filename=save_filename)
        # print(studyuid, seriesuid, f1)
    if mode == "train":
        # writer.add_scalars(
        #     f"Metrics",
        #     {
        #         "F1_SCORE": f1,
        #         "ACCURACY": accuracy,
        #         "RECALL": recall,
        #         "PRECISION": precision,
        #     },
        #     epoch,
        # )

        # CHECKPOINT
        is_best = f1 > best_metric
        best_metric = max(f1, best_metric)

    
        # Save All slices prediction for scan prediction and bootstraping
        
    if save_prediction:
        data.to_csv(f"eval_{mode}.csv", index=False)

    


    return best_metric
