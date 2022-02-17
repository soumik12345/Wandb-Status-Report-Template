import wandb
from fastprogress import progress_bar
from fastai.vision.all import *


__all__ = ["faccuracy", "dice_per_class", "create_dice_table"]

def faccuracy(inp, targ):
    "Foreground accuracy per item"
    return foreground_acc(inp.unsqueeze(0), targ.unsqueeze(0)).item()


def dice_per_class(inp, targ, class_labels):
    "Compute dice per class"
    dice_scores = {}
    eps = 1e-6
    for c in range(inp.shape[0]):
        dec_preds = inp.argmax(dim=0)
        p = torch.where(dec_preds == c, 1, 0)
        t = torch.where(targ == c, 1, 0)
        c_inter = (p*t).float().sum().item()
        c_union = (p+t).float().sum().item()
        if c_union:
            dice_scores[c] = 2.*c_inter/(c_union + eps)
        else:
            dice_scores[c] = -1
    return {class_labels[c]:dice_scores[c] for c in range(inp.shape[0])}


def create_dice_table(samples, outputs, predictions, class_labels):
    "Creates a wandb table with predictions and targets side by side"
    def _to_str(l):
        return [str(x) for x in l]
    
    
    
    pb = progress_bar(list(zip(samples, outputs, predictions)))
    
    data = []
    
    for (image, label), pred_label, prediction, in pb:
        pb.comment = "Creating a wandb.Table with preds"
        dice_scores = list(dice_per_class(prediction, label, class_labels).values())
        facc = faccuracy(prediction, label)
        image = image.permute(1, 2, 0)
        data.append([
            wandb.Image(image),
            wandb.Image(
                image,
                masks={
                    "predictions":  {
                        'mask_data':  pred_label[0].numpy(),
                        'class_labels':class_labels
                    }
                }
            ),
            wandb.Image(
                image,
                masks={
                    "ground_truths": {
                        'mask_data': label.numpy(),
                        'class_labels':class_labels
                    }
                }
            ),
            *dice_scores,
            facc,
        ])
    table = wandb.Table(data=data, 
                        columns=["Image", "Predicted_Mask", "Ground_Truth"]+ _to_str(class_labels.values()) + ["Foreground Acc"])
    return table