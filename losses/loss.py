from torch import nn


class TorchLoss(nn.Module):
    def __init__(self, cls_coef = 0.1):
        super().__init__()
        self.segmentation_loss = nn.BCEWithLogitsLoss()
        self.cls_loss = nn.BCEWithLogitsLoss()
        self.cls_coef = cls_coef

    def forward(
        self, 
        preds_mask, preds_label,
        target_mask, target_label
    ):
        segmentation_loss = self.segmentation_loss(preds_mask, target_mask)
        cls_loss = self.cls_loss(preds_label, target_label)
        return segmentation_loss + self.cls_coef * cls_loss