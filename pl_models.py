import os

import cv2 as cv
import lightning as L
import numpy as np
import pandas as pd
import torch
from losses.loss import TorchLoss
from models.model import TorchModel
from torch import optim
from torchmetrics import MetricCollection
from torchmetrics.classification import BinaryAUROC, BinaryJaccardIndex
from transformers import get_cosine_schedule_with_warmup


class TrainPipeline(L.LightningModule):
    def __init__(self, config, train_loader, val_loader) -> None:
        super().__init__()
        self.config = config

        self.model = TorchModel(config['model'])
        if config['weights'] is not None:
            state_dict = torch.load(
                config['weights'],
                map_location='cpu'
            )['state_dict']
            self.load_state_dict(state_dict, strict=True)
        self.criterion = TorchLoss()
        segmentation_metrics = MetricCollection([BinaryJaccardIndex()])
        classification_metrics = MetricCollection([BinaryAUROC()])

        self.train_segmentation_metrics = segmentation_metrics.clone(
            postfix="/segmentation/train"
        )
        self.train_classification_metrics = classification_metrics.clone(
            postfix="/classification/train"
        )

        self.val_segmentation_metrics = segmentation_metrics.clone(
            postfix="/segmentation/val"
        )
        self.val_classification_metrics = classification_metrics.clone(
            postfix="/classification/val"
        )

        self.train_loader = train_loader
        self.val_loader = val_loader
        # In case of DDP
        # self.num_training_steps = math.ceil(len(self.train_loader) / len(config['trainer']['devices']))
        self.num_training_steps = len(self.train_loader)

        self.save_hyperparameters(config)

    def configure_optimizers(self):
        if self.config['optimizer'] == "adam":
            optimizer = optim.Adam(
                filter(lambda p: p.requires_grad, self.model.parameters()),
                **self.config['optimizer_params']
            )
        elif self.config['optimizer'] == "sgd":
            optimizer = optim.SGD(
                filter(lambda p: p.requires_grad, self.model.parameters()),
                momentum=0.9, nesterov=True,
                **self.config['optimizer_params']
            )
        else:
            raise ValueError(
                f"Unknown optimizer name: {self.config['optimizer']}")

        scheduler_params = self.config['scheduler_params']
        if self.hparams.scheduler == "plateau":
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer=optimizer,
                patience=scheduler_params['patience'],
                min_lr=scheduler_params['min_lr'],
                factor=scheduler_params['factor'],
                mode=scheduler_params['mode'],
                verbose=scheduler_params['verbose'],
            )

            lr_scheduler = {
                'scheduler': scheduler,
                'interval': 'epoch',
                'monitor': scheduler_params['target_metric']
            }
        elif self.config['scheduler'] == "cosine":
            scheduler = get_cosine_schedule_with_warmup(
                optimizer,
                num_warmup_steps=self.num_training_steps *
                scheduler_params['warmup_epochs'],
                num_training_steps=int(
                    self.num_training_steps * self.config['trainer']['max_epochs'])
            )

            lr_scheduler = {
                'scheduler': scheduler,
                'interval': 'step'
            }
        else:
            raise ValueError(
                f"Unknown scheduler name: {self.config['scheduler']}")

        return [optimizer], [lr_scheduler]

    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        return self.val_loader

    def training_step(self, batch, batch_idx):
        x, y_mask, y_label = batch
        y_mask_pred, y_label_pred = self.model(x)
        loss = self.criterion(
            preds_mask=y_mask_pred,
            preds_label=y_label_pred,
            target_mask=y_mask,
            target_label=y_label
        )

        self.log("Loss/train", loss, prog_bar=True)
        self.train_segmentation_metrics.update(y_mask_pred, y_mask.long())
        self.train_classification_metrics.update(y_label_pred, y_label)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y_mask, y_label = batch
        y_mask_pred, y_label_pred = self.model(x)
        loss = self.criterion(
            preds_mask=y_mask_pred,
            preds_label=y_label_pred,
            target_mask=y_mask,
            target_label=y_label
        )

        self.log("Loss/val", loss, prog_bar=True)
        self.val_segmentation_metrics.update(y_mask_pred, y_mask.long())
        self.val_classification_metrics.update(y_label_pred, y_label)

    def on_train_epoch_end(self):
        train_segmentation_metrics = self.train_segmentation_metrics.compute()
        self.log_dict(train_segmentation_metrics)
        self.train_segmentation_metrics.reset()

        train_classification_metrics = self.train_classification_metrics.compute()
        self.log_dict(train_classification_metrics)
        self.train_classification_metrics.reset()

    def on_validation_epoch_end(self):
        val_segmentation_metrics = self.val_segmentation_metrics.compute()
        self.log_dict(val_segmentation_metrics)
        self.val_segmentation_metrics.reset()

        val_classification_metrics = self.val_classification_metrics.compute()
        self.log_dict(val_classification_metrics)
        self.val_classification_metrics.reset()


class TestPipeline(L.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.model = TorchModel(config['model'])
        state_dict = torch.load(config['weights'], map_location='cpu')[
            'state_dict']
        self.load_state_dict(state_dict, strict=True)
        self.test_outputs = []
        self.save_path = os.path.join(
            self.config['save_path'],
            self.config['test_name']
        )
        os.makedirs(self.save_path, exist_ok=True)

    def sync_across_gpus(self, tensors):
        tensors = self.all_gather(tensors)
        return torch.cat([t for t in tensors])

    def test_step(self, batch):
        x, ids = batch
        pred_mask, pred_label = self.model(x)
        pred_mask = torch.sigmoid(pred_mask)
        pred_label = torch.sigmoid(pred_label)

        self._seve_preds_masks(ids, pred_mask)

        self.test_outputs.append({
            "pred_label": pred_label,
            "idx": ids
        })

    def _seve_preds_masks(self, ids, pred_mask_batch):
        pred_mask_batch *= 255
        pred_mask_batch = pred_mask_batch.squeeze(1)
        pred_mask_batch = pred_mask_batch.cpu().numpy()
        pred_mask_batch
        
        ids = ids.cpu().numpy()
        for idx, mask in zip(ids, pred_mask_batch):
            cv.imwrite(f'{self.save_path}/{idx}.png', mask.astype(np.uint8))

    def on_test_epoch_end(self):
        all_test_outputs = self.all_gather(self.test_outputs)
        if self.trainer.is_global_zero:
            pred_label = torch.cat(
                [o['pred_label'] for o in all_test_outputs], dim=0
            ).cpu().detach().tolist()
            
            idx = torch.cat(
                [o['idx'] for o in all_test_outputs],
                dim=0
            ).cpu().detach().tolist()
            
            df = pd.DataFrame(
                {'idx': idx, 'label': pred_label}
            ).drop_duplicates()
            
            file_path = os.path.join(
                self.config['save_path'], self.config['test_name'], "predictions.csv")
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            df.to_csv(str(file_path), index=False)
