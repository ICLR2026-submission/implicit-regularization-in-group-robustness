import torch
import copy
import matplotlib.pyplot as plt
import pandas as pd
import json
import os
import torch.nn.functional as F
import random

from torch import optim
from collections import defaultdict
from torch.optim import lr_scheduler
from tqdm import tqdm

from .utils import *
from torch.nn import functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline


class Trainer:
    def __init__(
        self,
        model,
        criterion,
        device,
        optimizer=None,
        scheduler=None,
        model_path=None,
        log=None,
        clip_max_norm=0,
        reg_type=None,
        reg_strength=0,
        grad_tracker=None,
        accumulation_steps=None,
        force_calculate_reg=None,
    ):
        self.model = model
        self.device = device

        # optimization
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler # use None to not use scheduler
        self.clip_max_norm = clip_max_norm  # TODO: do we need to use?
        self.reg_type = reg_type
        self.reg_strength = reg_strength

        # history
        self.num_trained_epochs = 0
        self.logger = log
        self.history = []

        self.best_metrics_acc = None
        self.best_metrics_wga = None
        if model_path is None:
            model_path = f"./{self.__class__.__name__}"
        self.best_model_acc_path = f"{model_path}_best_model_acc.pt"
        self.best_model_wga_path = f"{model_path}_best_model_wga.pt"
        self.final_model_path = f"{model_path}_final_model.pt"

        self.dfr_by_model_selection = {}
        self.grad_tracker = grad_tracker

        self.accumulation_steps = accumulation_steps
        
        self.force_calculate_reg = force_calculate_reg

    def log(self, metrics):
        # does not log if logger is None
        if self.logger:
            self.logger(metrics)
        else:
            # for notebook
            self.history.append(metrics)
            print(metrics)

    def check_for_improvement(self, metrics):
        if (
            self.best_metrics_wga is None
            or metrics["val/worst_group_acc"]
            > self.best_metrics_wga["val/worst_group_acc"]
        ):
            torch.save(self.model.state_dict(), self.best_model_wga_path)
            self.model.to(self.device)
            self.best_metrics_wga = metrics
        if (
            self.best_metrics_acc is None
            or metrics["val/accuracy"] > self.best_metrics_acc["val/accuracy"]
        ):
            torch.save(self.model.state_dict(), self.best_model_acc_path)
            self.model.to(self.device)
            self.best_metrics_acc = metrics

    def get_best_model(self, by="wga"):
        model = copy.deepcopy(self.model)
        if by == "wga":
            model.load_state_dict(torch.load(self.best_model_wga_path))
        elif by == "acc":
            model.load_state_dict(torch.load(self.best_model_acc_path))
        elif by == "last_epoch":
            model.load_state_dict(torch.load(self.final_model_path))
        else:
            raise ValueError(f"Unknown model selection: {by}")

        model.to(self.device)
        return model

    def reset(self):
        self.train_history = defaultdict(list)
        self.test_history = defaultdict(list)
        self.num_trained_epochs = 0
        self.best_loss = float("inf")
        self.best_model = None
        self.scheduler.last_epoch = -1

    def _get_lr(self):
        return self.optimizer.param_groups[0]["lr"]

    def _scheduler_step(self, loss=None):
        # does nothing if self.scheduler is None
        if self.scheduler is None:
            return
        if isinstance(self.scheduler, lr_scheduler.ReduceLROnPlateau):
            self.scheduler.step(loss)
        else:
            self.scheduler.step()

    def caclulate_metrics(
        self, logits, features, labels, bias_labels, full_metrics=True
    ):
        metrics = {}
        pred_label = logits.argmax(dim=1)
        metrics["accuracy"] = (pred_label == labels).float().mean().item()
        wga, group_accs = worst_group_accuracy(pred_label, labels, bias_labels)
        metrics["worst_group_acc"] = wga
        for group in group_accs.keys():
            metrics[f"g{group}_acc"] = group_accs[group]
        metrics["mean_group_acc"] = sum(group_accs.values()) / len(group_accs)
        
        if full_metrics:
            # TODO: add things you want to monitor here
            pass
        return metrics

    def prediction_and_loss(self, images, labels, train=False):
        """returns logits, features, loss"""

        def batch_grad_norm(loss, only_last_layer=False):
            params = self.model.fc.parameters() if only_last_layer else self.model.parameters()
            grads = torch.autograd.grad(loss, params, create_graph=True)
            return sum((g.pow(2).sum()) for g in grads) / 4


        logits = self.model(images)
        loss = self.criterion(logits, labels)
        reg = 0

        if (train and self.reg_type == "batch_grad_norm") or self.force_calculate_reg:
            reg = batch_grad_norm(loss)
        elif train and self.reg_type == "batch_grad_norm_last_layer":
            reg = batch_grad_norm(loss, only_last_layer=True)

        loss = loss + self.reg_strength * reg

        return logits, None, loss, reg

    def _optimization_step(self):
        if self.clip_max_norm > 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_max_norm)
        self.optimizer.step()
        self.optimizer.zero_grad()
        
    def _track_gradients(self, loader):
        if self.grad_tracker is None:
            return {}

        results = self.grad_tracker.calculate_metrics(
            self.model, loader
        )
        self.optimizer.zero_grad()
        return results

    def run_epoch(self, loader, dataset="train", full_metrics=True, train=None):
        """Run one epoch of training or evaluation"""
        train = train if train is not None else dataset == "train"
        
        # only on train
        if self.grad_tracker and train:
            self.log(self._track_gradients(loader))

        self.model.train() if train else self.model.eval()
        total_loss, total = 0, 0
        epoch_reg_values = []

        all_features, all_labels, all_bias_labels, all_logits = [], [], [], []

        dataset_size = len(loader.dataset)
        mini_batch_size = loader.batch_size if loader.batch_size is not None else 1
        desired_batch_size = mini_batch_size * (self.accumulation_steps if self.accumulation_steps is not None else 1)
        last_desired_batch_size = dataset_size % desired_batch_size

        for i, batch in enumerate(tqdm(loader, leave=False)):
            if len(batch) == 4:
                images, labels, groups, bias_labels = batch
            elif len(batch) == 3:
                images, labels, bias_labels = batch
            else:
                raise ValueError(f"Batch length: {len(batch)} not supported")

            images, labels, bias_labels = images.to(self.device), labels.to(self.device), bias_labels.to(self.device)
            current_batch_size = images.size(0)

            with torch.set_grad_enabled(train or self.force_calculate_reg):
                logits, features, loss, reg = self.prediction_and_loss(images, labels, train)
            
            # accumulation conditions:
            if train:
                # normal case
                if self.accumulation_steps is None or self.accumulation_steps <= 1:
                    scale = 1.0
                # if we are in the last 'large' batch
                elif i + self.accumulation_steps >= len(loader) and last_desired_batch_size > 0:
                    scale = current_batch_size / last_desired_batch_size
                # normal accumulation step
                else:
                    scale = current_batch_size / desired_batch_size  # equal to 1/self.accumulation_steps
                scaled_loss = loss * scale
                scaled_loss.backward()
            # end of accumulation steps or end of loader
            if train and (self.accumulation_steps is None or (i+1)%self.accumulation_steps==0 or (i+1)==len(loader)):
                self._optimization_step()

            if features is not None:
                all_features.append(features.detach())
            all_logits.append(logits.detach())
            all_labels.append(labels.detach())
            all_bias_labels.append(bias_labels.detach())

            total_loss += loss.item() * images.size(0)
            total += images.size(0)
            if reg > 0:
                epoch_reg_values.append(reg.item())

                
        if self.force_calculate_reg:
            torch.cuda.empty_cache()

        epoch_loss = total_loss / total
        if train:
            self._scheduler_step(epoch_loss)
        all_features = torch.cat(all_features) if all_features else None
        all_labels = torch.cat(all_labels)
        all_bias_labels = torch.cat(all_bias_labels) if all_bias_labels else None
        all_logits = torch.cat(all_logits)

        metrics = {}
        metrics["loss"] = epoch_loss
        if epoch_reg_values:
            metrics["reg"] = np.mean(epoch_reg_values)
        metrics.update(
            self.caclulate_metrics(
                all_logits,
                all_features,
                all_labels,
                all_bias_labels,
                full_metrics=full_metrics,
            )
        )

        return {f"{dataset}/{k}":v for k, v in metrics.items()}

    def train(
        self,
        train_loader,
        val_loader,
        num_epochs=10,
        full_metrics=True,
        model_selection='wga',
        validate_each_epoch=True,
    ):  
        for e in range(self.num_trained_epochs, self.num_trained_epochs + num_epochs):
            metrics = self.run_epoch(train_loader, dataset="train", full_metrics=full_metrics)
            print(f"\rTrain: {self._get_metrics_str(metrics)}, lr: {self._get_lr():.3e}")
            self.log(metrics)

            if validate_each_epoch and val_loader is not None:
                metrics = self.run_epoch(val_loader, dataset="val", full_metrics=full_metrics)
                self.log(metrics)
                print(f"\rValidation: {self._get_metrics_str(metrics)}")
                self.check_for_improvement(metrics)
            self.num_trained_epochs += 1

        torch.save(self.model.state_dict(), self.final_model_path)
        if validate_each_epoch:
            metrics = self.run_epoch(val_loader, dataset="final_model/val", full_metrics=True)

        self.log(metrics)

    def _get_metrics_str(self, metrics):
        result = f"Epoch: {self.num_trained_epochs}"
        for key, value in metrics.items():
            if key.lower() == "loss":
                result += f", {key}: {value:0.2e}"
            else:
                result += f", {key}: {value:0.2f}"
        return result

    def test(self, loader, dataset, balanced_loader=None, model_selection="wga", run_dfr=True, run_sharp_flat=True, run_epoch=True, **kwargs):
        if model_selection is not None:
            name = f"best_model_by_{model_selection}/{dataset}"
            self.model = self.get_best_model(by=model_selection)
        self.model.eval()
        metrics = {}
        if run_epoch:
            metrics.update(self.run_epoch(
                loader,
                dataset=name,
                train=False,
                full_metrics=True,
            ))
        if run_sharp_flat:
            metrics[f'{name}/sharpness'] = calculate_sharpness(
                self.model, self.criterion, loader, self.device, **kwargs)
            metrics[f'{name}/flatness'] = calculate_flatness(
                self.model, self.criterion, loader, self.device, **kwargs)
        
        if run_dfr and balanced_loader is not None:
            # TODO: need to set validation_ratio for each dataset differently?
            if model_selection not in self.dfr_by_model_selection:
                dfr_trainer = DFR(self.model, validation_ratio=0.5)
                dfr_trainer.train(balanced_loader, device=self.device)
                self.dfr_by_model_selection[model_selection] = dfr_trainer
            
            dfr_trainer = self.dfr_by_model_selection[model_selection]
            # dfr_results = dfr_trainer.test(loader, device=self.device)
            
            fc_copy = copy.deepcopy(self.model.fc)
            self.model = dfr_trainer.set_dfr_weights()
            dfr_results = self.run_epoch(
                loader,
                dataset=name,
                train=False,
                full_metrics=True,
            )
            metrics.update({f"{name}/dfr_{k}":v for k, v in dfr_results.items()})
            # restore fc:
            self.model.fc = fc_copy
    
        self.log(metrics)
        return metrics


def caclulate_dfr_metrics(logits, labels, bias_labels):
    # inputs are numpy arrayss
    metrics = {}
    is_binary = (logits.ndim == 1) or (logits.ndim == 2 and logits.shape[1] == 1)
    if is_binary:
        logits = logits.flatten()
    logits_t = torch.from_numpy(logits).float()
    labels_t = torch.from_numpy(labels).long()
    if is_binary:
        predictions = (logits > 0).astype(int)
        metrics["loss"] = F.binary_cross_entropy_with_logits(logits_t, labels_t.float()).item()
    else:
        predictions = logits.argmax(1)
        metrics["loss"] = F.cross_entropy(logits_t, labels_t).item()
    metrics["accuracy"] = (predictions == labels).mean().item()
    metrics["worst_group_acc"] = worst_group_accuracy(predictions, labels, bias_labels)[0]
    return metrics


class DFR:
    def __init__(
        self, model, validation_ratio=0.5, penalty_costs=[0.01, 0.1, 1, 10, 100]
    ):
        self.model = model
        self.validation_ratio = validation_ratio
        self.penalty_costs = sorted(penalty_costs, reverse=True)
        self.best_pipeline = None

    def train(self, balanced_dataloader, device="cuda"):
        with torch.no_grad():
            features, y, bias, _ = get_features(
                balanced_dataloader, self.model, to_numpy=True, device=device
            )
        val_split = int(self.validation_ratio * len(features))
        indices = np.random.permutation(len(features))
        val_indices = indices[:val_split]
        train_indices = indices[val_split:]

        x_val, y_val, bias_val = (
            features[val_indices],
            y[val_indices],
            bias[val_indices],
        )
        
        x_train, y_train = (
            features[train_indices],
            y[train_indices],
        )
        
        del features, y, bias
        
        self.scaler = StandardScaler()
        x_train = self.scaler.fit_transform(x_train)
        x_val = self.scaler.transform(x_val)

        # DFR
        best_metric = -1
        best_c = 0
        best_pipeline = None
        pipeline = LogisticRegression(penalty="l1", solver="saga", max_iter=500, warm_start=True)
        for c in self.penalty_costs:
            pipeline.set_params(
                C=c,
                # max_iter=800 if c >= 1 else 300,
                # max_iter=1e-3 if c >= 1 else 1e-4,
            )

            pipeline.fit(x_train, y_train)
            logits_val = pipeline.decision_function(x_val)
            val_metric = caclulate_dfr_metrics(logits_val, y_val, bias_val)
            print(f"c: {c} --> {val_metric}")
            if val_metric["worst_group_acc"] > best_metric:
                best_metric = val_metric["worst_group_acc"]
                best_c = c
                best_pipeline = copy.deepcopy(pipeline)
        print(f"best c: {best_c}")
        self.best_pipeline = best_pipeline
        
    def set_dfr_weights(self, layer_name="fc", device="cuda"):
        """
        Set DFR weights to a specific layer by name
        """
        if self.best_pipeline is None:
            raise ValueError("DFR must be trained first. Call train() method.")

        # Extract weights from trained logistic regression
        lr_weights = torch.from_numpy(self.best_pipeline.coef_).float().to(device)
        lr_bias = torch.from_numpy(self.best_pipeline.intercept_).float().to(device)

        # Get the target layer
        target_layer = getattr(self.model, layer_name)

        # Update weights
        with torch.no_grad():
            target_layer.weight.data[0] = torch.zeros_like(lr_weights)
            target_layer.weight.data[1] = lr_weights
            
            if target_layer.bias is not None:
                target_layer.bias.data[0] = 0.0
                target_layer.bias.data[1] = lr_bias

        return self.model


    def test(self, test_loader, device="cuda"):
        features, y, bias, _ = get_features(
            test_loader, self.model, to_numpy=True, device=device
        )
        pred = self.best_pipeline.decision_function(self.scaler.transform(features))
        return caclulate_dfr_metrics(pred, y, bias)
