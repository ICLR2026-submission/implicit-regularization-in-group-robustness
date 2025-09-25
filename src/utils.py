import torch
from torch import nn
import torch.nn.functional as F

import json
import re
import os
import numpy as np
import pandas as pd
import yaml
import itertools
from collections import defaultdict
from torch.utils.data import Subset, DataLoader


class GradientTracker:
    def __init__(self, groups, criterion, device, spurious_corr):
        self.criterion = criterion
        self.device    = device
        self.groups = [int(g) for g in groups]
        self.spurious_corr = spurious_corr

    def calculate_metrics(self, model, loader):
        model.eval()
        params = [p for p in model.parameters() if p.requires_grad]

        group_vecs = defaultdict(lambda: 0)
        nums = defaultdict(lambda: 0)
        group_loss = defaultdict(lambda: 0)
        BATCH_SIZE = None
        grad_norms = []
        total_loss = 0
        for images, labels, groups, _ in loader:
            if BATCH_SIZE is None:
                BATCH_SIZE = len(labels)

            images = images.to(self.device)
            labels = labels.to(self.device)
            groups = groups.to(self.device)

            preds = model(images)

            loss = self.criterion(preds, labels)
            total_loss += loss.item() * BATCH_SIZE
            grads = torch.autograd.grad(loss, params, retain_graph=True)
            with torch.no_grad():
                vec = torch.cat([g.view(-1) for g in grads])
                norm = torch.norm(vec, p=2)
                grad_norms.append(norm)
            # build & compute each group’s gradient vector

            for grp in self.groups:
                mask = (grp == groups)
                if not mask.any():
                    continue
                loss_g  = self.criterion(preds[mask], labels[mask])
                grads_g = torch.autograd.grad(loss_g, params, retain_graph=True)
                with torch.no_grad():
                    vec_g = torch.cat([g.view(-1) for g in grads_g])
                    n_grp = mask.sum().item()
                    nums[grp] += n_grp
                    group_vecs[grp] += vec_g * n_grp
                    group_loss[grp] += loss_g.item() * n_grp

            
        results = {}
        g_total = torch.stack(list(group_vecs.values())).sum(dim=0) / sum(nums.values())
        group_vecs = {g: v/nums[g] for g, v in group_vecs.items()}
        group_loss = {g: v/nums[g] for g, v in group_loss.items()}
        l_maj = (group_loss[1] + group_loss[2]) / 2
        l_min = (group_loss[-1] + group_loss[-2]) / 2

        all_norms = torch.stack(grad_norms)        # tensor of shape [num_batches]

        results['E0/mean_batch_norm'] = all_norms.mean()
        results['E0/reg'] = all_norms.mean()*1e-3/4
        # results['loss'] = total_loss/len(loader.dataset) 
        # results['C(w)'] = results['loss'] - results['reg']
        # results['reg/C(w)'] = results['reg']/results['C(w)']
        # results['reg/loss'] = results['reg']/results['loss']

        results['norm_g_total'] = torch.linalg.norm(g_total)
        results['norm_mu1'] = torch.linalg.norm(group_vecs[1])
        results['norm_mu2'] = torch.linalg.norm(group_vecs[2])
        results['norm_mu-1'] = torch.linalg.norm(group_vecs[-1])
        results['norm_mu-2'] = torch.linalg.norm(group_vecs[-2])
        results['mu1mu2'] = torch.dot(group_vecs[1], group_vecs[2])
        results['mu-1mu-2'] = torch.dot(group_vecs[-1], group_vecs[-2])
        results['(m1+m2)(m-1+m-2)'] = torch.dot(group_vecs[1] + group_vecs[2], group_vecs[-1] + group_vecs[-2])
        results['s^2/2*mu1mu2'] = (self.spurious_corr ** 2 / 2) * results['mu1mu2']
        results['s(1-s)(m1+m2)(m-1+m-2)'] = self.spurious_corr * (1 - self.spurious_corr) * results['(m1+m2)(m-1+m-2)']
        results['(1-s)^2/2*mu-1mu-2'] = ((1 - self.spurious_corr) ** 2 / 2) * results['mu-1mu-2']
        a = group_vecs[1] + group_vecs[2]
        b = group_vecs[-1] + group_vecs[-2]
        d = a - b
        results['norm_a'] = torch.linalg.norm(a)
        results['norm_b'] = torch.linalg.norm(b)
        results['norm_d'] = torch.linalg.norm(d)
        
        results['E0/|a|/|b|'] = results['norm_a']/results['norm_b']

        results['c1'] = results['norm_mu1']**2 + results['norm_mu2']**2
        results['c2'] = results['norm_mu-1']**2 + results['norm_mu-2']**2
        results['c1-c2'] = results['c1'] - results['c2']
        results['d^2'] = results['norm_a']**2 + results['norm_b']**2 - 2*(results['(m1+m2)(m-1+m-2)'])
        results['-bTd'] = (
            -  results['(m1+m2)(m-1+m-2)']
            + results['norm_b']**2
        )
        results['E0/-bTd/d^2'] = results['-bTd']/results['d^2']
        results['(c1-c2)/d^2'] = results['c1-c2']/results['d^2']
        results['E0/s*'] = results['E0/-bTd/d^2'] + results['(c1-c2)/d^2']/(BATCH_SIZE-1)
        results['cosine_a_b'] = results['(m1+m2)(m-1+m-2)']/(results['norm_b']*results['norm_a'])

        # D = (
        #     (self.spurious_corr ** 2 / 2) * results['mu1mu2'] +
        #     self.spurious_corr * (1 - self.spurious_corr) * results['(m1+m2)(m-1+m-2)'] +
        #     ((1 - self.spurious_corr) ** 2 / 2) * results['mu-1mu-2'] -
        #     (
        #         (self.spurious_corr / 2) * results['norm_mu1'] ** 2 +
        #         (self.spurious_corr / 2) * results['norm_mu2'] ** 2 -
        #         ((1-self.spurious_corr) / 2) * results['norm_mu-1'] ** 2 -
        #         ((1-self.spurious_corr) / 2) * results['norm_mu-2'] ** 2
        #     )
        # )
        # results['D'] = D.item()
        # results['D/B2'] = D.item() / BATCH_SIZE**2

        # results["dI/dB"] = (
        #     1 / (4 * BATCH_SIZE**2) * torch.linalg.norm(self.spurious_corr * a + (1 - self.spurious_corr) * b) ** 2 -
        #     1 / (2 * BATCH_SIZE**2) * (self.spurious_corr * results["c1"] + (1 - self.spurious_corr) * results["c2"])
        # )
        
        # results['I(B,s)'] = (
        #     1/4*(1-1/BATCH_SIZE) * torch.linalg.norm(self.spurious_corr * a + (1 - self.spurious_corr) * b) ** 2 +
        #     1/(2*BATCH_SIZE) * (self.spurious_corr * results["c1"] + (1 - self.spurious_corr) * results["c2"])
        # )
        # beta = 1/4*(1-1/BATCH_SIZE)
        # alpha = 1/(2*BATCH_SIZE)
        
        # results['dI(B,s)/ds'] = (
        #     2*beta*(self.spurious_corr * results['norm_d']**2 + torch.dot(d, b)) + 
        #     alpha * (results['c1']-results['c2'])
        # )
        
        # def get_dI_ds(s):
        #     numerator = (
        #         (
        #             2*beta*(s*results['norm_d'] ** 2 + torch.dot(d, b)) + 
        #             alpha * (results['c1'] - results['c2'])
        #         ) * (s * l_maj + (1-s) * l_min) - 
        #         (
        #             beta * torch.linalg.norm(s * a + (1-s) * b) + 
        #             alpha * (s * results['c1'] + (1-s) * results['c2'])
        #         ) * (l_maj - l_min)
        #     )
        #     denumerator = (s * l_maj + (1-s) * l_min)**2
        #     return numerator, denumerator, numerator / denumerator

        # # results['d(I/C(w))/ds_numerator|s=0'], _, results['d(I/C(w))/ds|s=0'] = get_dI_ds(0)
        # # results['d(I/C(w))/ds_numerator|s=0.5'], _, results['d(I/C(w))/ds|s=0.5'] = get_dI_ds(0.5)
        # # results['d(I/C(w))/ds_numerator|s=1'], _, results['d(I/C(w))/ds|s=1'] = get_dI_ds(1)
        # results['d(I/C(w))/ds_numerator|s=s'], _, results['d(I/C(w))/ds|s=s'] = get_dI_ds(self.spurious_corr)
        # results['l_min/l_maj'] = l_min / l_maj
        # results['l_maj/l_min'] = l_maj / l_min
        
        return results


def calculate_flatness(model, loss_fn, dataloader, device, num_flat_iterations=2,
                            n_samples_per_iteration=128, epsilon=1e-3,
                            use_last_layer_only=True, **kwargs):
    model.eval()
    params = list(model.fc.parameters()) if use_last_layer_only else \
             [p for p in model.parameters() if p.requires_grad]
    
    # n_batches = int(np.ceil(n_samples_per_iteration/dataloader.batch_size))
    it = iter(dataloader)
    flat_values = []

    for i in range(num_flat_iterations):
        inputs, targets = [], []
        total_collected = 0
        while total_collected < n_samples_per_iteration:
            try:
                x, y = next(it)[:2]
            except StopIteration:
                it = iter(dataloader)
                continue
            inputs.append(x)
            targets.append(y)
            total_collected += x.size(0)
        inputs = torch.cat(inputs)[:n_samples_per_iteration].to(device)
        targets = torch.cat(targets)[:n_samples_per_iteration].to(device)
        
        # 2) sample one random unit direction
        vecs = [torch.randn_like(p) for p in params]
        norm = torch.sqrt(sum((v**2).sum() for v in vecs))
        vecs = [v / (norm + 1e-6) for v in vecs]

        original_params = [p.data.clone() for p in params]

        # 3) forward at θ + ε v
        with torch.no_grad():
            for p, v in zip(params, vecs):
                p.add_( epsilon * v)
            loss_pos = loss_fn(model(inputs), targets).item()

            for p, v in zip(params, vecs):
                p.add_(-2*epsilon * v)
            loss_neg = loss_fn(model(inputs), targets).item()

            # restore
            for p, orig_p in zip(params, original_params):
                p.data.copy_(orig_p)

            # 4) finite-difference directional derivative
            flat_val = abs(loss_pos - loss_neg) / (2*epsilon)
            flat_values.append(flat_val)
    return np.mean(flat_values)


# TODO: it is done only one iteration (one random init for v)
def calculate_sharpness(model, loss_fn, dataloader, device,
                        n_samples_per_iteration=128, num_sharp_iterations=5, power_iters=5,
                        use_last_layer_only=True, **kwargs):
    model.eval()
    
    if use_last_layer_only:
        params = list(model.fc.parameters())
    else:
        params = [p for p in model.parameters() if p.requires_grad]

    # Efficient sample collection
    
    it = iter(dataloader)
    sharpness_values = []

    for i in range(num_sharp_iterations):
        inputs, targets = [], []
        total_collected = 0
        while total_collected < n_samples_per_iteration:
            try:
                x, y = next(it)[:2]
            except StopIteration:
                it = iter(dataloader)
                continue
            inputs.append(x)
            targets.append(y)
            total_collected += x.size(0)
        inputs = torch.cat(inputs)[:n_samples_per_iteration].to(device)
        targets = torch.cat(targets)[:n_samples_per_iteration].to(device)

        # Gradient computation phase
        with torch.autograd.set_grad_enabled(True):
            loss = loss_fn(model(inputs), targets)
            grads = torch.autograd.grad(loss, params, create_graph=True)
        
        g_flat = torch.cat([g.contiguous().view(-1) for g in grads])

        # Power iteration with proper graph retention
        v = torch.randn_like(g_flat, requires_grad=False)
        v /= (v.norm() + 1e-6)
        
        for i in range(power_iters):
            # Retain graph for all iterations except the last one
            retain_graph = i < power_iters - 1
            
            # Hessian-vector product
            with torch.autograd.set_grad_enabled(True):
                Hv = torch.autograd.grad(
                    torch.dot(g_flat, v),
                    params,
                    create_graph=False,
                    retain_graph=retain_graph
                )
            
            Hv_flat = torch.cat([h.contiguous().view(-1) for h in Hv])
            
            # Update vector without gradient tracking
            with torch.no_grad():
                v = Hv_flat.detach()
                v /= (v.norm() + 1e-6)

        # Final sharpness calculation
        with torch.no_grad():
            sharpness = torch.dot(Hv_flat, v).item()
            sharpness_values.append(sharpness)
    
    return np.mean(sharpness_values)


def get_model_classifier(model):
    return list(model.children())[-1]


def get_model_feature_extractor(model):
    return torch.nn.Sequential(*list(model.children())[:-1], torch.nn.Flatten())


def get_last_layer_features_and_outputs(model, inputs):
    feature_extractor = get_model_feature_extractor(model)
    classifier_head = get_model_classifier(model)

    features = feature_extractor(inputs)
    features = torch.flatten(features, 1)
    outputs = classifier_head(features)

    return features, outputs


def extract_info(folder_name):
    """Extracts sp, batch size, and seed from the folder name."""
    match = re.search(r"(.+?)_nclass\d+_sp(\d*\.?\d+)_bs(\d+)_seed(\d+)", folder_name)
    if match:
        return (
            match.group(1),
            float(match.group(2)),
            int(match.group(3)),
            int(match.group(4)),
        )
    return None, None, None, None


def load_histories(history_path, mode, prefix=None):
    metric_data = {}
    for folder in os.listdir(history_path):
        if (prefix != None) and (not folder.startswith(prefix)):
            continue
        folder_path = os.path.join(history_path, folder)
        if os.path.isdir(folder_path):
            exp, sp, batch_size, seed = extract_info(folder)
            if sp is None:
                continue

            pkl_file = os.path.join(folder_path, f"{mode}_history.pkl")
            if not os.path.exists(pkl_file):
                continue

            with open(pkl_file, "rb") as f:
                history = json.load(f)

            for metric, values in history.items():
                if metric not in metric_data:
                    metric_data[metric] = []
                metric_data[metric].append((values, sp, batch_size, seed))

    return metric_data


def pretty_print(d: dict):
    print(yaml.dump(d, allow_unicode=True, default_flow_style=False))


def worst_group_accuracy(pred_label, labels, bias_labels):
    """Handles both tensors and numpy arrays."""

    if isinstance(pred_label, torch.Tensor):
        labels = labels.cpu().detach().numpy()
        pred_label = pred_label.cpu().detach().numpy()
        bias_labels = bias_labels.cpu().detach().numpy()

    groups = np.where(bias_labels == labels, labels + 1, -labels - 1)
    df = pd.DataFrame(
        {
            "labels": labels,
            "predictions": pred_label,
            "groups": groups,
        }
    )

    # TODO: can be simplified
    df["correct"] = (df["labels"] == df["predictions"]).astype(int)
    group_stats = df.groupby("groups")["correct"].agg(["sum", "count"])
    group_stats["accuracy"] = group_stats["sum"] / group_stats["count"]

    worst_group = group_stats["accuracy"].idxmin()
    worst_acc = group_stats.loc[worst_group, "accuracy"]
    group_accuracies = group_stats["accuracy"].to_dict()

    return worst_acc, group_accuracies


def get_features(loader, model, to_numpy=False, device="cuda"):
    features = []
    labels = []
    bias_labels = []
    logits = []
    bias = None

    for batch in loader:
        if len(batch) == 4:
            images, t, _, bias = batch
        elif len(batch) == 3:
            images, t, bias = batch
        elif len(batch) == 2:
            images, t = batch

        images = images.to(device)
        with torch.no_grad():
            feat, out = get_last_layer_features_and_outputs(model, images)

        if to_numpy:
            feat = feat.cpu().numpy()
            out = out.cpu().numpy()
            t = t.cpu().numpy()
            if bias is not None:
                bias = bias.cpu().numpy()

        features.append(feat)
        logits.append(out)
        labels.append(t)
        if bias is not None:
            bias_labels.append(bias)

    if to_numpy:
        features = np.concatenate(features)
        labels = np.concatenate(labels)
        bias_labels = np.concatenate(bias_labels) if len(bias_labels) else None
        logits = np.concatenate(logits)
    else:
        features = torch.cat(features)
        labels = torch.cat(labels)
        bias_labels = torch.cat(bias_labels)
        logits = torch.cat(logits)

    return features, labels, bias_labels, logits


def last_layer_grad(features, logits, labels):
    probs = F.softmax(logits, dim=-1)
    reg = 0
    for k in torch.unique(labels).detach().cpu().numpy():
        class_mask = labels == k
        pulls = -(1 - probs[class_mask][:, [k]]) * features[class_mask]
        pushes = probs[~class_mask][:, [k]] * features[~class_mask]
        batch_grad = torch.cat((pulls, pushes), dim=0).mean(dim=0)
        reg = reg + torch.linalg.vector_norm(batch_grad) ** 2
    return reg / 4



def subsample_dataset(dataset, return_loader=False, sampled_group_size=None, **kwargs):
    # Group indices by group labels
    group_indices = defaultdict(list)
    for idx in range(len(dataset)):
        _, target, _, spurious_label = dataset[idx]
        group_indices[f"{target}-{spurious_label}"].append(idx)
    
    # Find minimum group size
    if sampled_group_size is None:
        sampled_group_size = min(len(indices) for indices in group_indices.values())
    
    # Sample equally from each group
    balanced_indices = []
    for indices in group_indices.values():
        sampled = np.random.choice(indices, size=sampled_group_size, replace=False)
        balanced_indices.extend(sampled.tolist())
    
    # Shuffle indices
    np.random.shuffle(balanced_indices)
    
    balanced_dataset = Subset(dataset, balanced_indices)
    if return_loader:
        return DataLoader(balanced_dataset, **kwargs)
    return balanced_dataset



def replace_batchnorm(model, num_groups=32, transfer_weights=False):
    for name, module in model.named_children():
        if isinstance(module, nn.BatchNorm2d):
            num_channels = module.num_features
            
            new_gn = nn.GroupNorm(num_groups=num_groups, num_channels=num_channels).to(module.weight.device)
            
            # Transfer weight and bias parameters from BatchNorm to GroupNorm
            if transfer_weights:
                with torch.no_grad():
                    new_gn.weight.copy_(module.weight)
                    new_gn.bias.copy_(module.bias)
            
            setattr(model, name, new_gn)
        
        else:
            replace_batchnorm(module, num_groups=num_groups)