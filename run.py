from ast import arg
import os
from pathlib import Path
import sys
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
import src.cmnist_dataset
import wandb
import torch.nn as nn

from torchvision import transforms, datasets, models
from torch.utils.data import DataLoader, Dataset
from torch.utils.data import random_split
from collections import Counter
from torch import optim
from torch.optim import lr_scheduler


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import src
from src.models import get_model
from src.utils import pretty_print, GradientTracker, subsample_dataset
from src.datasets import DatasetGetter


def comma_seperated_ints(input_string):
    return [int(item.strip()) for item in input_string.split(",")]


def get_dominocmf_args(parser):
    parser.add_argument(
        "--domniocmf_train_count",
        required=False,
        default=10000,
        type=int,
        help="Count of train data in DominoCMF dataset.",
    )
    parser.add_argument(
        "--domniocmf_val_count",
        required=False,
        default=10000,
        type=int,
        help="Count of val data in DominoCMF dataset.",
    )
    parser.add_argument(
        "--domniocmf_test_count",
        required=False,
        default=4000,
        type=int,
        help="Count of test data in DominoCMF dataset.",
    )
    parser.add_argument(
        "--dominocmf_shape_correlation",
        required=False,
        default=0.75,
        type=float,
        help="Correlation between shape and true label in DominoCMF dataset.",
    )
    parser.add_argument(
        "--dominocmf_color_correlation",
        required=False,
        default=0.95,
        type=float,
        help="Correlation between color and true label in DominoCMF dataset.",
    )
    parser.add_argument(
        "--dominocmf_val_group_policy",
        required=False,
        default="shape",
        type=str,
        help="Policy for determining how each group is labeled in the validation dataset.",
        choices=["shape", "color", "combined"],
    )
    parser.add_argument(
        "--dominocmf_test_group_policy",
        required=False,
        default="combined",
        type=str,
        help="Policy for determining how each group is labeled in the test dataset.",
        choices=["shape", "color", "combined"],
    )
    return parser


def get_args_parser():
    parser = argparse.ArgumentParser(description="Trainer")
    parser.add_argument(
        "--dataset",
        required=False,
        default="cifar10",
        type=str,
        help="name of the dataset being cifar10 or domino",
    )
    parser.add_argument(
        "--selected_classes",
        required=False,
        default=None,
        type=comma_seperated_ints,
        help="Comma-separated list of selected classes",
    )

    parser.add_argument("--batch_size", type=int, default=128, help="batch size")
    parser.add_argument("--spurious_correlation", type=float, default=0.5, help="spurious correlation")
    parser.add_argument("--exp_name", required=False, default="verify", type=str, help="experiment name")
    parser.add_argument(
        "--checkpoint_path",
        required=False,
        type=str,
        default="experiments/checkpoints/",
        help="path to checkpoints",
    )
    parser.add_argument(
        "--history_path",
        required=False,
        type=str,
        default="experiments/history/",
        help="path to history",
    )
    parser.add_argument(
        "--cifar_datapath",
        required=False,
        type=str,
        default="data/CIFAR10",
        help="path to cifar dataset",
    )
    parser.add_argument(
        "--celeba_datapath",
        required=False,
        type=str,
        default="data/CelebA",
        help="path to celeba dataset",
    )
    parser.add_argument(
        "--celeba_super_balanced", 
        type=int,
        default=0,
        choices=[0, 1],
        help="If 1 and dataset=celeba, make train split 1387 samples per subgroup (y,gender).",
    )
    parser.add_argument(
        "--celeba_val_super_balanced",
        type=int,
        default=0,
        choices=[0, 1],
        help="If 1 and dataset=celeba, make val split 1387 samples per subgroup (y,gender).",
    )
    parser.add_argument(
        "--waterbirds_datapath",
        required=False,
        type=str,
        default="data/Waterbirds",
        help="path to waterbirds dataset",
    )
    parser.add_argument(
        "--mnist_datapath",
        required=False,
        type=str,
        default="data/MNIST",
        help="path to mnist dataset",
    )
    parser.add_argument(
        "--fmnist_datapath",
        required=False,
        type=str,
        default="data/FMNIST",
        help="path to fmnist dataset",
    )
    parser.add_argument(
        "--architecture",
        required=False,
        type=str,
        default="resnet18",
        choices=["mlp", "resnet18", "resnet50"],
        help="architecture",
    )
    parser.add_argument("--gpu_number", type=int, default=0, help="gpu number")
    parser.add_argument("--image_size", type=int, default=32, help="image size")
    parser.add_argument("--random_seed", type=int, default=0, help="random seed")
    parser.add_argument("--epochs", type=int, default=100, help="epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-5, help="weight_decay")
    parser.add_argument(
        "--full_metrics",
        type=int,
        default=0,
        choices=[0, 1],
        help="calculate NC, diversity metrics",
    )
    parser.add_argument(
        "--wandb_run",
        help="Name for the wandb run. The model logs will be saved at `run/dataset/{wandb_run}_run_number/`",
    )
    parser.add_argument("--num_workers", type=int, default=8, help="number of workers for data loaders")
    parser.add_argument(
        "--model_selection",
        type=str,
        default="wga",
        choices=["wga", "acc"],
        help="Model selection by acc or wga of validation representing known and unknown.",
    )
    parser.add_argument(
        "--setup",
        type=str,
        default="unknown",
        choices=["unknown", "known"],
        help="known and unknown for validation spurious correlation",
    )
    parser.add_argument(
        "--scheduler",
        type=str,
        default="",
        choices=["", "step", "cosine", "plateau"],
        help="scheduler type",
    )
    parser.add_argument(
        "--reg_type",
        type=str,
        default="",
        choices=["", "batch_grad_norm", "batch_grad_norm_last_layer"],
        help="regularization type",
    )
    parser.add_argument(
        "--reg_strength",
        type=float,
        default=0.0,
        help="regularization strength",
    )
    parser.add_argument(
        "--mlp_hidden_dims",
        type=comma_seperated_ints,
        default="256,256",
        help="comma seperated list of mlp hidden dims",
    )
    parser.add_argument(
        "--track_grad",
        type=int,
        default=0,
        choices=[0, 1],
        help="exhustive gradient tracking - prefer big batch size",
    )
    parser.add_argument(
        "--dfr_balanced_dataset",
        type=str,
        default="val",
        choices=["val", "train"],
    )
    parser.add_argument(
        "--run_dfr",
        type=int,
        default=0,
        choices=[0, 1],
    )
    parser.add_argument(
        "--run_sharp_flat",
        type=int,
        default=0,
        choices=[0, 1],
    )
    parser.add_argument(
        "--max_batch_size",
        type=int,
        default=0,
        help="maximum batch size. If batch_size is larger than this, it will trigger gradient accumulation.",
    )
    parser.add_argument(
        "--replace_batchnorm",
        type=int,
        default=0,
        choices=[0, 1],
    )
    parser.add_argument(
        "--force_calculate_reg",
        type=int,
        default=0,
        choices=[0, 1],
    )
    parser.add_argument(
        "--no_val_validation",
        type=int,
        default=0,
        choices=[0, 1],
        help="If 1, skip validation each epoch and select model from last epoch.",
    )
    parser = get_dominocmf_args(parser)
    return parser


def main():
    parser = argparse.ArgumentParser("Trainer", parents=[get_args_parser()], add_help=False)
    args = parser.parse_args()
    if args.wandb_run is None:
        print(f"No wandb name generated by us.")

    wandb.init(
        project=f"TrainingDynamics_{args.dataset}",
        config={**vars(args)},
        settings=wandb.Settings(code_dir="."),
    )

    print(f"*** Run Config *** ")
    pretty_print({**vars(args)})

    device = f"cuda:{args.gpu_number}" if torch.cuda.is_available() else "cpu"

    print("Running on {} ...".format(device))

    torch.multiprocessing.set_sharing_strategy("file_system")

    np.random.seed(args.random_seed)
    generator = torch.Generator().manual_seed(args.random_seed)
    torch.manual_seed(args.random_seed)

    transf = (
        transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )
        if "resnet" in args.architecture
        else transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )
    )

    mnist_selected_classes = None
    match args.dataset:
        case "cifar10":
            root = args.cifar_datapath
            selected_classes = args.selected_classes if args.selected_classes is not None else [1, 9]
            num_classes = len(selected_classes)
        case "cmnist":
            root = args.mnist_datapath
            selected_classes = args.selected_classes if args.selected_classes is not None else np.arange(10, dtype=int)
            num_classes = len(selected_classes)
        case "domino":
            root = args.cifar_datapath
            selected_classes = args.selected_classes if args.selected_classes is not None else [1, 9]
            mnist_selected_classes = args.selected_classes if args.selected_classes is not None else [0, 1]
            num_classes = len(selected_classes)
        case "waterbirds":
            root = args.waterbirds_datapath
            selected_classes = args.selected_classes if args.selected_classes is not None else [0, 1]
            num_classes = 2
        case "celeba":
            root = args.celeba_datapath
            selected_classes = args.selected_classes if args.selected_classes is not None else [0, 1]
            num_classes = 2
        case _:
            raise ValueError("Invalid dataset")

    VAL_RATIO = 0.2
    datagetter = DatasetGetter(args.dataset, root, args.spurious_correlation, VAL_RATIO, selected_classes, seed=args.random_seed)
    extra_kwargs = {"mnist_selected_classes": mnist_selected_classes, "args": args}
    trainset, valset, testset, testset_biased = [
        datagetter.get_dataset(split, setup, transf, **extra_kwargs)
        for split, setup in [("train", "unknown"), ("val", args.setup), ("test", "known"), ("test", "unknown")]
    ]

    if args.dataset == "dominocmf":
        raise NotImplementedError("Need to implement testset_biased")
        # trainset, valset, testset = src.dominocmf_dataset.get_domino_cmf_datasets(args)
        # num_classes = testset.n_classes

    if args.max_batch_size > 0 and args.batch_size > args.max_batch_size:
        if args.batch_size % args.max_batch_size != 0:
            raise ValueError(f"max_batch_size {args.max_batch_size} should divide batch_size {args.batch_size}")
        mini_batch_size = args.max_batch_size
        accumulation_steps = args.batch_size // args.max_batch_size
    else:
        mini_batch_size = args.batch_size
        accumulation_steps = None

    trainloader = DataLoader(
        trainset,
        batch_size=mini_batch_size,
        num_workers=args.num_workers,
        shuffle=True,
        pin_memory=True,
    )
    valloader = DataLoader(
        valset,
        batch_size=mini_batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        pin_memory=True,
    )
    testloader = DataLoader(
        testset,
        batch_size=mini_batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        pin_memory=True,
    )
    testloader_biased = DataLoader(
        testset_biased,
        batch_size=mini_batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        pin_memory=True,
    )

    model = get_model(
        model_name=args.architecture,
        num_classes=num_classes,
        mlp_hidden_dims=args.mlp_hidden_dims,
        device=device,
    )
    if args.replace_batchnorm == 1:
        print("Replacing all batchnorm layers with groupnorm (num_groups=32)")
        # hardcoding num_groups=32 for now
        src.utils.replace_batchnorm(model, num_groups=32, transfer_weights=False)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, momentum=0.9)
    
    if args.scheduler == "step":
        scheduler = lr_scheduler.StepLR(optimizer, step_size=40, gamma=0.1)
    elif args.scheduler == "cosine":
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    elif args.scheduler == "plateau":
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.1)
    else:
        scheduler = None

    checkpoint_path = Path(args.checkpoint_path)
    checkpoint_path.mkdir(exist_ok=True, parents=True)
    model_path = os.path.join(
        checkpoint_path,
        f"{wandb.run.name}_dataset{args.dataset}_nclass{num_classes}_sp{args.spurious_correlation}_bs{args.batch_size}_lr{args.lr}_setup{args.setup}_seed{args.random_seed}",
    )

    grad_tracker = None
    if args.track_grad:
        classes = np.arange(num_classes, dtype=int)
        groups = np.concatenate([classes + 1, -classes - 1])
        groups = torch.from_numpy(groups).to(device)
        print(f"Using gradient tracking for groups: {groups}")
        grad_tracker = GradientTracker(
            groups=groups,
            criterion=criterion,
            device=device,
            spurious_corr=args.spurious_correlation,
        )

    trainer = src.trainer.Trainer(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        model_path=model_path,
        log=wandb.log,
        reg_type=args.reg_type,
        reg_strength=args.reg_strength,
        grad_tracker=grad_tracker,
        accumulation_steps=accumulation_steps,
        force_calculate_reg=bool(args.force_calculate_reg)
    )

    trainer.train(
        trainloader,
        valloader,
        args.epochs,
        full_metrics=args.full_metrics,
        validate_each_epoch=(args.no_val_validation == 0),
    )

    if grad_tracker is None:  # in order to not run during test
        if args.dfr_balanced_dataset == "train":
            balanced_loader = subsample_dataset(
                trainset,
                return_loader=True,
                batch_size=mini_batch_size,
                num_workers=args.num_workers,
                shuffle=False,
                pin_memory=True,
            )
        else:
            balanced_loader = valloader

        if args.no_val_validation == 0:
            # Standard behavior: validate each epoch and select best by args.model_selection
            trainer.test(
                testloader,
                balanced_loader=balanced_loader,
                dataset="test",
                model_selection=args.model_selection,
                run_dfr=args.run_dfr,
                run_sharp_flat=args.run_sharp_flat,
            )
            trainer.test(
                testloader,
                balanced_loader=balanced_loader,
                dataset="test",
                model_selection="last_epoch",
                run_dfr=args.run_dfr,
                run_sharp_flat=args.run_sharp_flat,
            )

            trainer.test(
                testloader_biased,
                balanced_loader=balanced_loader,
                dataset="test_biased",
                model_selection=args.model_selection,
                run_dfr=args.run_dfr,
                run_sharp_flat=args.run_sharp_flat,
            )
            trainer.test(
                testloader_biased,
                balanced_loader=balanced_loader,
                dataset="test_biased",
                model_selection="last_epoch",
                run_dfr=args.run_dfr,
                run_sharp_flat=args.run_sharp_flat,
            )

            trainer.test(
                trainloader,
                balanced_loader=balanced_loader,
                dataset="train",
                model_selection=args.model_selection,
                run_dfr=args.run_dfr,
                run_sharp_flat=args.run_sharp_flat,
            )
            trainer.test(
                trainloader,
                balanced_loader=balanced_loader,
                dataset="train",
                model_selection="last_epoch",
                run_dfr=args.run_dfr,
                run_sharp_flat=args.run_sharp_flat,
            )
        else:
            # No per-epoch validation: evaluate the last epoch checkpoint
            trainer.test(
                testloader,
                balanced_loader=balanced_loader,
                dataset="test",
                model_selection="last_epoch",
                run_dfr=args.run_dfr,
                run_sharp_flat=args.run_sharp_flat,
            )
            trainer.test(
                testloader_biased,
                balanced_loader=balanced_loader,
                dataset="test_biased",
                model_selection="last_epoch",
                run_dfr=args.run_dfr,
                run_sharp_flat=args.run_sharp_flat,
            )
            trainer.test(
                trainloader,
                balanced_loader=balanced_loader,
                dataset="train",
                model_selection="last_epoch",
                run_dfr=args.run_dfr,
                run_sharp_flat=args.run_sharp_flat,
            )

    wandb.finish()


if __name__ == "__main__":
    main()
