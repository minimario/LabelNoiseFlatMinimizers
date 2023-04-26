import argparse
import torch
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import LearningRateMonitor
from data import CIFAR10Data

# from module import CIFAR10Module
from module_rs import CIFAR10ModuleRS
from callbacks import *
from pathlib import Path
import wandb


def main(args):
    seed_everything(args.seed)
    logger = WandbLogger(
        entity="codegen", project="randomized-smoothing", name=args.name
    )
    logger.experiment.config.update(args)

    model = CIFAR10ModuleRS(args)
    # model = CIFAR10Module()
    if args.init is not None:
        model.load_state_dict(torch.load(Path(args.init)))
    data = CIFAR10Data(args)

    if args.callbacks:
        callbacks = [
            LearningRateMonitor(log_momentum=True),
            TimeEpoch(),
            TotalGradient(args),
            WeightNorm(),
        ]
    else:
        callbacks = [LearningRateMonitor(log_momentum=True), TimeEpoch()]

    if args.fullbatch:
        accumulate_grad_batches = 50000 // args.batch_size
        log_every_n_steps = 1
    else:
        accumulate_grad_batches = 1
        log_every_n_steps = 5000

    print("Logging every {} steps".format(log_every_n_steps))

    trainer = Trainer(
        logger=logger,
        max_epochs=args.max_epochs,
        callbacks=callbacks,
        log_every_n_steps=log_every_n_steps,
        precision=args.precision,
        deterministic=True,
        benchmark=True,
        accumulate_grad_batches=accumulate_grad_batches,
    )
    trainer.fit(model, data)
    if args.save:
        save_file = Path(args.save)
        save_file.parent.mkdir(exist_ok=True, parents=True)
        torch.save(model.state_dict(), save_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, help="name used for wandb logger")
    parser.add_argument("--init", type=str, help="initial weights to use")
    parser.add_argument(
        "--max_epochs", type=int, default=1000, help="number of epochs to run for"
    )
    parser.add_argument("--precision", type=int, default=32, help="precision to use")
    parser.add_argument("--seed", type=int, default=0, help="random seed")
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="number of workers to use for data loading",
    )
    parser.add_argument(
        "--save", type=str, default=None, help="output file to save model weights"
    )
    parser.add_argument(
        "--callbacks", action="store_true", help="whether to compute gradient callbacks"
    )
    parser.add_argument(
        "--fullbatch",
        action="store_true",
        help="whether to aggregate batches to emulate full batch training",
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=["resnet18", "vgg16"],
        default="resnet18",
        help="model to use",
    )
    parser.add_argument(
        "--norm_layer",
        type=str,
        choices=["groupnorm", "batchnorm", "none"],
        default="groupnorm",
        help="normalization layer to use for resnet",
    )
    parser.add_argument("--group_size", type=int, default=32, help="channels per group")
    parser.add_argument("--lr", type=float, default=1, help="learning rate")
    parser.add_argument("--momentum", type=float, default=0, help="momentum")
    parser.add_argument("--weight_decay", type=float, default=0, help="weight decay")
    parser.add_argument(
        "--smoothing",
        type=float,
        default=0,
        help="probability of flipping a label for label smoothing/label noise",
    )
    parser.add_argument(
        "--label_noise",
        action="store_true",
        help="whether to use randomized label noise instead of label smoothing",
    )
    parser.add_argument(
        "--warmup",
        type=float,
        default=0,
        help="length of warmup phase (between 0 and 1)",
    )
    parser.add_argument(
        "--div_start",
        type=float,
        default=float("inf"),
        help="factor to divide learning rate by during warmup",
    )
    parser.add_argument(
        "--div_end",
        type=float,
        default=float("inf"),
        help="factor to divide learning rate by during cosine annealing",
    )
    parser.add_argument(
        "--freezeBN",
        action="store_true",
        help="whether to freeze batch norm during training",
    )
    parser.add_argument("--rho", type=float, default=2, help="rho for SAM")

    parser.add_argument("--batch_size", type=int, default=256, help="batch size")
    parser.add_argument(
        "--DA", action="store_true", help="whether to use data augmentation"
    )
    parser.add_argument(
        "--rand_data",
        action="store_true",
        help="whether to use random data/labels for adversarial init",
    )

    args = parser.parse_args()
    main(args)
