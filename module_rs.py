import torch
from torch.optim.lr_scheduler import LambdaLR
import pytorch_lightning as pl

# from pytorch_lightning.metrics import Accuracy
from loss import LabelSmoothingLoss
from models import resnet18  # , vgg16
from bisect import bisect
from torch import nn
from rs import RS
from torch.optim.lr_scheduler import LambdaLR


class CIFAR10ModuleRS(pl.LightningModule):
    def __init__(self, args):
        super().__init__()

        model_dict = {
            "resnet18": resnet18,
            "vgg16": None,
        }
        norm_dict = {
            "groupnorm": lambda x: nn.GroupNorm(x // args.group_size, x),
            "batchnorm": nn.BatchNorm2d,
            "none": nn.Identity,
        }
        self.criterion = LabelSmoothingLoss(
            10, args.smoothing, label_noise=args.label_noise
        )
        self.model = model_dict[args.model](
            num_classes=10, norm_layer=norm_dict[args.norm_layer]
        )
        self.args = args
        self.automatic_optimization = False

    def forward(self, batch):
        if self.args.freezeBN:
            self.model.eval()
        x, y = batch
        output = self.model(x)
        loss, trueloss = self.criterion(output, y)
        _, predictions = output.max(-1)
        accuracy = 100 * predictions.eq(y).sum() / len(y)
        return loss, trueloss, accuracy

    # def training_step(self, batch, batch_idx):
    #     optimizer = self.optimizers()
    #     optimizer.zero_grad()

    #     loss, trueloss, accuracy = self.forward(batch)
    #     print("fwd losses: ", loss.item(), trueloss.item())
    #     self.manual_backward(loss)
    #     optimizer.step()

    #     scheduler = self.lr_schedulers()
    #     if self.trainer.is_last_batch:
    #         scheduler.step()

    #     self.log("loss/train", trueloss, on_epoch=True)
    #     self.log("acc/train", accuracy, on_epoch=True)

    def training_step(self, batch, batch_idx):
        scheduler = self.lr_schedulers()
        if batch_idx == 0:
            scheduler.step()

        optimizer = self.optimizers()
        optimizer.zero_grad()

        optimizer.first_step(zero_grad=False)
        loss, trueloss, accuracy = self.forward(batch)
        self.manual_backward(loss)
        optimizer.second_step(zero_grad=False)

        self.log("loss/train", trueloss, on_epoch=True)
        self.log("acc/train", accuracy, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):
        _, trueloss, accuracy = self.forward(batch)
        self.log("loss/val", trueloss)
        self.log("acc/val", accuracy)

    def configure_optimizers(self):
        warmup_epochs = 300

        def lambda_scheduler(epoch):
            if epoch < warmup_epochs:
                return epoch / warmup_epochs
            else:
                return 1

        base_optimizer = torch.optim.SGD
        optimizer = RS(
            self.model.parameters(),
            base_optimizer,
            rho=self.args.rho,
            lr=self.args.lr,
            momentum=self.args.momentum,
            weight_decay=self.args.weight_decay,
        )

        scheduler = LambdaLR(optimizer, lr_lambda=lambda_scheduler)
        scheduler = {
            "scheduler": scheduler,
            "name": "learning_rate",
        }
        return [optimizer], [scheduler]

        # sgd_optimizer = torch.optim.SGD(
        #     self.model.parameters(),
        #     lr=self.args.lr,
        #     weight_decay=self.args.weight_decay,
        #     momentum=self.args.momentum,
        # )
        # scheduler = LambdaLR(sgd_optimizer, lr_lambda=lambda_scheduler)
        # scheduler = {
        #     "scheduler": scheduler,
        #     "name": "learning_rate",
        # }

        # return [sgd_optimizer], [scheduler]
