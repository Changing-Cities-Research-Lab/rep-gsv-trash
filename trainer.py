

import os
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.tensorboard as tbx
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.utils as vutils
from torchmetrics.classification import MulticlassConfusionMatrix
import matplotlib.pyplot as plt
import seaborn as sn
import pandas as pd
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas




def plot_confusion_matrix(c_m):
  """

  Args:
      c_m: confusion matrix tensor
  Returns:
      returns the plt of the matrix as a heatmap as a quick and easy way to observe the confusion matrix.

  """

  num_classes = c_m.shape[0]
  df_cm = pd.DataFrame(c_m.detach().cpu().numpy() , index = [i for i in range(num_classes)],
                  columns = [i for i in range(num_classes)])
  fig, ax = plt.subplots(figsize=(10,7))
  sn.heatmap(df_cm, annot=True, ax=ax)
  return fig


def get_output_label(out, classes):
    """

    Args:
        out: the output of the classifier
        classes: number of classes

    Returns:
        returns the onehot encoding for the output of the classifier

    """
    val, index = torch.max(out, dim =1)
    return F.one_hot(index, num_classes = classes)


def get_accuracy(eval_labels, output_labels):
    """

    Args:
        eval_labels: truth labels
        output_labels: onehot output labels of the model
    Returns:
        accuracy of the predictions vs the truth labels
    """
    temp = F.one_hot(eval_labels, output_labels.shape[1])
    acc = torch.where(temp == output_labels, temp, 0).to(torch.float)
    summed = acc.sum(axis=1)
    return torch.mean(summed)


def evaluate(model, device, classes, l_f, eval_dataloader):
    """
    Args:
        model: Classifier model being trained
        device: the device used: cpu or
        classes:
        l_f:
        eval_dataloader:

    Returns:

    """
    model.to(device).eval()
    with torch.no_grad():
        conf_total = None
        accuracy, losses = ([],[])
        for data, label in tqdm(eval_dataloader,desc="Evaluating Model"):
            data = data.to(device)
            label = label.to(device)
            out = model(data)
            loss = l_f(out, label)
            output_label = get_output_label(out,classes)
            acc = get_accuracy(label, output_label)
            conf = MulticlassConfusionMatrix(num_classes=out.shape[1]).to(device)
            curr = conf(torch.max(out, dim=1)[1], label)
            if conf_total is None:
              conf_total = curr
            else:
              conf_total += curr

            accuracy.append(acc)
            losses.append(loss)

        metrics = {
            "Loss": torch.stack(losses).mean().item(),
            "Accuracy": torch.stack(accuracy).mean().item()
        }
        print("Confusion Matrix", conf_total)
        print(metrics)
        figure = plot_confusion_matrix(conf_total)
        canvas = FigureCanvas(figure)
        canvas.draw()
        image = np.frombuffer(canvas.tostring_rgb(), dtype='uint8')
        image = image.reshape(canvas.get_width_height()[::-1] + (3,))
        image = np.transpose(image, (2, 0, 1))

        return metrics, image


class Trainer:
    """"

    Trainer performs training, checkpointing and logging.
    Attributes:
        model (Module): Torch model.
        opt (Optimizer): Torch optimizer for model.
        sch (Scheduler): Torch lr scheduler.
        train_dataloader (Dataloader): Torch train dataloader.
        eval_dataloader (Dataloader): Torch eval dataloader.
        log_dir (str): Path to store log outputs.
        ckpt_dir (str): Path to store and load checkpoints.
        device (Device): Torch device to perform training on.
    """

    def __init__(
        self,
        model,
        opt,
        sch,
        loss,
        train_dataloader,
        eval_dataloader,
        log_dir,
        ckpt_dir,
        device,
    ):
        # Setup models, dataloader, optimizers
        self.model = model.to(device)
        self.loss = loss
        self.opt = opt
        self.sch = sch
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader



        # Setup training parameters
        self.device = device
        self.step = 0
        self.ckpt_dir = ckpt_dir

        self.logger = tbx.SummaryWriter(log_dir)

    def _log(self, metrics, samples):
        r"""
        Logs metrics and samples to Tensorboard.
        """
        for k, v in metrics.items():
            self.logger.add_scalar(k, v, self.step)
        self.logger.add_image("Samples", samples, self.step)
        self.logger.flush()
    def _state_dict(self):
        return {
            "model": self.model.state_dict(),
            "opt": self.opt.state_dict(),
            "sch": self.sch.state_dict(),
            "step": self.step,
        }
    def _load_state_dict(self, state_dict):
        """

        Args:
            state_dict: given dictionary of checkpoint

        initiates model with the saved dictionary values.

        """
        self.model.load_state_dict(state_dict["model"])
        self.opt.load_state_dict(state_dict["opt"])
        self.sch.load_state_dict(state_dict["sch"])
        self.step = state_dict["step"]



    def _load_checkpoint(self, is_best = False):
        r"""
        Finds the last checkpoint in ckpt_dir and load states.
        is_best: parameter that loads model with best validation score if exists.
        """

        ckpt_paths = [f for f in os.listdir(self.ckpt_dir) if f.endswith(".pth")]
        if is_best and "best.pth" in ckpt_paths:# loads best if is_best is true
            print("Found early stopping model continuing from early stopping point.")
            self._load_state_dict(torch.load("best.pth"))
        if ckpt_paths:  # Train from scratch if no checkpoints were found
            ckpt_path = sorted(ckpt_paths, key=lambda f: int(f[:-4]))[-1]
            ckpt_path = os.path.join(self.ckpt_dir, ckpt_path)
            self._load_state_dict(torch.load(ckpt_path))



    def _save_checkpoint(self, is_best = False):
        r"""
        Saves model, optimizer and trainer states.
        """
        step = str(self.step) if not is_best else "best"
        ckpt_path = os.path.join(self.ckpt_dir, f"{step}.pth")
        # Remove ckpt_path if it already exists
        if os.path.exists(ckpt_path):
            os.remove(ckpt_path)
        torch.save(self._state_dict(), ckpt_path)





    def train(self, max_steps,  eval_every, ckpt_every):
        r"""
        Performs training, checkpointing and logging.
        Attributes:
            max_steps (int): Number of steps before stopping.
            repeat_d (int): Number of discriminator updates before a generator update.
            eval_every (int): Number of steps before logging to Tensorboard.
            ckpt_every (int): Number of steps before checkpointing models.

        """

        self._load_checkpoint()
        best = float('inf')
        while True:
                pbar = tqdm(self.train_dataloader)
                for data, label in pbar:

                    # Training step

                    data = data.to(self.device)
                    label = label.to(self.device)
                    self.opt.zero_grad()
                    self.model.train()
                    out = self.model(data)
                    loss = self.loss(out, label)
                    self.model.zero_grad()
                    loss.backward()
                    self.opt.step()
                    self.sch.step()


                    pbar.set_description(
                        f"L(G):{loss.item():.2f}|{self.step}/{max_steps}"
                    )

                    if self.step != 0 and self.step % eval_every == 0:

                        metrics, image = evaluate(
                                self.model,
                                self.device,
                                out.shape[1],
                                self.loss,
                                self.eval_dataloader
                            )

                        if metrics['Loss'] < best:
                            best = metrics['Loss']
                            self._save_checkpoint(True)

                        self._log(
                            metrics, image
                        )


                    if self.step != 0 and self.step % ckpt_every == 0:
                        self._save_checkpoint()

                    self.step += 1
                    if self.step > max_steps:
                        return


