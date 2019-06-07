import datetime
import logging
import config
import os
import tqdm
import random

import numpy as np
import scipy.stats
import torch.nn.functional as F
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter

from dataset import BrownDataset, random_split
from model import LM
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

INDEX = 0


class LMTrainer:
    def __init__(self, model, dataset, checkpoint_filename):
        self.checkpoint_filename = checkpoint_filename
        self.model = model.cuda()
        self.dataset = dataset

        self.writer = SummaryWriter()
        self.global_step = 0
        self.last_log = datetime.datetime.now()
        self.last_valid = datetime.datetime.now()

        # Dataset
        train_dataset, test_dataset = random_split(dataset, [90, 10])

        self.loaders = {
            "train": DataLoader(
                train_dataset,
                batch_size=config.BATCH_SIZE,
                shuffle=True,
                drop_last=True,
            ),
            "test": DataLoader(
                test_dataset, batch_size=config.BATCH_SIZE, shuffle=True, drop_last=True
            ),
        }
        inputs = next(iter(self.loaders["test"]))
        model(inputs.cuda())
        self.writer.add_graph(model, inputs.cuda())

        # Optimizer
        self.optim = torch.optim.lr_scheduler.CyclicLR(
            torch.optim.SGD(
                self.model.parameters(), lr=config.LEARNING_RATE, momentum=0.9
            ),
            base_lr=0,
            max_lr=config.LEARNING_RATE,
        )

        self.best_valid = 1e9
        self.patience = 0

    def run_batch_learn(self, i, batch):
        self.optim.optimizer.zero_grad()
        self.model.train()
        loss, metrics = self.compute_loss_and_metrics_from_indices(batch)
        loss.backward()
        self.optim.optimizer.step()
        self.optim.step()
        self.writer.add_scalar(
            "train/lr",
            torch.Tensor([self.optim.get_lr()]),
            global_step=self.global_step,
        )
        self.global_step += config.BATCH_SIZE

    def evaluate_batch(self, mode, batch):
        loss, metrics = self.compute_loss_and_metrics_from_indices(batch)
        for name, metric in sorted(metrics.items()):
            scalar_name = f"{mode}/{name}"
            self.writer.add_scalar(scalar_name, metric, global_step=self.global_step)
        return loss

    def evaluate(self, mode):
        total_loss = 0
        with torch.no_grad():
            self.model.eval()
            loader = self.loaders[mode]
            for batch in tqdm.tqdm(loader, total=len(loader), desc=mode):
                batch = batch.cuda()
                loss = self.evaluate_batch(mode, batch)
                total_loss += loss.item()
        return total_loss

    def run_epoch(self):
        loader = self.loaders["train"]
        for i, batch in tqdm.tqdm(enumerate(loader), total=len(loader), desc="train"):
            batch = batch.cuda()
            self.run_batch_learn(i, batch)
            now = datetime.datetime.now()
            if now - self.last_log > config.LOG_INTERVAL:
                self.last_log = now
                self.evaluate_batch("train", batch)

        if now - self.last_valid > config.TEST_INTERVAL:
            valid_loss = self.evaluate("test")
            self.last_valid = now

            if valid_loss < self.best_valid:
                torch.save(self.model, self.checkpoint_filename)
                self.best_valid = valid_loss
                self.patience = 0
            else:
                self.patience += 1
                msg = f"Did not improve valid loss, model not saved (patience: {self.patience})"
                logging.info(msg)

    def compute_loss_and_metrics_from_indices(self, indices):
        attn = self.model.forward(indices)  # attn: [B, C, D]
        decoded = self.model.decoder(attn)  # decoded: [B, C, V]
        B, C, V = decoded.shape
        x = (
            decoded[:, :-1, :].contiguous().view(-1, decoded.size(-1))
        )  # [B * (C - 1), V]
        logits = F.log_softmax(x, dim=-1)  # [B * (C - 1), V]
        next_words = indices[:, 1:].contiguous().view(-1)  # [B * (C - 1)]
        prediction_loss = F.nll_loss(logits, next_words, reduction="sum") / attn.size(0)
        loss_words = prediction_loss / config.CONTEXT_SIZE
        _, topk = logits.topk(10, -1)
        topks = {
            f"P_{k}": (topk[:, :k] == next_words.view(-1, 1)).sum(-1).float().mean()
            * 100
            for k in [1, 5, 10]
        }

        losses = {"loss/context": prediction_loss, "loss/words": loss_words, **topks}
        loss = prediction_loss
        if random.random() < 0.01:
            plt.clf()
            values = torch.exp(logits.reshape(B, C - 1, V)[0, -1, :])
            plt.plot(values.detach().cpu().numpy())
            mu = 500
            sigma = 100
            x = np.linspace(mu - 5 * sigma, mu + 5 * sigma, 100)
            plt.plot(x, scipy.stats.norm.pdf(x, mu, sigma))
            mu = 500
            sigma = 10
            x = np.linspace(mu - 5 * sigma, mu + 5 * sigma, 100)
            plt.plot(x, scipy.stats.norm.pdf(x, mu, sigma))
            global INDEX
            filename = f"figures/out{INDEX}.png"
            plt.savefig(filename)
            INDEX += 1
            print(f"Saved {filename}")
        return loss, losses


def train():
    # generate file paths
    now = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    savedir = os.path.join(config.SAVE_DIR, "hierarchical_bottom", f"{now}")
    os.makedirs(savedir, exist_ok=True)
    checkpointfn = os.path.join(savedir, "checkpoint.model")
    logfn = os.path.join(savedir, "run.log")
    torch.manual_seed(config.SEED)

    # create logger
    logger = logging.getLogger()
    if logger.hasHandlers():
        logger.handlers.clear()

    logger.setLevel(logging.DEBUG)

    fmt = logging.Formatter("%(asctime)s %(levelname)-8s: %(message)s")
    console = logging.StreamHandler()
    console.setFormatter(fmt)
    logger.addHandler(console)

    logfile = logging.FileHandler(logfn, "a")
    logfile.setFormatter(fmt)
    logfile.setLevel(logging.DEBUG)
    logger.addHandler(logfile)

    with open("config.py", "r") as f:
        for l in f:
            logging.debug(l.strip())

    dataset = BrownDataset(config.CONTEXT_SIZE)
    model = LM()
    num_params = sum(p.numel() for p in model.parameters())
    logging.debug(f"The model has {num_params:,} parameters")

    # Trainer init
    logging.debug("Initiate the training environment")
    trainer = LMTrainer(model, dataset, checkpointfn)

    # Training
    logging.debug("Starting the training")
    for epoch in tqdm.tqdm(range(config.EPOCHS), total=config.EPOCHS, desc="EPOCH"):
        trainer.run_epoch()
        if trainer.patience > config.PATIENCE:
            logging.info("patience over {}, exiting".format(config.PATIENCE))
            break


if __name__ == "__main__":
    train()
