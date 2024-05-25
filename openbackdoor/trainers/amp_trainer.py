from openbackdoor.victims import Victim, PLMVictim, MLMVictim
from openbackdoor.utils import logger, evaluate_classification
from openbackdoor.data import get_dataloader, wrap_dataset
from transformers import  AdamW, get_linear_schedule_with_warmup
import torch
from datetime import datetime
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset
import os
from tqdm import tqdm
from typing import *
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from umap import UMAP
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib.font_manager as fm
from torch import autograd
import seaborn as sns
import copy
import pickle
import json

# from accelerate import Accelerator
# from torch.cuda.amp import autocast, GradScaler
import deepspeed
from .trainer import Trainer, getHighDimFreq
deepspeed.ops.op_builder.CPUAdamBuilder().load()

class AMPTrainer(Trainer):
    def __init__(self, cmdArgs, **kwargs):
        super(AMPTrainer, self).__init__(**kwargs)
        self.cmdArgs = cmdArgs
    
    def register(self, model: Union[Victim, PLMVictim, MLMVictim], dataloader, metrics):
        r"""
        Register model, dataloader and optimizer
        """
        self.model = model
        logger.info("cast model to float16")
        self.metrics = metrics
        self.main_metric = self.metrics[0]
        self.split_names = dataloader.keys()
        self.model.train()
        self.model.zero_grad()
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in self.model.named_parameters() if (not any(nd in n for nd in no_decay)) and p.requires_grad], 'weight_decay': self.weight_decay},
            {'params': [p for n, p in self.model.named_parameters() if (any(nd in n for nd in no_decay)) and p.requires_grad], 'weight_decay': 0.0}
            ]
        self.optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=self.lr)
        
        self.modelEngine, self.optimizer, _, _ = deepspeed.initialize(args=self.cmdArgs, model=self.model, model_parameters=optimizer_grouped_parameters)
        
        train_length = len(dataloader["train"])
        
        self.poison_loss_all = []
        self.normal_loss_all = []
        if self.visualize:
            poison_loss_before_tuning, normal_loss_before_tuning = self.comp_loss(model, dataloader["train"])
            self.poison_loss_all.append(poison_loss_before_tuning)
            self.normal_loss_all.append(normal_loss_before_tuning)
            self.hidden_states, self.labels, self.poison_labels = self.compute_hidden(model, dataloader["train"])
            
            devDataset = ConcatDataset([dataloader['dev-clean'].dataset, dataloader['dev-poison'].dataset])
            devDataloader = DataLoader(devDataset, batch_size=dataloader['dev-clean'].batch_size, collate_fn=dataloader['dev-clean'].collate_fn)
            
            self.dev_hidden_states, self.dev_labels, self.dev_poison_labels = self.compute_hidden(model, devDataloader)
            
        
        
        # Train
        logger.info("***** Training *****")
        logger.info("  Num Epochs = %d", self.epochs)
        logger.info("  Instantaneous batch size per GPU = %d", self.batch_size)
        logger.info("  Gradient Accumulation steps = %d", self.gradient_accumulation_steps)
        logger.info("  Total optimization steps = %d", self.epochs * train_length)
        
        if self.frequencyConfig['frequencyVis']:
            logger.info("\nRegister Frequency Infomation\n")
            train_dataloader = dataloader["train"]
            trainDataset = train_dataloader.dataset
            trainCleanDataset, trainPoisonDatset = [data for data in trainDataset if data[2] == 0], [data for data in trainDataset if data[2] == 1]
            
            self.staticDataLoaders = {
                'train-clean':DataLoader(trainCleanDataset, batch_size=train_dataloader.batch_size,  collate_fn=train_dataloader.collate_fn, shuffle=False), 
                'train-poison':DataLoader(trainPoisonDatset, batch_size=train_dataloader.batch_size, collate_fn=train_dataloader.collate_fn, shuffle=False), 
                'dev-clean':DataLoader(dataloader['dev-clean'].dataset, batch_size=train_dataloader.batch_size, collate_fn=train_dataloader.collate_fn, shuffle=False), 
                'dev-poison': DataLoader(dataloader['dev-poison'].dataset, batch_size=train_dataloader.batch_size, collate_fn=train_dataloader.collate_fn, shuffle=False)
            }
            self.staticOneHotLabels = {name:self.getoneHotLabels(loader) for name, loader in self.staticDataLoaders.items()}
            
            self.kernels = {name:self.getKernel(loader) for name, loader in self.staticDataLoaders.items()}
            
            self.lowNorm, self.highNorm = {name:[[] for _ in range(self.frequencyConfig['kernelNum'])] for name in self.staticDataLoaders.keys()}, {name:[[] for _ in range(self.frequencyConfig['kernelNum'])] for name in self.staticDataLoaders.keys()}
            self.lowDeviation, self.highDeviation = {name:[[] for _ in range(self.frequencyConfig['kernelNum'])] for name in self.staticDataLoaders.keys()}, {name:[[] for _ in range(self.frequencyConfig['kernelNum'])] for name in self.staticDataLoaders.keys()}
            self.labelFreqLow, self.labelFreqHigh = {}, {}
            for name in self.staticDataLoaders.keys():
                labelLow, labelHigh = getHighDimFreq(self.staticOneHotLabels[name], self.kernels[name])
                self.labelFreqLow[name] = labelLow
                self.labelFreqHigh[name] = labelHigh
        else:
            logger.info("Disable Frequency Analysis")

    def train_one_epoch(self, epoch: int, epoch_iterator):
        """
        Train one epoch function.

        Args:
            epoch (:obj:`int`): current epoch.
            epoch_iterator (:obj:`torch.utils.data.DataLoader`): dataloader for training.
        
        Returns:
            :obj:`float`: average loss of the epoch.
        """
        self.model.train()
        total_loss = 0
        poison_loss_list, normal_loss_list = [], []
        for step, batch in enumerate(epoch_iterator):
            batch_inputs, batch_labels = self.model.process(batch)
            output = self.modelEngine(batch_inputs)
            logits = output.logits
            loss = self.loss_function.forward(logits, batch_labels)
            
            if self.visualize:
                poison_labels = batch["poison_label"]
                for l, poison_label in zip(loss, poison_labels):
                    if poison_label == 1:
                        poison_loss_list.append(l.item())
                    else:
                        normal_loss_list.append(l.item())
                loss = loss.mean()
                
            if self.gradient_accumulation_steps > 1:
                loss = loss / self.gradient_accumulation_steps
            
            self.modelEngine.backward(loss)


            self.modelEngine.step()

            torch.cuda.empty_cache()
            if self.frequencyConfig['frequencyVis'] and epoch < self.frequencyConfig['freqVisEpoch'] and (step + 1) % self.frequencyConfig['computeFrequencyStep'] == 0:
                with torch.no_grad():
                    self.saveFrequencyState()

        avg_loss = total_loss / len(epoch_iterator)
        avg_poison_loss = sum(poison_loss_list) / len(poison_loss_list) if self.visualize else 0
        avg_normal_loss = sum(normal_loss_list) / len(normal_loss_list) if self.visualize else 0
        
        return avg_loss, avg_poison_loss, avg_normal_loss
    