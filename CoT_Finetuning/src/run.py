from pytorch_lightning.accelerators import accelerator
from pytorch_lightning.loggers import WandbLogger
import pytorch_lightning as pl

import argparse
from argparse import ArgumentParser
import os
import json

import sys
sys.path.append('../')
from evaluation import evaluate
from T5 import T5_small
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
import numpy as np
import torch
import random
import nltk


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--config', default=None, type=str)
    parser.add_argument('--checkpoint_path',type=str,default='') # In case of evaluation, for training it should be empty
    
    arg_ = parser.parse_args()
    if arg_.config == None:
        raise NameError("Include a config file in the argument please.")

    #Getting configurations
    with open(arg_.config) as config_file:
        hparam = json.load(config_file)
    hparam = argparse.Namespace(**hparam)

    #Setting GPUs to use
    # os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    # os.environ["CUDA_VISIBLE_DEVICES"]=hparam.CUDA_VISIBLE_DEVICES

    if 'weight_decay' not in hparam:
        hparam.weight_decay = 0.0
    if 'grad_norm' not in hparam:
        hparam.grad_norm = 0.8
    if 'output_log' not in hparam:
        hparam.output_log = f'log/{hparam.wandb_run_name}.csv'
    if 'learning_rate' not in hparam:
        hparam.learning_rate = None
    if 'gradient_accumulation_steps' not in hparam:
        hparam.gradient_accumulation_steps = 1
    if 'num_train_epochs' not in hparam:
        hparam.num_train_epochs = 0
    if 'use_lr_scheduling' not in hparam:
        hparam.use_lr_scheduling = False
    if 'num_workers' not in hparam:
        hparam.num_workers = 0
    if 'output_dir' not in hparam:
        hparam.output_dir = None
    if 'data_dir' not in hparam:
        hparam.data_dir = "../../data_extraction/data"
    if 'wandb_log' not in hparam:
        hparam.wandb_log = False
    if 'accelerator' not in hparam:
        hparam.accelerator = "deepspeed_stage_2"
    if 'max_steps' not in hparam:
        hparam.max_steps = None
    if 'checkpoint_path' not in hparam:
        hparam.checkpoint_path =''
    if 'method' not in hparam: 
        hparam.method = 'mixed'
    if 'eval_with_prob' not in hparam:
        hparam.eval_with_prob = False
    if 'eval_with_rouge' not in hparam:
        hparam.eval_with_rouge = False
    if 'eos_token' not in hparam:
        hparam.eos_token = True
    if arg_.checkpoint_path == '':
        print("no checkpoint loaded!!!")

    if hparam.wandb_log:
        wandb_logger = WandbLogger(project=hparam.wandb_project, name=hparam.wandb_run_name,entity="lklab_kaist")
        wandb_logger.log_hyperparams(hparam)
    else:
        wandb_logger = None
    nltk.download('punkt')
    #Setting configurations
    args_dict = dict(
        train_data=hparam.train_data,
        eval_data=hparam.eval_data,
        max_input_length=hparam.input_length,
        max_output_length=hparam.output_length,
        num_train_epochs=hparam.num_train_epochs,
        method=hparam.method,
        train_batch_size=hparam.train_batch_size,
        eval_batch_size=hparam.eval_batch_size,
        n_gpu=hparam.ngpu,
        model_name_or_path=hparam.model,
        output_log=hparam.output_log,
        mode=hparam.mode,
        output_dir=hparam.output_dir,
        data_dir=hparam.data_dir,
        weight_decay=hparam.weight_decay,
        learning_rate=hparam.learning_rate,
        gradient_accumulation_steps=hparam.gradient_accumulation_steps,
        max_grad_norm=hparam.grad_norm,
        use_lr_scheduling=hparam.use_lr_scheduling,
        num_workers=hparam.num_workers,
        accelerator=hparam.accelerator,
        max_steps=hparam.max_steps,
        checkpoint_path=arg_.checkpoint_path,
        opt_level='O1',
        eval_with_prob=hparam.eval_with_prob,
        eval_with_rouge=hparam.eval_with_rouge,
        eos_token=hparam.eos_token,
        wandb_run_name = hparam.wandb_run_name,
        peft_method = hparam.peft_method
    )
    args = argparse.Namespace(**args_dict)
    checkpoint_callback = False # Do not save model checkpoints when output dir is empty
    callbacks=[]     

    # Logging Learning Rate Scheduling
    if args.use_lr_scheduling and hparam.wandb_log:
        callbacks.append(pl.callbacks.LearningRateMonitor())

    set_seed(42)

    tokenizer = None
    if args.checkpoint_path!="":
        print(args.checkpoint_path)
        t5_3b = torch.load(args.checkpoint_path)
        model = T5_small(args)
        t0_new={}
        for key, value in t5_3b.items():
            t0_new['model.'+key] = value
        model.load_state_dict(t0_new, strict=False)
    else:
        model = T5_small(args)

    if (args.mode == 'evaluate') or (args.mode == 'rationale_evaluate'):
        import time
        start = time.time()
        evaluate(args, model, tokenizer) 
        end = time.time()
        print(f'Time: {end-start}')
    elif (args.mode == 'finetune') or (args.mode == 'rationale_tune'):
        train_params = dict(
            accumulate_grad_batches=args.gradient_accumulation_steps,
            accelerator='gpu',
            devices=args.n_gpu,
            max_epochs=args.num_train_epochs,
            gradient_clip_val=args.max_grad_norm,
            precision = "bf16",
            amp_backend="native",
            enable_checkpointing=checkpoint_callback,
            val_check_interval= 2500,
            logger=wandb_logger,
            callbacks = callbacks,
            strategy=args.accelerator,
        )
        trainer= pl.Trainer(**train_params)
        import time
        start = time.time()
        trainer.fit(model)
        end = time.time()
        print(f'Time: {end-start}')