import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks.lr_monitor import LearningRateMonitor
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import DeviceStatsMonitor, ModelCheckpoint

import argparse
from omegaconf import OmegaConf

from sharelock.data.data import DataModule
from sharelock.models.model import ShareLock

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train ShareLock model")
    parser.add_argument("--config", type=str, default="configs/default_config.yaml", help="Path to config file")
    parser.add_argument("--eval_only", action="store_true", help="Whether to only evaluate the model on the test dataset")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint to load (.ckpt)")
    args, unknown_args = parser.parse_known_args()

    # Load hyperparameters and checkpoint (if provided)
    config = OmegaConf.load(args.config)
    if args.checkpoint is not None:
        checkpoint = torch.load(args.checkpoint)
        config = OmegaConf.merge(config, checkpoint["hyper_parameters"])
    cli_config = OmegaConf.from_cli()
    config = OmegaConf.merge(config, cli_config)
    
    # Seeding
    pl.seed_everything(config.seed, workers=True)
    
    # Initialize data module
    print("Loading data")
    data_module = DataModule(config)
    
    # Initialize model
    print("Loading model")
    if args.checkpoint is not None:
        model = ShareLock.load_from_checkpoint(args.checkpoint, config=config)
    else:
        model = ShareLock(config)
    
    # Initialize callbacks
    callbacks = []
    checkpointing = ModelCheckpoint(
        filename="best_model",
        save_top_k=1,
        monitor="validation_loss",
        mode="min",
    )
    callbacks.append(checkpointing)
    if config.training.early_stopping:
        callbacks.append(EarlyStopping(
            monitor="validation_loss",
            patience=config.training.early_stopping_patience,
            min_delta=0.1,
            mode="min",
        ))
    
    # Set up logging for evaluation and results
    logger = pl.loggers.TensorBoardLogger(save_dir=config.logging.save_dir, name=config.experiment_name)
    callbacks.append(LearningRateMonitor(logging_interval="step"))
    
    # Initialize trainer
    print("Loading trainer")
    trainer = pl.Trainer(
        logger=logger,
        max_steps=config.training.max_steps,
        log_every_n_steps=config.logging.log_every_n_steps,
        val_check_interval=config.logging.val_check_interval,
        check_val_every_n_epoch=None,
        callbacks=callbacks,
        gradient_clip_val=config.training.max_grad_norm,
        accumulate_grad_batches=config.training.accumulate_grad_batches,
        )
    
    if args.eval_only:
        # Load the best model from the checkpoint
        assert args.checkpoint is not None, "Checkpoint must be provided for evaluation"
    else:
        # Train the model
        trainer.fit(model, data_module)
        
        model = ShareLock.load_from_checkpoint(checkpointing.best_model_path, config=config)
    
    # Evaluate the model
    trainer.test(model, data_module)
    
    