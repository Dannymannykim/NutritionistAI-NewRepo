from ultralytics import YOLO
import torch
from pathlib import Path
import yaml
import pandas as pd
from collections import Counter
import random
from sklearn.model_selection import KFold
import shutil
from tqdm import tqdm
import datetime
import os
import glob
from ray import tune
from ray.tune.search.optuna import OptunaSearch
import yaml
#import wandb
#from wandb.integration.ultralytics import add_wandb_callback
from ultralytics import settings, cfg
from utils import compile_args

# monkey-patch tuner (for ray tune optuna)
import ultralytics.utils.tuner as tuner_mod
from ultralytics.cfg import TASK2DATA, TASK2METRIC, get_cfg, get_save_dir
from ultralytics.utils import DEFAULT_CFG, DEFAULT_CFG_DICT, LOGGER, NUM_THREADS, checks, colorstr
from ultralytics.utils.callbacks import raytune as rt

# monkey-patch trainer (for defining validation frequency during training)
import ultralytics.engine.trainer as trainer_mod
import time
import warnings
import math
import numpy as np
import torch
from torch import distributed as dist
from ultralytics.utils.torch_utils import (
    TORCH_2_4,
    EarlyStopping,
    ModelEMA,
    autocast,
    convert_optimizer_state_dict_to_fp16,
    init_seeds,
    one_cycle,
    select_device,
    strip_optimizer,
    torch_distributed_zero_first,
    unset_deterministic,
)
from ultralytics.utils import (
    DEFAULT_CFG,
    LOCAL_RANK,
    LOGGER,
    RANK,
    TQDM,
    YAML,
    callbacks,
    clean_url,
    colorstr,
    emojis,
)

def ultralytic_tune(
      model, 
      data, 
      iterations=10,
      use_ray=False, 
      # grace_period=None, this is only for ray integration
      project="proj",
      name=None,
      search_space=None,
      device='cuda',
      **train_args
):
   """
   Runs hyperparameter tuning on a YOLO model using Ultralytics' built-in or Ray Tune engine.

   Parameters
   ----------
   model : ultralytics.engine.model.Model
      The YOLO model instance (e.g., YOLO("yolov8s-seg.pt")).
   data : str or dict
      Path to dataset YAML file or dictionary containing dataset info.
   iterations : int, optional
      Number of trials to run (default is 10).
   use_ray : bool, optional
      If True, use Ray Tune for distributed HPO (default is False).
   grace_period : int, optional
   project : str, optional
      Top-level directory for logs and model runs (default is None).
   search_space : dict, optional
      Dictionary defining hyperparameter ranges to search (default is None).
      For ray tune, tune.uniform() will be automatically applied.
   name : str, optional
      Subdirectory name under `project/` for the tuning run (default is None).
   space : dict
      Search space for tuning.
   device : str, optional
      Device to train on, e.g., "cuda", "cpu", or "cuda:0" (default is 'cuda').

   Returns
   -------
   None
   """
   args = dict(
      data=data,
      iterations=iterations,
      use_ray=use_ray,
      project=project,
      space=search_space,
      name=name,
      device=device,
      **(train_args or {})
   )
      
   #if use_ray:
   #   args['grace_period'] = grace_period
   #   args['space'] = search_space#{key: tune.uniform(val[0], val[1]) for key, val in search_space.items()}
   
   model.tune(task='segment', **args)

def ultralytic_train(
      model,
      data,
      epochs=50,
      patience=100,
      batch=16,
      imgsz=640,
      save=True,
      project=None,
      name=None,
      exist_ok=False,
      seed=0,
      classes=None,
      rect=False,
      multi_scale=False,
      cos_lr=False,
      fraction=1.0,
      freeze=11,
      overlap_mask=True,
      cache=False,
      val=True,
      plots=False,
      device='cuda',
      **hyperparams,
):
   """
   Runs training on a YOLO model using Ultralytics' built-in.

   Parameters
   ----------
   model : ultralytics.engine.model.Model
      The YOLO model instance (e.g., YOLO("yolov8s-seg.pt")).
   data : str or dict
      Path to dataset YAML file or dictionary containing dataset info.
   epochs : int, optional
      Maximum number of epochs per trial (default is 50).
   patience : int, optional
      Number of epochs to wait for improvement before early stopping (default is 100).
   fraction : float, optional
      Fraction of training data to use in each trial (default is 1.0).
   plots : bool, optional
      If True, generate plots for each trial (default is True).
   save : bool, optional
      If True, save the best model from each trial (default is True).
   val : bool, optional
      If True, run validation at the end of each epoch (default is True).
   imgsz : int or tuple, optional
      Image size used during training (default is 640).
   device : str, optional
      Device to train on, e.g., "cuda", "cpu", or "cuda:0" (default is 'cuda').
   freeze : int, optional
      Number of backbone layers to freeze (default is 11).
   overlap_mask : bool, optional
      Determines whether object masks should be merged into a single mask for training, 
      or kept separate for each object (default is True). In case of overlap, the smaller 
      mask is overlaid on top of the larger mask during merge. False for instance segmentaiton.
   project : str, optional
      Top-level directory for logs and model runs (default is None).
   name : str, optional
      Subdirectory name under `project/` for the tuning run (default is None).
   exist_ok : bool, optional
      If True, existing project/name folder can be overwritten (default is False).
   classes : list[int], optional
      Subset of class indices to use for training (default is None).

   Returns
   -------
   None
   """

   model.train(
      data=data,
      epochs=epochs,
      task='segment',
      patience=patience,
      batch=batch,
      imgsz=imgsz,
      save=save,
      project=project,
      name=name,
      exist_ok=exist_ok,
      seed=seed,
      classes=classes,
      rect=rect,
      multi_scale=multi_scale,
      cos_lr=cos_lr,
      fraction=fraction,
      freeze=freeze,
      overlap_mask=overlap_mask,
      cache=cache,
      val=val,
      plots=plots,
      device=device,
      **(hyperparams or {})
   )
   
def run_train(model, data, args, hyperparams, use_sweep=False, project=None, name=None):
   project = project if project else "runs/segment/trains"
   wandb_project = project.replace("/", "-")

   last = project.split('/')[-1]
   mapping = {
      "trains": "train",
      "tunes": "tune",
      "tunes_sweep": "tune",
      "tunes_ray": "tune"
   }
   name = name if name else mapping[last]

   if use_sweep: # Tuning with WandB sweep uses train method with hyperparams set in sweep.yaml
      wandb.init(project=wandb_project)
      hyperparams = wandb.config
      model = YOLO(model.model_name)
      # Add WandB callback for logging
      add_wandb_callback(model)

   ultralytic_train(model, data, project=project, name=name, **{**args, **hyperparams})

def run_tune(model, data, args, hyperparams, project=None, name=None):
   """
   Tune Integrations:
      - WandB Sweep
         - Creates project directory 'runs/segment/tunes_sweep'
      - Ultralytic's tune w/ ray tune integration
         - optuna, bayes, random gridsearch, etc
         - Creates project directory 'runs/segment/tunes_ray'
      - Ultralytic's tune w/o ray tune integration
         - genetic algorithm
         - Creates project directory 'runs/segment/tunes'
   
   Parameters
   ----------
   data (str) : .yaml file with the path to training and validation set.
   args (dict) : arguments required for tuning, as well as training. 
   hyperparams (dict) : separate arguments dict for hyperparameters.
   name (str) : name of fold; this is only for cross-validation.

   Returns
   -------
   None
   """
   use_sweep = args.pop('use_sweep', False) # pop since this isn't an actual arg
   use_ray = args.get('use_ray', False)
   search_space = args['search_space']

   if use_sweep and use_ray:
      raise ValueError("Sweep and Ray can't both be true. Set either one to False.")
   elif use_ray and search_space:
      args['search_space'] = {
         key: (
            tune.loguniform(val[0], val[1]) if key == 'lr0' 
            else tune.uniform(val[0], val[1]) if key != 'freeze'
            else tune.choice(val)  # use tune.choice for 'freeze'
         )
         for key, val in args['search_space'].items()
      }
      if 'freeze' in args['search_space']: # if freeze is set in train_args, search_space fails to overwrite
         args.pop('freeze')

   else: # grace_period and use_wandb (for monkey-patch) is only an argument for ray
      args.pop('grace_period')
      args.pop('use_wandb')
      args.pop('use_ray')
   
   project = project if project else f"runs/segment/tunes{('_sweep' if use_sweep else '_ray' if use_ray else '')}"
   name = name if name else "tune"
   
   if use_sweep:

      settings.update({'wandb': True}) # very important; wandb will not plot charts w/ proper values otherwise
      iterations = args.pop('iterations') # remove iterations from args since WandB uses it in call, not the train method
      args.pop('search_space')

      with open("sweep.yaml") as f:
         sweep_yaml = yaml.safe_load(f)

      wandb_project = project.replace("/", "-")
      sweep_id = wandb.sweep(sweep=sweep_yaml, project=wandb_project) # creates sweep
      wandb.agent(sweep_id, function=lambda: run_train(
            model, 
            data=data,
            args=args,
            hyperparams=hyperparams, # this isn't exactly used since wandb uses its own sweep.yaml configs 
            use_sweep=True,
            project=project, 
            name=name,
         ), 
         count=iterations
      ) # runs agent on sweep

   else:
      ultralytic_tune(model, data, project=project, name=name, **args)

def run_process(
      model_pth,
      cv_path, 
      train_args, 
      tune_args, 
      hyperparameters, 
      process='train', 
      cross_val=False,
      project=None
   ): # check if setting configs_train and tune = None still works
   """
   cv_path: path to premade cv fold dataset. if cross_val=False, uses default full dataset in ./datasets/full.
   train_args: all arguments pertaining to training process.
   tune_args: all arguments pertaining to tuning process.
   hyperparameters: model hyperparameters and data augmentation params.
   project: directory for logs and model runs.
   process: process to run ("train" or "tune"; default is "train")
   cross_val: boolean to indicate whether to run cross-validation.
   """
   run_fn = run_train if process == 'train' else run_tune
   
   if process == 'train':
      tune_args = {}

   use_wandb = train_args.get('use_wandb', False)
   
   if use_wandb and not tune_args.get('use_ray', False): # the 2nd condition is to prevent wandb from running twice (logging from both ultralytics and ray integration)
      settings.update({'wandb': True})
   else:
      settings.update({'wandb': False})

   if process == 'train': # only for train bc need to pass "use_wandb" arg to monkey-patched ray integration fn
      train_args.pop('use_wandb', False)

   full_args = {**train_args, **tune_args}
   
   # Set up and run tuning w/o cross validation
   if not cross_val:
      model = YOLO(model_pth)
      dataset_config_yaml = "D:/Work/ML_projects/food_classifier/datasets/full/coco_seg.yaml"
      
      run_fn(model, data=dataset_config_yaml, args=full_args, hyperparams=hyperparameters, project=project)
      #print(model.names)
      return
   
   # Set up and run tuning w/ cross validation
   ds_yamls = list(cv_path.rglob("*.yaml"))
   results = {}
   i = 1
   
   while glob.glob(f"runs/segment/{process}s/cv_{process}{i}*"):
      i += 1   
   name = f"cv_{process}{str(i)}"
   
   for k, dataset_yaml in enumerate(ds_yamls):
      model = YOLO(model_pth) # for training we may want to use one model
      
      results[k] = run_fn(model, str(dataset_yaml), args=full_args, hyperparams=hyperparameters, project=project, name=f"{name}_fold{k+1}")
     
   return results
   
def custom_ray_tune(
      model,
      space: dict = None,
      grace_period: int = 10,
      gpu_per_trial: int = None,
      max_samples: int = 10,
      **train_args,
):

   if train_args is None:
      train_args = {}

   try:
      checks.check_requirements("ray[tune]")

      import ray
      from ray import tune
      from ray.air import RunConfig
      from ray.air.integrations.wandb import WandbLoggerCallback
      from ray.tune.schedulers import ASHAScheduler
   except ImportError:
      raise ModuleNotFoundError('Ray Tune required but not found. To install run: pip install "ray[tune]"')

   use_wandb = train_args.pop("use_wandb", False)
   use_optuna = train_args.pop("optuna", False)

   try:
      import wandb

      assert hasattr(wandb, "__version__")
   except (ImportError, AssertionError):
      use_wandb = False
   
   checks.check_version(ray.__version__, ">=2.0.0", "ray")
   default_space = {
      # 'optimizer': tune.choice(['SGD', 'Adam', 'AdamW', 'NAdam', 'RAdam', 'RMSProp']),
      "lr0": tune.uniform(1e-5, 1e-1),
      "lrf": tune.uniform(0.01, 1.0),  # final OneCycleLR learning rate (lr0 * lrf)
      "momentum": tune.uniform(0.6, 0.98),  # SGD momentum/Adam beta1
      "weight_decay": tune.uniform(0.0, 0.001),  # optimizer weight decay
      "warmup_epochs": tune.uniform(0.0, 5.0),  # warmup epochs (fractions ok)
      "warmup_momentum": tune.uniform(0.0, 0.95),  # warmup initial momentum
      "box": tune.uniform(0.02, 0.2),  # box loss gain
      "cls": tune.uniform(0.2, 4.0),  # cls loss gain (scale with pixels)
      "hsv_h": tune.uniform(0.0, 0.1),  # image HSV-Hue augmentation (fraction)
      "hsv_s": tune.uniform(0.0, 0.9),  # image HSV-Saturation augmentation (fraction)
      "hsv_v": tune.uniform(0.0, 0.9),  # image HSV-Value augmentation (fraction)
      "degrees": tune.uniform(0.0, 45.0),  # image rotation (+/- deg)
      "translate": tune.uniform(0.0, 0.9),  # image translation (+/- fraction)
      "scale": tune.uniform(0.0, 0.9),  # image scale (+/- gain)
      "shear": tune.uniform(0.0, 10.0),  # image shear (+/- deg)
      "perspective": tune.uniform(0.0, 0.001),  # image perspective (+/- fraction), range 0-0.001
      "flipud": tune.uniform(0.0, 1.0),  # image flip up-down (probability)
      "fliplr": tune.uniform(0.0, 1.0),  # image flip left-right (probability)
      "bgr": tune.uniform(0.0, 1.0),  # image channel BGR (probability)
      "mosaic": tune.uniform(0.0, 1.0),  # image mosaic (probability)
      "mixup": tune.uniform(0.0, 1.0),  # image mixup (probability)
      "cutmix": tune.uniform(0.0, 1.0),  # image cutmix (probability)
      "copy_paste": tune.uniform(0.0, 1.0),  # segment copy-paste (probability)
   }

   # Put the model in ray store
   task = model.task
   model_in_store = ray.put(model)

   def _tune(config):
      """Train the YOLO model with the specified hyperparameters and return results."""
      model_to_train = ray.get(model_in_store)  # get the model from ray store for tuning
      model_to_train.reset_callbacks()
      config.update(train_args)
      results = model_to_train.train(**config)
      return results.results_dict

   # Get search space
   if not space and not train_args.get("resume"):
      space = default_space
      LOGGER.warning("Search space not provided, using default search space.")

   # Get dataset
   data = train_args.get("data", TASK2DATA[task])
   space["data"] = data
   if "data" not in train_args:
      LOGGER.warning(f'Data not provided, using default "data={data}".')

   # Define the trainable function with allocated resources
   trainable_with_resources = tune.with_resources(_tune, {"cpu": NUM_THREADS, "gpu": gpu_per_trial or 1}) # changes 0 to 1; seems to make no change

   # Define the ASHA scheduler for hyperparameter search
   asha_scheduler = ASHAScheduler(
      time_attr="epoch",
      metric=TASK2METRIC[task],
      mode="max",
      max_t=train_args.get("epochs") or DEFAULT_CFG_DICT["epochs"] or 100,
      grace_period=grace_period,
      reduction_factor=3,
   )

   # Define the callbacks for the hyperparameter search
   tuner_callbacks = [WandbLoggerCallback(project="runs-segment-tunes_ray")] if use_wandb else [] # may want to specify wandb_project

   # Create the Ray Tune hyperparameter search tuner
   tune_dir = get_save_dir(
      get_cfg(
         DEFAULT_CFG,
         {**train_args, **{"exist_ok": train_args.pop("resume", False)}},  # resume w/ same tune_dir
      ),
      name=train_args.pop("name", "tune"),  # runs/{task}/{tune_dir}
   ).resolve()  # must be absolute dir
   tune_dir.mkdir(parents=True, exist_ok=True)
   if tune.Tuner.can_restore(tune_dir):
      LOGGER.info(f"{colorstr('Tuner: ')} Resuming tuning run {tune_dir}...")
      tuner = tune.Tuner.restore(str(tune_dir), trainable=trainable_with_resources, resume_errored=True)
   else:
      optuna_search = OptunaSearch(
         space,
         metric="metrics/mAP50-95(M)",#["val/dfl_loss", "val/seg_loss", "val/cls_loss", "val/box_loss"],
         mode="max"#["min", "min", "min", "min"]
      )
      if use_optuna:
         tuner = tune.Tuner(
            trainable_with_resources,
            #param_space=space,
            tune_config=tune.TuneConfig(
                  scheduler=asha_scheduler,
                  num_samples=max_samples,
                  trial_name_creator=lambda trial: f"{trial.trainable_name}_{trial.trial_id}",
                  trial_dirname_creator=lambda trial: f"{trial.trainable_name}_{trial.trial_id}",
                  search_alg=optuna_search
            ),
            run_config=RunConfig(callbacks=tuner_callbacks, storage_path=tune_dir.parent, name=tune_dir.name),
         )
      else:
         tuner = tune.Tuner(
            trainable_with_resources,
            param_space=space,
            tune_config=tune.TuneConfig(
                  scheduler=asha_scheduler,
                  num_samples=max_samples,
                  trial_name_creator=lambda trial: f"{trial.trainable_name}_{trial.trial_id}",
                  trial_dirname_creator=lambda trial: f"{trial.trainable_name}_{trial.trial_id}",
            ),
            run_config=RunConfig(callbacks=tuner_callbacks, storage_path=tune_dir.parent, name=tune_dir.name),
         )

   # Run the hyperparameter search
   tuner.fit()

   # Get the results of the hyperparameter search
   results = tuner.get_results()

   # Shut down Ray to clean up workers
   ray.shutdown()

   return results

def custom_do_train(self, world_size=1):
   """Train the model with the specified world size."""
   if world_size > 1:
      self._setup_ddp(world_size)
   self._setup_train(world_size)

   nb = len(self.train_loader)  # number of batches
   nw = max(round(self.args.warmup_epochs * nb), 100) if self.args.warmup_epochs > 0 else -1  # warmup iterations
   last_opt_step = -1
   self.epoch_time = None
   self.epoch_time_start = time.time()
   self.train_time_start = time.time()
   self.run_callbacks("on_train_start")
   LOGGER.info(
      f"Image sizes {self.args.imgsz} train, {self.args.imgsz} val\n"
      f"Using {self.train_loader.num_workers * (world_size or 1)} dataloader workers\n"
      f"Logging results to {colorstr('bold', self.save_dir)}\n"
      f"Starting training for " + (f"{self.args.time} hours..." if self.args.time else f"{self.epochs} epochs...")
   )
   if self.args.close_mosaic:
      base_idx = (self.epochs - self.args.close_mosaic) * nb
      self.plot_idx.extend([base_idx, base_idx + 1, base_idx + 2])
   epoch = self.start_epoch
   self.optimizer.zero_grad()  # zero any resumed gradients to ensure stability on train start
   while True:
      self.epoch = epoch
      self.run_callbacks("on_train_epoch_start")
      with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # suppress 'Detected lr_scheduler.step() before optimizer.step()'
            self.scheduler.step()

      self._model_train()
      if RANK != -1:
            self.train_loader.sampler.set_epoch(epoch)
      pbar = enumerate(self.train_loader)
      # Update dataloader attributes (optional)
      if epoch == (self.epochs - self.args.close_mosaic):
            self._close_dataloader_mosaic()
            self.train_loader.reset()

      if RANK in {-1, 0}:
            LOGGER.info(self.progress_string())
            pbar = TQDM(enumerate(self.train_loader), total=nb)
      self.tloss = None
      for i, batch in pbar:
            self.run_callbacks("on_train_batch_start")
            # Warmup
            ni = i + nb * epoch
            if ni <= nw:
               xi = [0, nw]  # x interp
               self.accumulate = max(1, int(np.interp(ni, xi, [1, self.args.nbs / self.batch_size]).round()))
               for j, x in enumerate(self.optimizer.param_groups):
                  # Bias lr falls from 0.1 to lr0, all other lrs rise from 0.0 to lr0
                  x["lr"] = np.interp(
                        ni, xi, [self.args.warmup_bias_lr if j == 0 else 0.0, x["initial_lr"] * self.lf(epoch)]
                  )
                  if "momentum" in x:
                        x["momentum"] = np.interp(ni, xi, [self.args.warmup_momentum, self.args.momentum])

            # Forward
            with autocast(self.amp):
               batch = self.preprocess_batch(batch)
               loss, self.loss_items = self.model(batch)
               self.loss = loss.sum()
               if RANK != -1:
                  self.loss *= world_size
               self.tloss = (
                  (self.tloss * i + self.loss_items) / (i + 1) if self.tloss is not None else self.loss_items
               )

            # Backward
            self.scaler.scale(self.loss).backward()

            # Optimize - https://pytorch.org/docs/master/notes/amp_examples.html
            if ni - last_opt_step >= self.accumulate:
               self.optimizer_step()
               last_opt_step = ni

               # Timed stopping
               if self.args.time:
                  self.stop = (time.time() - self.train_time_start) > (self.args.time * 3600)
                  if RANK != -1:  # if DDP training
                        broadcast_list = [self.stop if RANK == 0 else None]
                        dist.broadcast_object_list(broadcast_list, 0)  # broadcast 'stop' to all ranks
                        self.stop = broadcast_list[0]
                  if self.stop:  # training time exceeded
                        break

            # Log
            if RANK in {-1, 0}:
               loss_length = self.tloss.shape[0] if len(self.tloss.shape) else 1
               pbar.set_description(
                  ("%11s" * 2 + "%11.4g" * (2 + loss_length))
                  % (
                        f"{epoch + 1}/{self.epochs}",
                        f"{self._get_memory():.3g}G",  # (GB) GPU memory util
                        *(self.tloss if loss_length > 1 else torch.unsqueeze(self.tloss, 0)),  # losses
                        batch["cls"].shape[0],  # batch size, i.e. 8
                        batch["img"].shape[-1],  # imgsz, i.e 640
                  )
               )
               self.run_callbacks("on_batch_end")
               if self.args.plots and ni in self.plot_idx:
                  self.plot_training_samples(batch, ni)

            self.run_callbacks("on_train_batch_end")

      self.lr = {f"lr/pg{ir}": x["lr"] for ir, x in enumerate(self.optimizer.param_groups)}  # for loggers
      self.run_callbacks("on_train_epoch_end")
      if RANK in {-1, 0}:
            final_epoch = epoch + 1 >= self.epochs
            self.ema.update_attr(self.model, include=["yaml", "nc", "args", "names", "stride", "class_weights"])

            # Validation
            if (self.args.val and (epoch + 1) % 2 == 0) or final_epoch or self.stopper.possible_stop or self.stop:#if self.args.val or final_epoch or self.stopper.possible_stop or self.stop:
               self._clear_memory(threshold=0.5)  # prevent VRAM spike
               self.metrics, self.fitness = self.validate()
            self.save_metrics(metrics={**self.label_loss_items(self.tloss), **self.metrics, **self.lr})
            self.stop |= self.stopper(epoch + 1, self.fitness) or final_epoch
            if self.args.time:
               self.stop |= (time.time() - self.train_time_start) > (self.args.time * 3600)

            # Save model
            if self.args.save or final_epoch:
               self.save_model()
               self.run_callbacks("on_model_save")

      # Scheduler
      t = time.time()
      self.epoch_time = t - self.epoch_time_start
      self.epoch_time_start = t
      if self.args.time:
            mean_epoch_time = (t - self.train_time_start) / (epoch - self.start_epoch + 1)
            self.epochs = self.args.epochs = math.ceil(self.args.time * 3600 / mean_epoch_time)
            self._setup_scheduler()
            self.scheduler.last_epoch = self.epoch  # do not move
            self.stop |= epoch >= self.epochs  # stop if exceeded epochs
      self.run_callbacks("on_fit_epoch_end")
      self._clear_memory(0.5)  # clear if memory utilization > 50%

      # Early Stopping
      if RANK != -1:  # if DDP training
            broadcast_list = [self.stop if RANK == 0 else None]
            dist.broadcast_object_list(broadcast_list, 0)  # broadcast 'stop' to all ranks
            self.stop = broadcast_list[0]
      if self.stop:
            break  # must break all DDP ranks
      epoch += 1

   if RANK in {-1, 0}:
      # Do final val with best.pt
      seconds = time.time() - self.train_time_start
      LOGGER.info(f"\n{epoch - self.start_epoch + 1} epochs completed in {seconds / 3600:.3f} hours.")
      self.final_eval()
      if self.args.plots:
            self.plot_metrics()
      self.run_callbacks("on_train_end")
   self._clear_memory()
   unset_deterministic()
   self.run_callbacks("teardown")

tuner_mod.run_ray_tune = custom_ray_tune
trainer_mod.BaseTrainer._do_train = custom_do_train

if __name__ == '__main__':
   # Idea is to classify food images -> connect to some api to calculate and display nutrient info -> feed into 
   # LLM for pros/cons
   # Big idea: perhaps have the LLM also take in the original classifications and try to correct certain suspicious classification
   # or ambiguities like butter/cheese.
   #import warnings
   #warnings.filterwarnings("ignore", category=UserWarning, module="wandb")
   
   # Note on annotation: combine overlapped objects for one bounding box

   # Let's get to know our "dream customers". Nutrition AIs are generally for health enthusiasts or gymers. 
   # I personally relate more with the gymers so that helps narrow down the number of classes (e.g. remove desserts, food items like pancakes, etc)
      # another reason to remove a class may be due to its similarity with another class; choose what's prio
      # for similar classes like yogurt/ranch consider having llm classify ranch or yogurt given the other classes.
         # ranch/yogurt next to cucumber most likely yogurt even if the items arent dessert.
   # Also remove raw ingredients like a whole onion, since that's unlikely to be a meal prep dish.

   # test if ingredients like potato do well bc currently we combine roasted potatoes with mashed.

   # estimate overlaps using bounding box. if they intersect blah blah

   # fried meat instead of fried chickne/cutlet/etc

   # Note: box scsore may not be the most accurate metric bc some ingredients are combined as one whole like clustered aspharagas

   # Do we want FPs or FNs?: We want to reduce FNs. FPs are fine bc many annotations dont always mask every parts of an ingredient so if
   # the model predicts a class for an unmasked area, the FP will be higher when it's actually correct.
   # 

   # context: rice + blueberry + sauce most likely peanutbutter = oatmeal

   # consider removing ingredients like spring onion.

   cv_path = Path("D:/Work/ML_projects/food_classifier/datasets/cv/cv_5_0-unstratified-visual")

   chosen_classes = [
      "cheese butter",
      "almond",
      "red beans",
      "cashew",
      "dried cranberries",
      "walnut",
      "peanut",
      "egg",
      "apple",
      "apricot",
      "avocado",
      "banana",
      "strawberry",
      "cherry",
      "blueberry",
      "raspberry",
      "mango",
      "olives",
      "peach",
      "lemon",
      "pear",
      "pineapple",
      "grape",
      "kiwi",
      "melon",
      "orange",
      "watermelon",
      "steak",
      "pork",
      "chicken duck",
      "sausage",
      "fried meat",
      "lamb",
      "sauce",
      "crab",
      "fish",
      "shellfish",
      "shrimp",
      "bread",
      "corn",
      "hamburg",
      "pizza",
      "wonton dumplings",
      "pasta",
      "rice",
      "pie",
      "tofu",
      "eggplant",
      "potato",
      "garlic",
      "cauliflower",
      "tomato",
      "seaweed",
      "spring onion",
      "ginger",
      "lettuce",
      "pumpkin",
      "cucumber",
      "white radish",
      "carrot",
      "asparagus",
      "bamboo shoots",
      "broccoli",
      "celery stick",
      "cilantro mint",
      "cabbage",
      "bean sprouts",
      "onion",
      "pepper",
      "green beans",
      "king oyster mushroom",
      "shiitake",
      "enoki mushroom",
      "oyster mushroom",
      "white button mushroom",
      "salad",
      "other ingredients"
   ]
   
   params1 = {
      "lr0": 0.07552710906155273,
      "batch": 16
   }

   params2 = {

   }
   
   search_space_freeze = {
      "lr0": (1e-5, 1e-1),          # Initial learning rate (log scale intended)
      "lrf": (0.01, 1.0),           # Final learning rate factor
      "freeze": [11],
      "momentum": (0.8, 0.95),      # Momentum
      #"weight_decay": (0.0, 0.001)
   }

   search_space = {
      "lr0": (1e-5, 1e-1),          # Initial learning rate (log scale intended)
      "lrf": (0.01, 1.0),           # Final learning rate factor
      "momentum": (0.8, 0.95),      # Momentum
      ##"weight_decay": (0.0, 0.001), # Weight decay (def: 0.0005)
      ##"warmup_epochs": (1.0, 5.0),  # Warmup epochs (def: 3)
      ##"warmup_momentum": (0.0, 0.95), # (def: 0.8)
      "box": (0.02, 0.2),           # Box loss weight
      "cls": (0.2, 4.0),            # Class loss weight
      "hsv_h": (0.0, 0.1),          # Hue augmentation range
      "translate": (0.0, 0.9),      # Translation augmentation range
      "degrees": (0.0, 45.0),       # Rotation
      "scale": (0.0, 0.9),          # Image scale
      # "shear": (0.0, 10.0),        # Optional: Shear (not recommended)
      # "perspective": (0.0, 0.001), # Optional: Perspective (not recommended)
      # "flipud": (0.0, 0.5),        # Flip upside down (not recommended)
      "fliplr": (0.0, 1.0),         # Flip left right
      "mosaic": (0.0, 1.0),        # Use of mosaic augmentation (be cautious)
      # "copy_paste": (0.0, 1.0),    # Copy-paste augmentation (be cautious)
   }

   train_args = {
      "epochs": 50,
      "batch": 8, # best batch size depends on model used; [obsolete] 32 seems to be best for GPU and CPU usage (2x faster than 16); 64 is too much and takes much longer 
      "optimizer": "auto",#"AdamW",
      "imgsz": 640, # 768 seems to error
      "freeze": None,#11,
      "fraction": 1.0,#1.0,
      "classes": None,
      "rect": False, # not compatible with wandb sweep due to resize error
      "multi_scale": False,
      "cos_lr": False, # False if low # of epochs; True may lead to better convergence over time.
      "overlap_mask": False, # False for instance seg prob
      "cache": "disk", # no noticeable change but "True" / "ram" may be faster
      "save": True,
      #"exist_ok": False,
      "val": True, # set True to perform valdiation every epoch; otherwise val is only done at end of training; wandb sweep requires val=True; UPDATE: val is done every 5 epochs
      #"cfg": "D:\\Work\\ML_Projects\\food_classifier\\custom_default.yaml", # required for adding 'val_interval' for monkey-patching validation frequency in training; otherwise, hard-code 'val_interval'
      #"val_interval": 5,
      "plots": True, # shows multiple plots for evaluation; provides various charts that are useful in wandb; test if this slows down process
      "seed": 0,
      #"workers": 8,
      "use_wandb": True,
      #half
   }

   tune_args = {
      "iterations": 20,
      "use_ray": True,
      "use_sweep": False,
      #"optuna": False, # only with ray
      "search_space": search_space_freeze,
      "grace_period": 10, # only with ray
   }

   if train_args['use_wandb']: # to suppress wandb-related warnings from imports
      import wandb
      from wandb.integration.ultralytics import add_wandb_callback

   model_pth = "./models/pretrained/yolo11m-seg.pt" # medium seems to do best
   process = 'train'
   cross_val = False

   external_args, train_args, tune_args = compile_args("./config.yaml")

   process = external_args['process']
   model_pth = external_args['model_pth'] # medium seems to do best
   cross_val = external_args['cross_val']

   if process == 'tune':
       

   full_args = {**train_args, **tune_args}
   
   full_args
   # Note: tuning takes both train and tune args
   run_process(model_pth, cv_path, train_args, tune_args, params2, process, cross_val) # for reg training perhaps add wandb callback

   run_process(
       process=process

   )
   # consider running val on training set
   #results = model.val(data='path/to/your/data.yaml', task='test') # add json_save; maybe split not task
   raise ImportError
   
   # train independently on model architecture sizes
   # tune on freeze layers (tune w/ parameters like lr, batch size)
   # tune on rest of hyperparameters

   sweep_name = "testing_sweep7"
   with open("sweep.yaml") as f:
      sweep_yaml = yaml.safe_load(f)
   
   sweep_id = wandb.sweep(sweep=sweep_yaml, project=sweep_name) # creates sweep
   wandb.agent(sweep_id, function=lambda: train2(sweep_name, train_args), count=2)
   raise ImportError
   wandb.agent(sweep_id, function=lambda: ultralytic_train(
            model="./models/pretrained/yolo11n-seg.pt", 
            data="D:/Work/ML_projects/food_classifier/datasets/full/coco_seg.yaml", 
            project="runs/segment/ignore", 
            name="tune_sweep",
            use_sweep=True, 
            wandb_project="testing_sweep7",
            **{**train_args, **tune_args, **params2}
         ), count=2) # runs agent on sweep
   raise ImportError
   model = YOLO("./models/pretrained/yolo11x-seg.pt")
   params = {
      "lr0": 0.0001,
      "batch": 8,
      "cls": 0.7
   }
   model.tune(data="D:/Work/ML_projects/food_classifier/datasets/full/coco_seg.yaml", 
              epochs=3, iterations=2, imgsz=640, use_ray=False, fraction=0.1, freeze=20, device='cuda', task='segment', plots=False, val=True, project="runs/segment/test_noray", **params)
   #train()
    #sweep_id = wandb.sweep(sweep=sweep_configuration, project="testing_sweep2")

    #wandb.agent(sweep_id, function=train, count=10)
    #print(torch.__version__)
    #print(torch.version.cuda)
    #print(torch.cuda.is_available())
    #torch.cuda.set_device(0)
    #wandb.login(key="db8fe96e97bd9914ccb3a065a1206ed944718e2e")

    #train()
    
   