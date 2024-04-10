import argparse, os

import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from tqdm.auto import tqdm

from dataset.celebv_hq import DataModule
from marlin_pytorch.config import resolve_config
from marlin_pytorch.util import read_yaml
from model.classifier import TD3MF
from util.earlystop_lr import EarlyStoppingLR
from util.lr_logger import LrLogger
from util.seed import Seed
from util.system_stats_logger import SystemStatsLogger

from config.grid_search_config import CONFIGURATIONS
import subprocess
import csv



def train(args, config):
    # dataset = args.dataset
    data_path = args.data_path
    resume_ckpt = args.resume
    n_gpus = args.n_gpus
    max_epochs = args.epochs

    finetune = config["finetune"]
    learning_rate = config["learning_rate"]
    task = config["task"]
    num_heads = config['num_heads']
    ir_layers = config['ir_layers']
    temporal_axis = config['temporal_axis']
    audio_pe = config['audio_positional_encoding']
    fusion = config['fusion']
    hidden_layers = config['hidden_layers']
    lp_only = config['lp_only']
    audio_backbone = config['audio_backbone']
    middle_fusion_type = config['middle_fusion_type']
    training_datasets = config['training_datasets']
    eval_datasets = config['eval_datasets']

    available_datasets = ["DeepfakeTIMIT", "DFDC", "FakeAVCeleb", "Forensics++", "RAVDESS"]
    for dataset in training_datasets + eval_datasets:
        if dataset not in available_datasets:
            raise ValueError(f"Dataset {dataset} not in {available_datasets}")

    assert task == "deepfake", "Multi class task currently not implemented."

    if task == "appearance":
        num_classes = 40
    elif task == "action":
        num_classes = 35
    elif task == "deepfake": 
        num_classes = 1  
    else:
        raise ValueError(f"Unknown task {task}")

    if finetune:
        backbone_config = resolve_config(config["backbone"])

        model = TD3MF(
            num_classes, config["backbone"], True, args.marlin_ckpt,
            "binary", learning_rate,
            args.n_gpus > 1, ir_layers, num_heads, temporal_axis=temporal_axis,
            audio_pe=audio_pe, fusion=fusion, hidden_layers=hidden_layers,
            lp_only=lp_only, audio_backbone=audio_backbone, middle_fusion_type=middle_fusion_type
        )

        dm = DataModule(
            data_path, finetune, task,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            clip_frames=backbone_config.n_frames,
            temporal_sample_rate=2, temporal_axis=temporal_axis,
            audio_feature=audio_backbone,
            training_datasets=training_datasets,
            eval_datasets=eval_datasets
        )

    else:
        model = TD3MF(
            num_classes, config["backbone"], False,
            None, "binary", learning_rate, args.n_gpus > 1,
            ir_layers, num_heads, temporal_axis=temporal_axis,
            audio_pe=audio_pe, fusion=fusion, hidden_layers=hidden_layers,
            lp_only=lp_only, audio_backbone=audio_backbone, middle_fusion_type=middle_fusion_type
        )

        dm = DataModule(
            data_path, finetune, task,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            feature_dir=config["backbone"],
            temporal_reduction=config["temporal_reduction"],
            temporal_axis=temporal_axis,
            audio_feature=audio_backbone,
            training_datasets=training_datasets,
            eval_datasets=eval_datasets
        )

    if args.skip_train:
        dm.setup()
        return resume_ckpt, dm

    strategy = None if n_gpus <= 1 else "ddp"
    accelerator = "cpu" if n_gpus == 0 else "gpu"

    ckpt_filename = config["model_name"] + "-{epoch}-{val_auc:.3f}"
    ckpt_monitor = "val_auc"

    try:
        precision = int(args.precision)
    except ValueError:
        precision = args.precision

    print(f"ckpt/{config['model_name']}")
    ckpt_callback = ModelCheckpoint(dirpath=f"ckpt/{config['model_name']}", save_last=True,
                                    filename=ckpt_filename,
                                    monitor=ckpt_monitor,
                                    mode="max")  # ,
    # save_top_k=-1)

    trainer = Trainer(log_every_n_steps=1, devices=n_gpus, accelerator=accelerator, benchmark=True,
                      logger=True, precision=precision, max_epochs=max_epochs,
                      strategy=strategy, resume_from_checkpoint=resume_ckpt,
                      callbacks=[ckpt_callback, LrLogger(), EarlyStoppingLR(1e-6), SystemStatsLogger()])

    trainer.fit(model, dm)

    print(f"\n\nSaving best model at: {ckpt_callback.best_model_path}")
    return ckpt_callback.best_model_path, dm


def eval_dataset(args, ckpt, dm):
    print("Load checkpoint", ckpt)
    model = TD3MF.load_from_checkpoint(ckpt)
    accelerator = "cpu" if args.n_gpus == 0 else "gpu"
    trainer = Trainer(log_every_n_steps=1, devices=1 if args.n_gpus > 0 else 0, accelerator=accelerator, benchmark=True,
                      logger=False, enable_checkpointing=False)
    Seed.set(42)
    model.eval()

    

    # collect predictions
    preds = trainer.predict(model, dm.test_dataloader())
    preds = torch.cat(preds)
    # collect ground truth
    ys = torch.zeros_like(preds, dtype=torch.bool)

    for i, (_, y, _) in enumerate(tqdm(dm.test_dataloader())):
        # print(ys, y)
        ys[i * args.batch_size: (i + 1) * args.batch_size] = y
    # print(y.shape, ys.shape)
    # print(torch.eq(y.float(), (preds > 0.5)))

    # print(ys)
    # preds = preds.sigmoid()
    # print((preds > 0.5))
    acc = model.acc_fn(preds, ys)
    auc = model.auc_fn(preds, ys)
    results = {
        "acc": acc,
        "auc": auc
    }
    print(results)

    return auc, acc


def evaluate(args):
    config = read_yaml(args.config)

    if args.grid_search:
        results = []
        outfile = "Grid_search_results"
        
        try:
            for search in tqdm(CONFIGURATIONS):
                batch_size, lr, epochs, fusion, attention_heads, h_dim, pe = search
                print(f"Running grid search with the following parameters:")
                print(f"Batch Size: {batch_size}")
                print(f"Learning Rate: {lr}")
                print(f"Epochs: {epochs}")
                print(f"Fusion: {fusion}")
                print(f"Attention Heads: {attention_heads}")
                print(f"Hidden Dimensions: {h_dim}")
                print(f"Audio Positional Encoding: {pe}")

                args.epochs = epochs
                config["learning_rate"] = lr
                config['num_heads'] = attention_heads
                config['audio_positional_encoding'] = pe
                config['fusion'] = fusion
                config['hidden_layers'] = h_dim
                args.batch_size = batch_size

                ckpt, dm = train(args, config)
                auc, acc = eval_dataset(args, ckpt, dm)
                results.append({
                    "batch_size": batch_size,
                    "learning_rate": lr,
                    "epochs": epochs,
                    "fusion": fusion,
                    "attention_heads": attention_heads,
                    "hidden_dimensions": h_dim,
                    "audio_positional_encoding": pe,
                    "auc": auc,
                    "acc": acc
                })

        except Exception:
            print("Saving results to ", outfile)
        results.sort(key=lambda x: x['auc'], reverse=True)
        results.sort(key=lambda x: x['auc'], reverse=True)
        with open(outfile, "w", newline='') as f:
            fieldnames = ["batch_size", "learning_rate", "epochs", "fusion", 
                        "attention_heads", "hidden_dimensions", "audio_positional_encoding", "auc", "acc"]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for result in results:
                writer.writerow(result)
        # with open(outfile, "w") as f:
        #     f.write("batch_size\tlearning_rate\tepochs\tfusion\tattention_heads\thidden_dimensions\taudio_positional_encoding\tauc\tacc\n")
        #     for result in results:
        #         f.write(f"{result['batch_size']}\t{result['learning_rate']}\t{result['epochs']}\t{result['fusion']}\t{result['attention_heads']}\t{result['hidden_dimensions']}\t{result['audio_positional_encoding']}\t{result['auc']}\t{result['acc']}\n")
    else:
        
        ckpt, dm = train(args, config)
        # print(f"ckpt: {ckpt}, dm: {dm}")
        eval_dataset(args, ckpt, dm)


if __name__ == '__main__':
    parser = argparse.ArgumentParser("CelebV-HQ evaluation")
    parser.add_argument("--config", type=str,
                        help="Path to CelebV-HQ evaluation config file.")
    parser.add_argument("--data_path", type=str,
                        help="Path to CelebV-HQ dataset.")
    # parser.add_argument("--dataset", type=str,
    #                     help="type of dataset")    
    parser.add_argument("--marlin_ckpt", type=str, default=None,
                        help="Path to MARLIN checkpoint. Default: None, load from online.")
    parser.add_argument("--n_gpus", type=int, default=1)
    parser.add_argument("--precision", type=str, default="32")
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=200,
                        help="Max epochs to train.")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume training.")
    parser.add_argument("--skip_train", action="store_true", default=False,
                        help="Skip training and evaluate only.")
    parser.add_argument("--grid_search", action="store_true", default=False,
                        help="Perform a grid search and report best parameters")

    args = parser.parse_args()


    if args.skip_train:
        assert args.resume is not None

    evaluate(args)
