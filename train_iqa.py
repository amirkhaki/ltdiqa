"""Train LTD for NR-IQA (regression on MOS).

Example:
    python train_iqa.py --name koniq_ltd --dataset_root /data/iqa \
        --train_sets KonIQ-10k --eval_sets KonIQ-10k SPAQ LIVE_Challenge --download
"""
import argparse
import json
import os
import random
import time
from types import SimpleNamespace

import numpy as np
import torch
from tensorboardX import SummaryWriter

from iqa.data import build_loaders
from iqa.engine import evaluate, mean_of
from networks.trainer import Trainer
import util


def get_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # experiment
    p.add_argument('--name', required=True)
    p.add_argument('--checkpoints_dir', default='./checkpoints')
    p.add_argument('--seed', type=int, default=3000)
    # data
    p.add_argument('--dataset_root', default='./data', help='where iqadataset stores / finds the datasets')
    p.add_argument('--download', action='store_true', help='let iqadataset download missing datasets')
    p.add_argument('--train_sets', nargs='+', default=['KonIQ-10k'], help='iqadataset names')
    p.add_argument('--eval_sets', nargs='*', default=[], help='extra test sets (cross-dataset if not in --train_sets)')
    p.add_argument('--split_dir', default='./splits')
    p.add_argument('--split_seed', type=int, default=0)
    p.add_argument('--split_csv', nargs='*', default=[], metavar='NAME=PATH',
                   help='official split CSVs (columns image_name,set), e.g. KonIQ-10k=koniq10k_distributions_sets.csv; '
                        'overrides the random split for that dataset')
    p.add_argument('--split_ratios', type=float, nargs=3, default=[0.8, 0.1, 0.1], metavar=('TRAIN', 'VAL', 'TEST'))
    p.add_argument('--crop', type=int, default=224)
    p.add_argument('--resize_short', type=int, default=0, help='resize short side first; 0 = keep native resolution')
    p.add_argument('--test_grid', type=int, default=3, help='eval uses test_grid x test_grid crops per image (1 = center crop)')
    p.add_argument('--num_workers', type=int, default=8)
    p.add_argument('--batch_size', type=int, default=64)
    p.add_argument('--eval_batch_size', type=int, default=16, help='images per batch at eval (x test_grid^2 crops)')
    # model
    p.add_argument('--arch', default='CLIP:ViT-L/14')
    p.add_argument('--select_k', type=int, default=5)
    p.add_argument('--layer_start', type=int, default=11, help='first hooked ViT block')
    p.add_argument('--layer_end', type=int, default=20, help='one past the last hooked ViT block')
    # optimisation
    p.add_argument('--loss', default='mse', choices=['mse', 'l1', 'huber'])
    p.add_argument('--optim', default='adam', choices=['adam', 'sgd'])
    p.add_argument('--lr', type=float, default=5e-5)
    p.add_argument('--beta1', type=float, default=0.9)
    p.add_argument('--weight_decay', type=float, default=0.0)
    p.add_argument('--epochs', type=int, default=30)
    p.add_argument('--patience', type=int, default=3, help='epochs without val-SRCC gain before lr /= 10')
    p.add_argument('--min_lr', type=float, default=1e-6, help='stop once lr would fall below this')
    return p.parse_args()


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def fmt(results):
    return ' | '.join(f"{k}: srcc {m['srcc']:.4f} plcc {m['plcc']:.4f}" for k, m in results.items())


def main():
    args = get_args()
    set_seed(args.seed)
    save_dir = os.path.join(args.checkpoints_dir, args.name)
    util.mkdirs(save_dir)
    with open(os.path.join(save_dir, 'args.json'), 'w') as f:
        json.dump(vars(args), f, indent=2)

    data = build_loaders(args)
    with open(os.path.join(save_dir, 'score_norm.json'), 'w') as f:
        json.dump(data['norms'], f, indent=2)

    # the legacy Trainer reads these fields from an options object
    opt = SimpleNamespace(
        name=args.name, checkpoints_dir=args.checkpoints_dir, gpu_ids=[0],
        arch=args.arch, select_k=args.select_k, layer_range=(args.layer_start, args.layer_end),
        fix_backbone=True, optim=args.optim, lr=args.lr, beta1=args.beta1,
        weight_decay=args.weight_decay, loss=args.loss, save_trainable_only=True)
    trainer = Trainer(opt)
    device = trainer.device
    writer = SummaryWriter(os.path.join(save_dir, 'tb'))
    n_trainable = sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)
    print(f"[model] trainable parameters: {n_trainable / 1e6:.2f}M")

    best_srcc, bad_epochs, t0 = -float('inf'), 0, time.time()
    best_path = os.path.join(save_dir, 'best')
    n_batches = len(data['train'])

    for epoch in range(1, args.epochs + 1):
        trainer.train()
        for i, batch in enumerate(data['train']):
            trainer.total_steps += 1
            trainer.set_input(batch)
            trainer.optimize_parameters()
            writer.add_scalar('train/loss', trainer.loss.item(), trainer.total_steps)
            print(f"\r[epoch {epoch:02d}/{args.epochs} | batch {i + 1:04d}/{n_batches:04d} | {time.time() - t0:.0f}s] "
                  f"loss {trainer.loss.item():.4f}", end='')
        print()

        val = evaluate(trainer.model, data['val'], device)
        srcc, plcc = mean_of(val, 'srcc'), mean_of(val, 'plcc')
        writer.add_scalar('val/srcc', srcc, epoch)
        writer.add_scalar('val/plcc', plcc, epoch)
        print(f"(val @ epoch {epoch}) {fmt(val)}")

        if srcc > best_srcc + 1e-4:
            best_srcc, bad_epochs = srcc, 0
            trainer.save_networks('best')
            print(f"  new best val SRCC {best_srcc:.4f} -> saved")
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience:
                if not trainer.adjust_learning_rate(args.min_lr):
                    print('lr below min_lr, stopping.')
                    break
                trainer.load_networks(best_path)  # continue from the best weights at the lower lr
                bad_epochs = 0
                print(f"  lr -> {trainer.optimizer.param_groups[0]['lr']:.1e}, reloaded best checkpoint")

    # final: best checkpoint on every test set
    trainer.load_networks(best_path)
    test = evaluate(trainer.model, data['test'], device)
    print('\n[test] ' + fmt(test))
    with open(os.path.join(save_dir, 'results.json'), 'w') as f:
        json.dump({'best_val_srcc': best_srcc, 'test': test}, f, indent=2)


if __name__ == '__main__':
    main()
