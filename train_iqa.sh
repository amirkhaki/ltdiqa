#!/bin/bash
# Train on KonIQ-10k (80/10/10 random split, seed 0), test in-domain + cross-dataset.
CUDA_VISIBLE_DEVICES=0 python train_iqa.py \
    --name=koniq_ltd \
    --dataset_root=./data \
    --download \
    --train_sets KonIQ-10k \
    --eval_sets KonIQ-10k SPAQ LIVE_Challenge \
    --arch="CLIP:ViT-L/14" \
    --select_k=5 \
    --loss=mse \
    --lr=0.00005 \
    --batch_size=64 \
    --epochs=30
