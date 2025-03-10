# Getting started with D3PD
## Training

To train D3PD with 8 GPUs, run:
```bash
bash tools/dist_train.sh $CONFIG 8
```

## Evaluation

To evaluate D3PD with 8 GPU, run:
```bash
bash tools/dist_test.sh $YOUR_CONFIG $YOUR_CKPT 8 --eval=mAP
```

