# PIM Detection System

Patient Involuntary Movement (PIM) detection using deep learning on pose estimation data.

## Training Scripts

### STGCN Training
```bash
python train_stgcn_full.py
```
- Uses Spatial-Temporal Graph Convolutional Network
- Optimized for GPU utilization with pre-cached data loading
- Mixed precision training with advanced optimizations

### LSTM Training
```bash
python train_lstm_full.py
```
- Uses LSTM networks for sequential movement classification
- Pre-cached data loading for constant GPU utilization
- Mixed precision training with cosine annealing LR scheduling

## Data Format

Place your training data in the `U:\pose_data\` directory with CSV files named as:
- `{movement_type}_{video_id}_data.csv`

Supported movement types:
- normal, decorticate, dystonia, chorea, myoclonus
- decerebrate, fencer posture, ballistic, tremor, versive head

Each CSV should contain columns: `timestamp`, `landmark_id`, `x`, `y`, `z`

## Models

Trained models are saved to the `models/` directory:
- `stgcn_enhanced_model.pth` - STGCN model
- `lstm_enhanced_model.pth` - LSTM model

## Requirements

Install dependencies:
```bash
pip install -r requirements.txt
```

## Key Features

- ✅ Pre-cached data loading (eliminates I/O bottlenecks)
- ✅ Mixed precision training (FP16/FP32)
- ✅ Advanced optimization (AdamW, weight decay, gradient clipping)
- ✅ Cosine annealing learning rate scheduling
- ✅ Early stopping with patience
- ✅ Comprehensive training history plotting