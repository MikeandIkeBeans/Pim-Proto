#!/usr/bin/env python3
"""
Enhanced LSTM Training for PIM Detection System

This script provides optimized LSTM training with pre-cached data loading for constant GPU utilization.

Usage:
    python train_lstm_full.py

Features:
- Pre-cached dataset loading (no I/O bottlenecks during training)
- Mixed precision training (FP16/FP32)
- Advanced optimization (AdamW + weight decay)
- Cosine annealing learning rate scheduling
- Early stopping with patience
- Gradient clipping
- Comprehensive training history plotting

Model Types:
- "normal": Standard LSTM architecture
- "joint_bone": Joint-Bone ensemble LSTM

Data Format:
- Expects CSV files in U:\pose_data\ directory
- Filenames should start with movement type (e.g., "normal_001_data.csv")
- Sequence parameters: 30 frames with 10-frame overlap for data augmentation
- Supported movements: normal, decorticate, dystonia, chorea, myoclonus,
  decerebrate, fencer posture, ballistic, tremor, versive head

Output:
- Trained model saved to models/lstm_enhanced_model.pth
- Training history plot saved as lstm_enhanced_training_history.png
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
import matplotlib.pyplot as plt
from pim_detection_system import PIMDatasetLSTM, PIMDetectorLSTM, JointBoneEnsembleLSTM
import time
from torch.cuda.amp import GradScaler, autocast
import torch.optim.lr_scheduler as lr_scheduler

class LSTMTrainer:
    """Enhanced trainer for LSTM-based PIM detection with pre-cached data loading"""

    def __init__(self, num_classes=10, learning_rate=0.001, batch_size=32, weight_decay=1e-4, model_type="normal"):
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.weight_decay = weight_decay
        self.model_type = model_type
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Movement classes (from PIM_MOVEMENTS)
        self.movement_classes = [
            'normal', 'decorticate', 'dystonia', 'chorea', 'myoclonus',
            'decerebrate', 'fencer posture', 'ballistic', 'tremor', 'versive head'
        ]

        # Initialize model
        self.model = JointBoneEnsembleLSTM(input_dim=3, hidden_dim=128, num_layers=3, num_classes=num_classes) \
                    if model_type == "joint_bone" else \
                    PIMDetectorLSTM(input_dim=3, hidden_dim=128, num_layers=3, num_classes=num_classes)
        self.model.to(self.device)

        # Enhanced optimizer (AdamW with weight decay)
        self.optimizer = optim.AdamW(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)

        # Learning rate scheduler with cosine annealing
        self.scheduler = lr_scheduler.CosineAnnealingWarmRestarts(self.optimizer, T_0=10, T_mult=2)

        # Mixed precision scaler
        self.scaler = GradScaler()

        # Early stopping parameters
        self.patience = 15
        self.best_val_loss = float('inf')
        self.early_stop_counter = 0

        # Loss function
        self.criterion = nn.CrossEntropyLoss()

        print(f"🧠 Enhanced LSTM Model ({model_type}) initialized on {self.device}")
        print(f"📊 Number of classes: {num_classes}")
        print(f"🔧 Learning rate: {learning_rate}, Batch size: {batch_size}")
        print(f"⚖️  Weight decay: {weight_decay}")
        print(f"📈 Cosine LR scheduling with warm restarts")
        print(f"🛑 Early stopping patience: {self.patience}")

        # Print GPU memory info if using CUDA
        if self.device.type == 'cuda':
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"🎮 GPU Memory: {gpu_memory:.1f} GB")
            print(f"🚀 Mixed precision training + enhanced optimizations enabled")

    def load_all_data(self, data_dir=r"U:\pose_data"):
        """Load ALL training data from the pose_data directory"""
        print(f"📂 Loading ALL training data from {data_dir} directory...")

        all_files = []
        all_labels = []

        if not os.path.exists(data_dir):
            raise FileNotFoundError(f"Data directory not found at {data_dir}")

        # Get all CSV files
        csv_files = [f for f in os.listdir(data_dir) if f.endswith('_data.csv')]

        for csv_file in csv_files:
            file_path = os.path.join(data_dir, csv_file)

            # Parse movement type from filename (first part before first underscore)
            movement = csv_file.split('_')[0]

            if movement in self.movement_classes:
                movement_idx = self.movement_classes.index(movement)
                all_files.append(file_path)
                all_labels.append(movement_idx)
            else:
                print(f"⚠️  Unknown movement type: {movement} in file {csv_file}")

        # Count files per movement
        for movement_idx, movement in enumerate(self.movement_classes):
            count = all_labels.count(movement_idx)
            print(f"  ✅ {movement}: {count} files")

        print(f"📊 Total files loaded: {len(all_files)}")

        if len(all_files) == 0:
            raise ValueError(f"No training files found! Check {data_dir} directory.")

        # Split into train/val
        train_files, val_files, train_labels, val_labels = train_test_split(
            all_files, all_labels, test_size=0.2, random_state=42, stratify=all_labels
        )

        print(f"🎯 Train: {len(train_files)} files, Val: {len(val_files)} files")

        # Create datasets (pre-cached loading)
        train_dataset = PIMDatasetLSTM(train_files, train_labels, seq_length=30, overlap=10)
        val_dataset = PIMDatasetLSTM(val_files, val_labels, seq_length=30, overlap=10)

        # Create data loaders (data is pre-cached, so minimal workers needed)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size,
                                shuffle=True, num_workers=0, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size,
                              shuffle=False, num_workers=0, pin_memory=True)

        return train_loader, val_loader

    def train_epoch(self, train_loader):
        """Train for one epoch with enhanced optimizations"""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0

        for batch_idx, (sequences, labels) in enumerate(train_loader):
            sequences, labels = sequences.to(self.device), labels.to(self.device)

            self.optimizer.zero_grad()

            # Mixed precision forward pass
            with autocast():
                outputs, _ = self.model(sequences)
                loss = self.criterion(outputs, labels)

            # Mixed precision backward pass with gradient clipping
            self.scaler.scale(loss).backward()

            # Gradient clipping
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            self.scaler.step(self.optimizer)
            self.scaler.update()

            # Statistics
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            if batch_idx % 10 == 0:  # Print progress
                current_lr = self.optimizer.param_groups[0]['lr']
                print(f"  Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}, Acc: {100.*correct/total:.2f}%, LR: {current_lr:.6f}")

        avg_loss = total_loss / len(train_loader)
        accuracy = 100. * correct / total

        return avg_loss, accuracy

    def validate(self, val_loader):
        """Validate the model"""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for sequences, labels in val_loader:
                sequences, labels = sequences.to(self.device), labels.to(self.device)

                with autocast():
                    outputs, _ = self.model(sequences)
                    loss = self.criterion(outputs, labels)

                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        avg_loss = total_loss / len(val_loader)
        accuracy = 100. * correct / total
        f1 = f1_score(all_labels, all_preds, average='weighted')

        return avg_loss, accuracy, f1

    def train(self, num_epochs=100, save_path='models/lstm_enhanced_model.pth', data_dir="pose_data"):
        """Enhanced training loop with early stopping and scheduling"""
        print("🚀 Starting Enhanced LSTM Training")
        print("=" * 60)

        # Load data
        train_loader, val_loader = self.load_all_data(data_dir)

        # Training history
        history = {
            'train_loss': [], 'train_acc': [],
            'val_loss': [], 'val_acc': [], 'val_f1': [],
            'learning_rates': []
        }

        best_val_acc = 0
        best_epoch = 0

        for epoch in range(num_epochs):
            print(f"\n📅 Epoch {epoch+1}/{num_epochs}")
            start_time = time.time()

            # Train
            train_loss, train_acc = self.train_epoch(train_loader)
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)

            # Validate
            val_loss, val_acc, val_f1 = self.validate(val_loader)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)
            history['val_f1'].append(val_f1)
            history['learning_rates'].append(self.optimizer.param_groups[0]['lr'])

            epoch_time = time.time() - start_time
            print(f"⏱️  Epoch {epoch+1}/{num_epochs} completed in {epoch_time:.2f}s")
            print(f"   Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            print(f"   Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%, Val F1: {val_f1:.4f}")

            # Save best model (by validation accuracy)
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_epoch = epoch + 1
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': self.scheduler.state_dict(),
                    'scaler_state_dict': self.scaler.state_dict(),
                    'val_acc': val_acc,
                    'val_f1': val_f1,
                    'best_val_loss': self.best_val_loss,
                    'classes': self.movement_classes,
                    'model_type': self.model_type
                }, save_path)
                print(f"💾 Saved best model (val_acc: {val_acc:.2f}%)")
                self.early_stop_counter = 0  # Reset early stopping counter
            else:
                self.early_stop_counter += 1

            # Early stopping check
            if self.early_stop_counter >= self.patience:
                print(f"🛑 Early stopping triggered after {epoch+1} epochs (no improvement for {self.patience} epochs)")
                break

            # Update learning rate
            self.scheduler.step()

        print("🎉 Training completed!")
        print(f"🏆 Best validation accuracy: {best_val_acc:.2f}% (epoch {best_epoch})")

        # Load best model for final evaluation
        checkpoint = torch.load(save_path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        final_val_loss, final_val_acc, final_val_f1 = self.validate(val_loader)

        print("📊 Final Model Performance:")
        print(f"   Validation Loss: {final_val_loss:.4f}")
        print(f"   Validation Accuracy: {final_val_acc:.2f}%")
        print(f"   Validation F1 Score: {final_val_f1:.4f}")

        return history

    def plot_training_history(self, history):
        """Plot comprehensive training history"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

        # Training and Validation Loss
        ax1.plot(history['train_loss'], label='Train Loss', color='blue')
        ax1.plot(history['val_loss'], label='Val Loss', color='red')
        ax1.set_title('Training and Validation Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Training and Validation Accuracy
        ax2.plot(history['train_acc'], label='Train Acc', color='green')
        ax2.plot(history['val_acc'], label='Val Acc', color='orange')
        ax2.set_title('Training and Validation Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # F1 Score
        ax3.plot(history['val_f1'], label='Val F1', color='purple')
        ax3.set_title('Validation F1 Score')
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('F1 Score')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # Learning Rate
        if 'learning_rates' in history:
            ax4.plot(history['learning_rates'], label='Learning Rate', color='red')
            ax4.set_title('Learning Rate Schedule')
            ax4.set_xlabel('Epoch')
            ax4.set_yscale('log')
            ax4.legend()
            ax4.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('lstm_enhanced_training_history.png', dpi=300, bbox_inches='tight')
        plt.show()

    def save_model(self, path='models/lstm_full_model.pth'):
        """Save the trained model"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'classes': self.movement_classes,
            'model_type': self.model_type
        }, path)
        print(f"💾 Model saved to {path}")

if __name__ == "__main__":
    # Train enhanced LSTM model
    trainer = LSTMTrainer(num_classes=10, learning_rate=0.001, batch_size=32, weight_decay=1e-4, model_type="normal")
    history = trainer.train(num_epochs=100, save_path='models/lstm_enhanced_model.pth', data_dir=r"U:\pose_data")
    trainer.plot_training_history(history)
    trainer.save_model('models/lstm_enhanced_model.pth')