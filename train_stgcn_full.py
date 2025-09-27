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
from stgcn_features import sequences_to_stgcn_batches
from stgcn_graph import build_partitions, mp_edges
from stgcn_model import STGCNTwoStream
import time

class PIMDataset(Dataset):
    """Dataset for PIM movement sequences"""
    def __init__(self, file_paths, labels, sequence_length=30):
        self.file_paths = file_paths
        self.labels = labels
        self.sequence_length = sequence_length
        self.edges = mp_edges()

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        label = self.labels[idx]

        try:
            # Load CSV data
            df = pd.read_csv(file_path)

            # Group by timestamp and create sequence [T, 33, 3]
            sequence_data = []
            timestamps = df['timestamp'].unique()

            for timestamp in timestamps[:self.sequence_length]:
                frame_data = df[df['timestamp'] == timestamp]
                if len(frame_data) == 33:  # Ensure we have all 33 landmarks
                    # Sort by landmark_id and extract x,y,z
                    frame_data = frame_data.sort_values('landmark_id')
                    landmarks = frame_data[['x', 'y', 'z']].values  # [33, 3]
                    sequence_data.append(landmarks)

            # Pad or truncate to sequence_length
            if len(sequence_data) < self.sequence_length:
                # Pad with zeros
                padding = [np.zeros((33, 3)) for _ in range(self.sequence_length - len(sequence_data))]
                sequence_data.extend(padding)
            else:
                sequence_data = sequence_data[:self.sequence_length]

            # Convert to numpy array [T, V, C]
            sequence_array = np.array(sequence_data)

            # Convert to ST-GCN format using sequences_to_stgcn_batches
            Xj, Xb = sequences_to_stgcn_batches([sequence_array], self.edges)

            # Return joints and bones tensors
            return torch.tensor(Xj[0], dtype=torch.float32), torch.tensor(Xb[0], dtype=torch.float32), torch.tensor(label, dtype=torch.long)

        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            # Return zero tensors and label 0 as fallback
            return torch.zeros((3, self.sequence_length, 33)), torch.zeros((3, self.sequence_length, 33)), torch.tensor(0, dtype=torch.long)

class STGCNTrainer:
    """Trainer for full ST-GCN model using all data"""

    def __init__(self, num_classes=10, learning_rate=0.001, batch_size=16):
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Movement classes
        self.movement_classes = [
            'ballistic', 'chorea', 'decerebrate', 'decorticate', 'dystonia',
            'fencer posture', 'myoclonus', 'tremor', 'versive head'
        ]

        # Initialize model
        self.edges = build_partitions()
        self.model = STGCNTwoStream(num_classes, self.edges).to(self.device)

        # Loss and optimizer
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

        print(f"🧠 ST-GCN Model initialized on {self.device}")
        print(f"📊 Number of classes: {num_classes}")
        print(f"🔧 Learning rate: {learning_rate}, Batch size: {batch_size}")

    def load_all_data(self):
        """Load ALL training data from the pose_data directory for comprehensive single model"""
        print("📂 Loading ALL training data from pose_data directory...")

        all_files = []
        all_labels = []

        # Load from pose_data directory - parse movement from filename
        data_dir = 'pose_data'

        if not os.path.exists(data_dir):
            raise FileNotFoundError(f"pose_data directory not found at {data_dir}")

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
            raise ValueError("No training files found! Check pose_data directory.")

        # Split into train/val
        train_files, val_files, train_labels, val_labels = train_test_split(
            all_files, all_labels, test_size=0.2, random_state=42, stratify=all_labels
        )

        print(f"🎯 Train: {len(train_files)} files, Val: {len(val_files)} files")

        # Create datasets
        train_dataset = PIMDataset(train_files, train_labels)
        val_dataset = PIMDataset(val_files, val_labels)

        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size,
                                shuffle=True, num_workers=0)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size,
                              shuffle=False, num_workers=0)

        return train_loader, val_loader

    def train_epoch(self, train_loader):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0

        for batch_idx, (joints, bones, labels) in enumerate(train_loader):
            joints, bones, labels = joints.to(self.device), bones.to(self.device), labels.to(self.device)

            self.optimizer.zero_grad()

            # Forward pass
            outputs = self.model(joints, bones)
            loss = self.criterion(outputs, labels)

            # Backward pass
            loss.backward()
            self.optimizer.step()

            # Statistics
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            if batch_idx % 50 == 0:
                print(f"  Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}, Acc: {100.*correct/total:.2f}%")

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
            for joints, bones, labels in val_loader:
                joints, bones, labels = joints.to(self.device), bones.to(self.device), labels.to(self.device)

                outputs = self.model(joints, bones)
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

    def train(self, num_epochs=50, save_path='models/stgcn_full_model.pth'):
        """Full training loop"""
        print("🚀 Starting ST-GCN Training")
        print("=" * 50)

        # Load data
        train_loader, val_loader = self.load_all_data()

        # Training history
        history = {
            'train_loss': [], 'train_acc': [],
            'val_loss': [], 'val_acc': [], 'val_f1': []
        }

        best_val_acc = 0

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

            epoch_time = time.time() - start_time

            print(f"⏱️  Epoch {epoch+1}/{num_epochs} completed in {epoch_time:.2f}s")

            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_acc': val_acc,
                    'val_f1': val_f1,
                    'classes': self.movement_classes
                }, save_path)
                print(f"💾 Saved best model (val_acc: {val_acc:.2f}%)")

        print("🎉 Training completed!")
        print(f"🏆 Best validation accuracy: {best_val_acc:.2f}%")

        # Plot training history
        self.plot_history(history)

        return history

    def plot_history(self, history):
        """Plot training history"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))

        # Loss
        ax1.plot(history['train_loss'], label='Train')
        ax1.plot(history['val_loss'], label='Validation')
        ax1.set_title('Loss')
        ax1.legend()

        # Accuracy
        ax2.plot(history['train_acc'], label='Train')
        ax2.plot(history['val_acc'], label='Validation')
        ax2.set_title('Accuracy (%)')
        ax2.legend()

        # F1 Score
        ax3.plot(history['val_f1'], label='Validation F1')
        ax3.set_title('F1 Score')
        ax3.legend()

        # Learning curves
        ax4.plot(history['train_acc'], label='Train Acc')
        ax4.plot(history['val_acc'], label='Val Acc')
        ax4.plot(history['val_f1'], label='Val F1')
        ax4.set_title('Performance Metrics')
        ax4.legend()

        plt.tight_layout()
        plt.savefig('stgcn_training_history.png', dpi=300, bbox_inches='tight')
        plt.show()

    def save_model(self, path='models/stgcn_full_model.pth'):
        """Save the trained model"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'classes': self.movement_classes,
            'edges': self.edges
        }, path)
        print(f"💾 Model saved to {path}")

if __name__ == "__main__":
    # Train full ST-GCN model
    trainer = STGCNTrainer(num_classes=9, learning_rate=0.001, batch_size=16)
    history = trainer.train(num_epochs=30, save_path='models/stgcn_full_teacher.pth')
    trainer.save_model('models/stgcn_full_teacher.pth')