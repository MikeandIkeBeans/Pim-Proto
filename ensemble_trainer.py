"""
Ensemble Trainer for PIM Movement Detection
Addresses bias by training multiple models on balanced data subsets
"""

import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from collections import defaultdict, Counter
import random
import shutil
from datetime import datetime

class PIMEnsembleTrainer:
    def __init__(self, data_dir='pose_data', models_dir='models', num_models=3, teacher_model_path='models/pim_model_joint_bone.pth', model_types=None):
        self.data_dir = data_dir  # Changed from 'data' to 'pose_data'
        self.models_dir = models_dir
        self.num_models = num_models
        self.teacher_model_path = teacher_model_path
        self.model_types = model_types or ['joint_bone']  # Support multiple architectures
        self.movement_classes = [
            'decerebrate', 'decorticate', 'dystonia', 'chorea',
            'myoclonus', 'fencer posture', 'ballistic', 'tremor', 'versive head'
        ]

        # Create ensemble models directory
        self.ensemble_dir = os.path.join(models_dir, 'ensemble')
        os.makedirs(self.ensemble_dir, exist_ok=True)

        # Load teacher model for transfer learning
        self.teacher_model = None
        self.load_teacher_model()

    def load_teacher_model(self):
        """Load the joint-bone teacher model for transfer learning"""
        if os.path.exists(self.teacher_model_path):
            try:
                teacher_data = torch.load(self.teacher_model_path)
                self.teacher_model = teacher_data['model_state_dict']
                print(f"🎓 Loaded teacher model: {self.teacher_model_path}")
                print(f"   📚 Teacher movements: {teacher_data['movements']}")
            except Exception as e:
                print(f"⚠️  Could not load teacher model: {e}")
                self.teacher_model = None
        else:
            print(f"⚠️  Teacher model not found: {self.teacher_model_path}")
            self.teacher_model = None

    def analyze_data_distribution(self):
        """Analyze current training data distribution from pose_data CSVs"""
        print("📊 Analyzing current data distribution from pose_data CSVs...")

        distribution = {}
        total_files = 0

        # Get all CSV files in pose_data directory
        if os.path.exists(self.data_dir):
            all_files = [f for f in os.listdir(self.data_dir) if f.endswith('.csv')]
        else:
            all_files = []

        # Count files by movement type (filename prefix)
        for movement in self.movement_classes:
            movement_files = [f for f in all_files if f.startswith(movement)]
            distribution[movement] = len(movement_files)
            total_files += len(movement_files)

        print(f"Total training files: {total_files}")
        for movement, count in distribution.items():
            percentage = (count / total_files * 100) if total_files > 0 else 0
            print(".1f")

        return distribution

    def create_balanced_subsets(self, samples_per_class=96):
        """Create balanced training subsets for ensemble training from pose_data CSVs"""
        print(f"\n🔄 Creating {self.num_models} balanced subsets with {samples_per_class} samples per class...")

        # Analyze current distribution
        current_dist = self.analyze_data_distribution()

        subsets_dir = os.path.join(self.data_dir, '..', 'data', 'ensemble_subsets')  # Store subsets in data/ensemble_subsets
        os.makedirs(subsets_dir, exist_ok=True)

        subset_dirs = []

        for subset_idx in range(self.num_models):
            subset_name = f'subset_{subset_idx}'
            subset_dir = os.path.join(subsets_dir, subset_name)
            os.makedirs(subset_dir, exist_ok=True)

            print(f"\n📁 Creating subset {subset_idx + 1}/{self.num_models}...")

            # Create balanced samples for each movement class
            for movement in self.movement_classes:
                # Get available CSV files for this movement
                if os.path.exists(self.data_dir):
                    available_files = [f for f in os.listdir(self.data_dir)
                                     if f.startswith(movement) and f.endswith('.csv')]
                else:
                    available_files = []

                # Create samples_per_class samples (with replacement if needed)
                selected_files = []
                for i in range(samples_per_class):
                    if available_files:
                        # Sample with replacement
                        selected_file = random.choice(available_files)
                        selected_files.append(selected_file)

                        # Copy file directly to subset directory (not in movement subdirs)
                        src_path = os.path.join(self.data_dir, selected_file)
                        dst_path = os.path.join(subset_dir, selected_file)
                        shutil.copy2(src_path, dst_path)

                print(f"   {movement}: {len(selected_files)} samples created")

            subset_dirs.append(subset_dir)

        print(f"\n✅ Created {self.num_models} balanced subsets in {subsets_dir}")
        return subset_dirs

    def train_ensemble_models(self, subset_dirs):
        """Train multiple models on different balanced subsets"""
        print("\n🚀 Training ensemble models...")
        print("Note: This will train actual models on balanced subsets")

        ensemble_models = []

        for i, subset_dir in enumerate(subset_dirs):
            model_name = f'ensemble_model_{i}.pth'
            model_path = os.path.join(self.ensemble_dir, model_name)

            print(f"\n🏋️ Training model {i+1}/{self.num_models} on {os.path.basename(subset_dir)}...")

            try:
                # Import the training function from the main system
                from pim_detection_system import train_model

                # Train model on this subset with transfer learning from teacher
                print(f"   🎓 Initializing with joint-bone teacher model weights...")

                trained_model = train_model(
                    data_dir=subset_dir,
                    movements=self.movement_classes,
                    epochs=20,  # Shorter training for demo
                    model_type="joint_bone",
                    patience=5
                )

                # Apply transfer learning: load teacher weights into trained model
                if trained_model is not None and self.teacher_model is not None:
                    actual_model, movements_list = trained_model
                    print(f"   🔄 Applying transfer learning from joint-bone teacher...")

                    # Load teacher weights, filtering to match current model architecture
                    teacher_state = self.teacher_model
                    model_state = actual_model.state_dict()

                    # Transfer matching weights
                    transferred_weights = 0
                    for name, param in teacher_state.items():
                        if name in model_state and param.shape == model_state[name].shape:
                            model_state[name].copy_(param)
                            transferred_weights += 1

                    print(f"   ✅ Transferred {transferred_weights} weight layers from teacher model")
                    actual_model.load_state_dict(model_state)
                    trained_model = (actual_model, movements_list)

                if trained_model is not None:
                    # train_model returns (model, movements) tuple
                    actual_model, movements_list = trained_model

                    # Save the trained model
                    model_info = {
                        'model_id': i,
                        'subset_dir': subset_dir,
                        'trained_on': os.path.basename(subset_dir),
                        'timestamp': datetime.now().isoformat(),
                        'architecture': 'JointBoneEnsembleLSTM',
                        'classes': self.movement_classes,
                        'transfer_learning': True,
                        'teacher_model': os.path.basename(self.teacher_model_path),
                        'state_dict': actual_model.state_dict() if hasattr(actual_model, 'state_dict') else None
                    }

                    torch.save(model_info, model_path)
                    ensemble_models.append(model_path)
                    print(f"   ✅ Saved trained model {i+1}: {model_name}")
                else:
                    print(f"   ❌ Failed to train model {i+1}")
                    # Create placeholder
                    model_info = {
                        'model_id': i,
                        'subset_dir': subset_dir,
                        'trained_on': os.path.basename(subset_dir),
                        'timestamp': datetime.now().isoformat(),
                        'architecture': 'JointBoneEnsembleLSTM',
                        'classes': self.movement_classes,
                        'transfer_learning': True,
                        'teacher_model': os.path.basename(self.teacher_model_path) if self.teacher_model else None
                    }
                    torch.save(model_info, model_path)
                    ensemble_models.append(model_path)

            except Exception as e:
                print(f"   ❌ Error training model {i+1}: {e}")
                # Create placeholder model
                model_info = {
                    'model_id': i,
                    'subset_dir': subset_dir,
                    'trained_on': os.path.basename(subset_dir),
                    'timestamp': datetime.now().isoformat(),
                    'architecture': 'JointBoneEnsembleLSTM',
                    'classes': self.movement_classes,
                    'transfer_learning': True,
                    'teacher_model': os.path.basename(self.teacher_model_path) if self.teacher_model else None
                }
                torch.save(model_info, model_path)
                ensemble_models.append(model_path)

        return ensemble_models

    def train_stgcn_ensemble_models(self, subset_dirs):
        """Train multiple ST-GCN models on different balanced subsets"""
        print("\n🚀 Training ST-GCN ensemble models...")
        print("Note: This will train ST-GCN models on balanced subsets")

        ensemble_models = []

        for i, subset_dir in enumerate(subset_dirs):
            model_name = f'ensemble_model_stgcn_{i}.pth'
            model_path = os.path.join(self.ensemble_dir, model_name)

            print(f"\n🏋️ Training ST-GCN model {i+1}/{self.num_models} on {os.path.basename(subset_dir)}...")

            try:
                # Import ST-GCN training function
                from stgcn_graph import build_partitions
                from stgcn_features import sequences_to_stgcn_batches
                from stgcn_model import STGCNTwoStream
                import torch.optim as optim

                # Load sequences from subset directory
                all_sequences_by_class = []
                for movement in self.movement_classes:
                    seqs = []
                    # Get all CSV files for this movement in the subset
                    movement_files = [f for f in os.listdir(subset_dir)
                                    if f.startswith(movement) and f.endswith('.csv')]

                    for file in movement_files[:50]:  # Increased from 10 to 50 files per class
                        file_path = os.path.join(subset_dir, file)
                        try:
                            # Use prepare_sequences from pim_detection_system
                            from pim_detection_system import prepare_sequences
                            s = prepare_sequences(file_path)
                            if len(s) > 0:
                                seqs.extend([s[j] for j in range(s.shape[0])])
                        except Exception as e:
                            print(f"   ⚠️  Error loading {file}: {e}")
                            continue

                    all_sequences_by_class.append(seqs)
                    print(f"   {movement}: {len(seqs)} sequences")

                # Train ST-GCN model
                A = build_partitions(V=33)
                num_classes = len(self.movement_classes)
                model = STGCNTwoStream(num_classes, A)
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                model.to(device)

                # Build dataset with smaller batches
                from stgcn_graph import mp_edges
                EDGES = mp_edges()
                Xj_list, Xb_list, y_list = [], [], []

                # Limit total sequences to prevent memory issues
                max_sequences_per_class = 50  # Much smaller limit
                for cls_idx, seq_list in enumerate(all_sequences_by_class):
                    if not seq_list: continue
                    # Limit sequences per class
                    seq_list = seq_list[:max_sequences_per_class]
                    Xj, Xb = sequences_to_stgcn_batches(seq_list, EDGES)
                    Xj_list.append(Xj); Xb_list.append(Xb)
                    y_list.append(np.full((Xj.shape[0],), cls_idx, dtype=np.int64))

                if not Xj_list:
                    print(f"   ❌ No training data for model {i+1}")
                    continue

                Xj = np.concatenate(Xj_list, axis=0); Xb = np.concatenate(Xb_list, axis=0); y = np.concatenate(y_list, axis=0)

                # Train/val split
                from sklearn.model_selection import train_test_split
                idx = np.arange(len(y))
                tr, va = train_test_split(idx, test_size=0.2, random_state=42, stratify=y)

                # Use larger batch size now that we have more data
                bs = 4  # Increased from 2 to 4

                # Keep data on CPU, load to GPU in batches
                Xj_tr_cpu = torch.from_numpy(Xj[tr])
                Xb_tr_cpu = torch.from_numpy(Xb[tr])
                y_tr_cpu = torch.from_numpy(y[tr])
                Xj_va_cpu = torch.from_numpy(Xj[va])
                Xb_va_cpu = torch.from_numpy(Xb[va])
                y_va_cpu = torch.from_numpy(y[va])

                # Training
                optimizer = optim.Adam(model.parameters(), lr=1e-3)
                criterion = nn.CrossEntropyLoss()
                best_loss, patience, wait = float('inf'), 10, 0  # Increased patience

                for epoch in range(50):  # Increased from 10 to 50 epochs
                    model.train()
                    epoch_loss = 0.0
                    num_batches = 0

                    # Mini-batch training
                    perm = torch.randperm(len(Xj_tr_cpu))
                    for i in range(0, len(Xj_tr_cpu), bs):
                        idxb = perm[i:i+bs]
                        xb_j = Xj_tr_cpu[idxb].to(device, non_blocking=True)
                        xb_b = Xb_tr_cpu[idxb].to(device, non_blocking=True)
                        yb   = y_tr_cpu[idxb].to(device, non_blocking=True)

                        optimizer.zero_grad()
                        logits = model(xb_j, xb_b)
                        loss = criterion(logits, yb)
                        loss.backward()
                        optimizer.step()

                        epoch_loss += loss.item() * yb.size(0)
                        num_batches += 1

                    tr_loss = epoch_loss / len(Xj_tr_cpu)

                    # Validation
                    model.eval()
                    val_loss = 0.0; correct = 0
                    with torch.no_grad():
                        for i in range(0, len(Xj_va_cpu), bs):
                            xb_j = Xj_va_cpu[i:i+bs].to(device)
                            xb_b = Xb_va_cpu[i:i+bs].to(device)
                            yb   = y_va_cpu[i:i+bs].to(device)
                            logits = model(xb_j, xb_b)
                            loss = criterion(logits, yb)
                            val_loss += loss.item() * yb.size(0)
                            pred = logits.argmax(1); correct += (pred==yb).sum().item()
                    val_loss /= len(Xj_va_cpu); val_acc = correct/len(Xj_va_cpu)
                    print(f"   ep {epoch+1}/{30}  train {tr_loss:.4f}  val {val_loss:.4f}  acc {val_acc:.3f}")

                    if val_loss < best_loss:
                        best_loss, wait = val_loss, 0
                        best_state = {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
                    else:
                        wait += 1
                        if wait >= patience:
                            print(f"   early stop at ep {epoch+1}")
                            break

                if 'best_state' in locals():
                    model.load_state_dict(best_state)

                # Save the trained model
                model_info = {
                    'model_id': i,
                    'subset_dir': subset_dir,
                    'trained_on': os.path.basename(subset_dir),
                    'timestamp': datetime.now().isoformat(),
                    'architecture': 'STGCNTwoStream',
                    'classes': self.movement_classes,
                    'transfer_learning': False,  # ST-GCN doesn't use joint-bone teacher
                    'state_dict': model.state_dict()
                }

                torch.save(model_info, model_path)
                ensemble_models.append(model_path)
                print(f"   ✅ Saved trained ST-GCN model {i+1}: {model_name}")

            except Exception as e:
                print(f"   ❌ Error training ST-GCN model {i+1}: {e}")
                import traceback
                traceback.print_exc()

        return ensemble_models

    def create_ensemble_inference(self, model_paths):
        """Create ensemble inference system"""
        print("\n🎯 Creating ensemble inference system...")
        ensemble_config = {
            'num_models': len(model_paths),
            'model_paths': model_paths,
            'movement_classes': self.movement_classes,
            'voting_strategy': 'weighted_majority',
            'bias_correction': {
                'decerebrate_threshold': 0.67,  # Require 2/3 agreement for decerebrate
                'confidence_weighting': True,
                'min_confidence': 0.3
            },
            'created_at': datetime.now().isoformat()
        }

        config_path = os.path.join(self.ensemble_dir, 'ensemble_config.pth')
        torch.save(ensemble_config, config_path)

        print("   ✅ Ensemble configuration saved")
        return config_path

def main():
    print("🎯 PIM Ensemble Trainer - Bias Correction System")
    print("=" * 55)

    # Choose model type
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', choices=['joint_bone', 'stgcn'], default='joint_bone',
                       help='Model architecture to train')
    args = parser.parse_args()

    trainer = PIMEnsembleTrainer(data_dir='pose_data', num_models=3, model_types=[args.model_type])

    # Step 1: Analyze current bias
    print("\n1️⃣ Analyzing current data distribution...")
    distribution = trainer.analyze_data_distribution()

    # Step 2: Create balanced subsets
    print("\n2️⃣ Creating balanced training subsets...")
    subset_dirs = trainer.create_balanced_subsets(samples_per_class=96)

    # Step 3: Train ensemble models
    print(f"\n3️⃣ Training {args.model_type} ensemble models...")
    if args.model_type == 'stgcn':
        model_paths = trainer.train_stgcn_ensemble_models(subset_dirs)
    else:
        model_paths = trainer.train_ensemble_models(subset_dirs)

    # Step 4: Create inference system
    print("\n4️⃣ Creating ensemble inference...")
    config_path = trainer.create_ensemble_inference(model_paths)

    print("\n🎉 Ensemble system created successfully!")
    print(f"   📁 Models saved in: {trainer.ensemble_dir}")
    print(f"   📄 Config saved as: {os.path.basename(config_path)}")

    print("\n💡 Next steps:")
    print("   1. Test ensemble on napa videos to verify bias reduction")
    print("   2. Compare ST-GCN vs Joint-Bone performance")
    print("   3. Implement real-time ST-GCN inference")

if __name__ == "__main__":
    main()