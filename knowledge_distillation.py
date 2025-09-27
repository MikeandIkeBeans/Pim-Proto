#!/usr/bin/env python3
"""
Knowledge Distillation: LSTM Teacher → ST-GCN Student
Use LSTM ensemble predictions as soft targets to train ST-GCN models
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from collections import defaultdict
import os
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt

class KnowledgeDistiller:
    def __init__(self, temperature=2.0, alpha=0.5):
        """
        Initialize knowledge distillation

        Args:
            temperature: Temperature for softening probabilities
            alpha: Weight for distillation loss vs hard label loss
        """
        self.temperature = temperature
        self.alpha = alpha
        self.movement_classes = ['decerebrate', 'decorticate', 'dystonia', 'chorea',
                               'myoclonus', 'fencer posture', 'ballistic', 'tremor', 'versive head']

        # Load teacher (LSTM) and student (ST-GCN) models
        self.teacher_models = self.load_teacher_models()
        self.student_models = self.load_student_models()

        print(f"🧑‍🏫 Loaded {len(self.teacher_models)} teacher models (LSTM)")
        print(f👨‍🎓 Loaded {len(self.student_models)} student models (ST-GCN)")

    def load_teacher_models(self):
        """Load LSTM models as teachers"""
        models = []
        for i in range(3):
            path = f'models/ensemble/ensemble_model_{i}.pth'
            if os.path.exists(path):
                model_data = torch.load(path)
                models.append(model_data)
        return models

    def load_student_models(self):
        """Load ST-GCN models as students"""
        models = []
        for i in range(3):
            path = f'models/ensemble/ensemble_model_stgcn_{i}.pth'
            if os.path.exists(path):
                model_data = torch.load(path)
                models.append(model_data)
        return models

    def load_model_from_state_dict(self, model_data, architecture):
        """Load model from state dict"""
        if architecture == 'JointBoneEnsembleLSTM':
            from mediapipe_processor import JointBoneEnsembleLSTM
            model = JointBoneEnsembleLSTM(
                input_dim=3, hidden_dim=128, num_layers=3,
                num_classes=len(self.movement_classes)
            )
        elif architecture == 'STGCN':
            from stgcn_model import STGCN
            model = STGCN(
                num_class=len(self.movement_classes),
                num_point=17,
                num_person=1,
                in_channels=3,
                graph_args={'layout': 'openpose', 'strategy': 'spatial'}
            )
        else:
            raise ValueError(f"Unknown architecture: {architecture}")

        model.load_state_dict(model_data['state_dict'])
        return model

    def get_teacher_soft_targets(self, sequence_tensor):
        """Get soft targets from teacher ensemble"""
        teacher_logits = []

        for teacher_data in self.teacher_models:
            teacher = self.load_model_from_state_dict(teacher_data, teacher_data['architecture'])
            teacher.eval()

            with torch.no_grad():
                if teacher_data['architecture'] == 'JointBoneEnsembleLSTM':
                    logits, _ = teacher(sequence_tensor.unsqueeze(0))
                else:
                    logits = teacher(sequence_tensor.unsqueeze(0))
                teacher_logits.append(logits)

        # Average teacher logits
        avg_logits = torch.mean(torch.stack(teacher_logits), dim=0)

        # Apply temperature scaling
        soft_targets = torch.softmax(avg_logits / self.temperature, dim=1)

        return soft_targets

    def distillation_loss(self, student_logits, soft_targets, hard_labels):
        """Calculate distillation loss"""
        # Soft targets loss (KL divergence)
        student_soft = torch.log_softmax(student_logits / self.temperature, dim=1)
        soft_loss = nn.KLDivLoss(reduction='batchmean')(student_soft, soft_targets)

        # Hard labels loss (cross entropy)
        hard_loss = nn.CrossEntropyLoss()(student_logits, hard_labels)

        # Combined loss
        loss = self.alpha * (self.temperature ** 2) * soft_loss + (1 - self.alpha) * hard_loss

        return loss

    def prepare_training_data(self, max_samples=1000):
        """Prepare training data with soft targets"""
        print("📚 Preparing training data for distillation...")

        sequences = []
        hard_labels = []
        soft_targets = []

        # Load data from ensemble subsets
        movements = ['ballistic', 'chorea', 'decerebrate', 'decorticate',
                    'dystonia', 'myoclonus', 'tremor', 'versive head']

        samples_per_movement = max_samples // len(movements)

        for movement in movements:
            print(f"  Processing {movement}...")
            csv_files = []

            # Check ensemble subsets
            subset_dir = f'data/ensemble_subsets/subset_0/{movement}'
            if os.path.exists(subset_dir):
                csv_files.extend([os.path.join(subset_dir, f)
                                for f in os.listdir(subset_dir) if f.endswith('.csv')])

            # Check main data directory
            main_dir = f'data/{movement}'
            if os.path.exists(main_dir):
                csv_files.extend([os.path.join(main_dir, f)
                                for f in os.listdir(main_dir) if f.endswith('.csv')])

            count = 0
            for csv_file in csv_files:
                try:
                    df = pd.read_csv(csv_file)
                    sequence_data = df.values

                    if len(sequence_data) >= 30:
                        # Create sequence tensor
                        sequence_tensor = torch.tensor(sequence_data[:30], dtype=torch.float32)

                        # Get soft targets from teachers
                        soft_target = self.get_teacher_soft_targets(sequence_tensor)

                        sequences.append(sequence_tensor)
                        hard_labels.append(self.movement_classes.index(movement))
                        soft_targets.append(soft_target.squeeze(0))

                        count += 1
                        if count >= samples_per_movement:
                            break

                except Exception as e:
                    continue

        # Convert to tensors
        sequences = torch.stack(sequences)
        hard_labels = torch.tensor(hard_labels, dtype=torch.long)
        soft_targets = torch.stack(soft_targets)

        print(f"✅ Prepared {len(sequences)} training samples")

        return sequences, hard_labels, soft_targets

    def distill_knowledge(self, epochs=10, batch_size=32, learning_rate=0.001):
        """Perform knowledge distillation"""
        print("🎓 Starting Knowledge Distillation")
        print(f"   Temperature: {self.temperature}")
        print(f"   Alpha: {self.alpha}")
        print(f"   Epochs: {epochs}")
        print("=" * 50)

        # Prepare data
        sequences, hard_labels, soft_targets = self.prepare_training_data()

        # Create dataset
        dataset = TensorDataset(sequences, hard_labels, soft_targets)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        # Train each student model
        distilled_models = []

        for i, student_data in enumerate(self.student_models):
            print(f"\n🧑‍🎓 Training Student Model {i+1}")
            print("-" * 30)

            # Load student model
            student = self.load_model_from_state_dict(student_data, student_data['architecture'])
            student.train()

            # Optimizer
            optimizer = optim.Adam(student.parameters(), lr=learning_rate)

            # Training loop
            losses = []

            for epoch in range(epochs):
                epoch_loss = 0.0
                batch_count = 0

                for batch_seq, batch_hard, batch_soft in dataloader:
                    optimizer.zero_grad()

                    # Forward pass
                    student_logits = student(batch_seq)

                    # Calculate distillation loss
                    loss = self.distillation_loss(student_logits, batch_soft, batch_hard)

                    # Backward pass
                    loss.backward()
                    optimizer.step()

                    epoch_loss += loss.item()
                    batch_count += 1

                avg_loss = epoch_loss / batch_count
                losses.append(avg_loss)

                if (epoch + 1) % 5 == 0:
                    print(".4f"
            # Save distilled model
            distilled_path = f'models/ensemble/ensemble_model_stgcn_{i}_distilled.pth'
            torch.save({
                'model_id': f'stgcn_{i}_distilled',
                'architecture': 'STGCN',
                'classes': self.movement_classes,
                'distillation': {
                    'teacher': 'LSTM_ensemble',
                    'temperature': self.temperature,
                    'alpha': self.alpha,
                    'epochs': epochs
                },
                'timestamp': datetime.now().isoformat(),
                'state_dict': student.state_dict()
            }, distilled_path)

            distilled_models.append(distilled_path)
            print(f"💾 Saved distilled model: {distilled_path}")

        # Update ensemble config to use distilled models
        config_path = 'models/ensemble/ensemble_config_distilled.pth'
        config = {
            'model_paths': distilled_models,
            'movement_classes': self.movement_classes,
            'voting_strategy': 'majority',
            'distillation_applied': True,
            'teacher_model': 'LSTM_ensemble'
        }
        torch.save(config, config_path)

        print(f"\n🎯 Distillation Complete!")
        print(f"📁 New config saved: {config_path}")
        print(f"🔄 Update your ensemble_inference.py to use: {config_path}")

        return distilled_models

def main():
    """Main function"""
    # Initialize distiller with reasonable defaults
    distiller = KnowledgeDistiller(temperature=2.0, alpha=0.7)

    # Run distillation
    distiller.distill_knowledge(epochs=15, batch_size=16, learning_rate=0.0005)

if __name__ == "__main__":
    main()