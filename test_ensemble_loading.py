#!/usr/bin/env python3
"""
Test script to check ensemble model loading
"""

from model_comparison_demo import ModelComparisonDemo
import os

# Define model configs
model_configs = {
    'joint_bone': {'path': 'models/pim_model_joint_bone.pth'},
    'ensemble_0': {'path': 'models/ensemble/ensemble_model_0.pth'},
    'ensemble_1': {'path': 'models/ensemble/ensemble_model_1.pth'},
    'ensemble_2': {'path': 'models/ensemble/ensemble_model_2.pth'},
    'stgcn_0': {'path': 'models/ensemble/ensemble_model_stgcn_0.pth'},
    'stgcn_1': {'path': 'models/ensemble/ensemble_model_stgcn_1.pth'},
    'stgcn_full': {'path': 'models/stgcn_full_comprehensive.pth'},
    'ensemble': {'path': 'ensemble_voting'}  # Special ensemble mode
}

# Check if models exist
for name, config in model_configs.items():
    if config['path'] and not os.path.exists(config['path']):
        print(f'Warning: {config["path"]} does not exist')

# Try to create demo
try:
    demo = ModelComparisonDemo(model_configs)
    print(f'Models loaded: {list(demo.models.keys())}')
    print(f'Active model: {demo.active_model}')
    print(f'Ensemble in models: {"ensemble" in demo.models}')
    if 'ensemble' in demo.models:
        print(f'Ensemble model type: {demo.models["ensemble"]["model_type"]}')
        print(f'Ensemble has {len(demo.models["ensemble"]["ensemble_models"])} models')
except Exception as e:
    print(f'Error: {e}')