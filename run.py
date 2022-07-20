from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, List, Optional

import numpy as np
import pandas as pd
from omegaconf import MISSING

from matching.MatchingEvaluation_class import MatchingEvaluation
from matching.Matching_class import Matching
from project_utils import Waterbirds_CelebA128_dataloader, features_loader


@dataclass
class DatasetConfig:
    """Base class of data configs."""
    dataset: str = MISSING
    features_path: str = MISSING
    labels_train_path: str = MISSING
    labels_val_path: str = MISSING
    labels_test_path: str = MISSING
    subgroup_vars: List[str] = MISSING
    outcome: str = MISSING


@dataclass
class MatchingConfig:
    fixed_caliper_interval: List[float] = MISSING
    std_caliper: Optional[float] = None
    reweight: Optional[str] = None
    temperature: Optional[float] = None


# ====================================== base config schema =======================================
@dataclass
class Config:
    """
    Configuration for this program
    """
    data: DatasetConfig = MISSING
    matching: MatchingConfig = MISSING
    seed: int = MISSING
    direction_list: Any = MISSING
    # Updated from hydra conf in main.py
    results_dir: str = ""


# ====================================== DATA LOADER =======================================
LOADER = {
    'celebA_128': Waterbirds_CelebA128_dataloader,
    'waterbirds': Waterbirds_CelebA128_dataloader,
}
POSITIONAL0 = {
    'celebA_128': 6,
    'waterbirds': 5,
}


def possible_values_check(cfg: Config):
    """Help functions checking an acceptable argument has been given"""
    if cfg.data.dataset not in LOADER.keys():
        raise ValueError(
            f'{cfg.data.dataset} value for DATASET argument invalid. It should be one of the following: {LOADER.keys()}]')


def prepare_data(cfg: Config):
    """Load and returns Attributes and Features data"""
    # Load train and test labels
    label_df_train = LOADER[cfg.data.dataset](Path(__file__).parent.resolve() / cfg.data.labels_train_path)
    label_df_val = LOADER[cfg.data.dataset](Path(__file__).parent.resolve() / cfg.data.labels_val_path)
    label_df_test = LOADER[cfg.data.dataset](Path(__file__).parent.resolve() /cfg.data.labels_test_path)
    print(f"Train Attributes shape: {label_df_train.shape}")
    print(f"Validation Attributes shape: {label_df_val.shape}")
    print(f"Test Attributes shape: {label_df_test.shape}")

    # Load Features; N.B. normalise
    print(f'Loading Features File: {datetime.now()}')
    features_train, features_val, features_test = features_loader(dir_=Path(__file__).parent.resolve() / cfg.data.features_path)
    print(f'Features Loaded: {datetime.now()}')
    print(f"Train Features shape: {features_train.shape}")
    print(f"Validation Features shape: {features_val.shape}")
    print(f"Test Features shape: {features_test.shape}")

    return label_df_train, label_df_val, label_df_test, features_train, features_val, features_test


def start(cfg: Config):
    # Sanity check
    possible_values_check(cfg)
    print(cfg)

    # =========== Load data ===========
    label_df_train, label_df_val, label_df_test, features_train, features_val, features_test = prepare_data(cfg)

    # =========== Run Matching ===========
    matching_obj = Matching(features_train=features_train, features_val=features_val, features_test=features_test,
                            label_df_train=label_df_train, label_df_val=label_df_val, label_df_test=label_df_test,
                            results_dir=Path(cfg.results_dir), outcome=cfg.data.outcome,
                            subgroup_vars=cfg.data.subgroup_vars,
                            **cfg.matching)

    matching_obj.generate_post_matching_idx(data_flag_list=["train"], direction_list=cfg.direction_list)

    # =========== Evaluate Matching ===========
    matching_eval_obj = MatchingEvaluation(matching_obj=matching_obj, results_dir=Path(cfg.results_dir))
    matching_eval_obj.run_evaluation()

    print(f"End: {datetime.now().time()}")
