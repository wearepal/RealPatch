from typing import Dict, List, Optional

import numpy as np
import pandas as pd


class ReWeight:
    def __init__(self,
                 label_df: Dict[str, pd.DataFrame],
                 outcome: str = None,
                 subgroups: List[str] = None,
                 reweight: Optional[str] = None, ):
        self.label_df = label_df
        self.outcome = outcome
        self.subgroup_vars = subgroups

        reweight_allowed_values = [None, 'no_reweight', 'spurious', 'target', 'subgroups']
        if reweight not in reweight_allowed_values:
            raise ValueError(
                f'{reweight} value for reweight argument invalid. It should be one of the following: {reweight_allowed_values}]')
        self.reweight: Optional[str] = reweight

        self.REWEIGHT = {None: self._no_reweight,
                         'no_reweight': self._no_reweight,
                         'spurious': self._reweight_spurious,
                         'target': self._reweight_target,
                         'subgroups': self._reweight_subgroups}

    def _set_idx(self, train_idx: Optional[list] = None, train_val: Optional[bool] = False):
        if train_val:
            train_set = pd.concat([self.label_df["train"], self.label_df["val"]])
            # If considering also validation set I need to add the indexes when considering only a subset of the
            # training set
            if train_idx is not None:
                train_idx += self.label_df["val"].index.tolist()
        else:
            train_set = self.label_df["train"]
        if train_idx is None:
            train_idx = train_set.index
        return train_set, train_idx

    def _no_reweight(self, treatment: str, train_idx: Optional[list] = None, train_val: Optional[bool] = False):
        """
        No re-weight function, all samples have the same weight 1
        Args:
            treatment: treatment variable. Only passed to be consistent with the other reweight functions
            train_idx: indices of the train set, needed for augmented train set
            train_val: Flag whether to use both train and validation sets
        """
        train_set, train_idx = self._set_idx(train_idx, train_val)
        return np.ones(len(train_set.loc[train_idx]))

    def _reweight_spurious(self, treatment: str, train_idx: Optional[list] = None, train_val: Optional[bool] = False):
        """
        Re-weight function without the outcome variable being defined. It only balances the two groups created by the
        treatment variable

        Args:
            treatment: treatment variable
            train_idx: indices of the train set, needed for augmented train set
            train_val: Flag whether to use both train and validation sets
        """
        train_set, train_idx = self._set_idx(train_idx, train_val)

        weights = train_set.loc[train_idx][[treatment]].copy()
        # Loop over the treatment (subgroup) variable values
        for t_v in [0, 1]:
            subgroup_treatment = weights[(weights[treatment] == t_v)]
            weights.loc[subgroup_treatment.index, "weights"] = 1 / (len(subgroup_treatment) * 2 / len(weights))

        return weights["weights"]

    def _reweight_target(self, treatment: str, train_idx: Optional[list] = None, train_val: Optional[bool] = False):
        """
        Re-weight function without the treatemnt variable being defined. It only balances the two groups created by the
        outcome variable

        Args:
            treatment: treatment variable
            train_idx: indices of the train set, needed for augmented train set
            train_val: Flag whether to use both train and validation sets
        """
        train_set, train_idx = self._set_idx(train_idx, train_val)

        weights = train_set.loc[train_idx][[self.outcome]].copy()
        # Loop over the outcome (subgroup) variable values
        for o_v in list(set(train_set.loc[train_idx][self.outcome])):
            subgroup_outcome = weights[(weights[self.outcome] == o_v)]
            weights.loc[subgroup_outcome.index, "weights"] = 1 / (len(subgroup_outcome) * 2 / len(weights))

        return weights["weights"]

    def _reweight_subgroups(self, treatment: str, train_idx: Optional[list] = None, train_val: Optional[bool] = False):
        """
        Re-weight function with the outcome variable. It balances the four subgroups created by the treatment and
        "outcome" variable.
        Re-weight defined as: 1 / (len(subgroup) * 4 / len(train))

        Args:
            treatment: treatment variable
            train_idx: indices of the train set, needed for augmented train set
            train_val: Flag whether to use both train and validation sets
        """
        train_set, train_idx = self._set_idx(train_idx, train_val)

        weights = train_set.loc[train_idx][[treatment, self.outcome]].copy()
        # Loop over outcome (group) variable values
        for o_v in list(set(train_set.loc[train_idx][self.outcome])):
            # Loop over the treatment (subgroup) variable values
            for t_v in [0, 1]:
                subgroup = weights[(weights[treatment] == t_v) & (weights[self.outcome] == o_v)]
                weights.loc[subgroup.index, "weights"] = 1 / (len(subgroup) * 4 / len(weights))

        return weights["weights"]
