import datetime
import itertools
from collections import defaultdict
from pathlib import Path
from typing import List, Optional, Union

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, pairwise_distances

from model_utils import ReWeight
from project_utils import filter_df


class Matching(ReWeight):
    def __init__(self,
                 features_train: pd.DataFrame,
                 features_val: pd.DataFrame,
                 features_test: pd.DataFrame,
                 label_df_train: pd.DataFrame,
                 label_df_val: pd.DataFrame,
                 label_df_test: pd.DataFrame,
                 results_dir: Path = None,
                 fixed_caliper_interval: List[float] = (.1, 0.9),
                 std_caliper: float = .2,
                 outcome: str = None,
                 subgroup_vars: List[str] = None,
                 reweight: Optional[str] = None,
                 temperature: Optional[float] = 1.,
                 ):
        """
        Args:
            features_train: resnet features train set
            features_test: resnet features test set
            label_df_train: attribute labels train set
            label_df_test: attribute labels test set
            # treatment_list: list of treatment variables. Outcome variable is not included
            results_dir: saving directory
            # seed: numpy random seed - WHERE IS IT USED?
            flag_propensity_score: flag whether to use the propensity score for matching or not (Euclidean distance)
            fixed_caliper_interval: interval (min, max) values to use if using fixed caliper
            std_caliper: alpha value to use if using std/distance caliper
            subgroup_vars: dictionary defining for each treatment variable the variable defining its subgroup
            reweight:TO-DO/CHECK IF NEEDED
        """
        self.features: dict = {"train": features_train,
                               "val": features_val,
                               "test": features_test}

        self.label_df: dict = {"train": label_df_train.loc[features_train.index],
                               "val": label_df_val.loc[features_val.index],
                               "test": label_df_test.loc[features_test.index]}

        # Inherit all the methods and properties from its parent ReWeight
        super().__init__(self.label_df, outcome, subgroup_vars, reweight)

        self.results_dir: Path = Path(results_dir)

        # self.flag_propensity_score: bool = flag_propensity_score
        self.fixed_caliper_interval = fixed_caliper_interval

        self.std_caliper: float = std_caliper
        self.temperature: float = temperature

        self.pre_matching_idx = {}
        self.propensity_score = {}
        self._generate_init()

        self.post_matching_idx = {}
        self.post_matching_idx_by_direction = defaultdict(dict)

    def _generate_init(self):
        """
        Run generate_pre_matching_idx and generate_propensity_score
        """
        for data_flag in ["train", "val", "test"]:
            self.pre_matching_idx[data_flag] = self.generate_pre_matching_idx(data_flag=data_flag)

        self.propensity_score = self.generate_propensity_score()

    def generate_pre_matching_idx(self, data_flag) -> dict:
        """
        Pre-Matching: For each treatment, save (filter) all indices of units in treatments and control group

        Args:
            data_flag: whether to get the pre-matching idx of train or test set
        """
        data = self.label_df[data_flag]

        pre_matching_idx = {}
        for treatment in self.subgroup_vars:
            _, treated_idx = filter_df(data, treatment, 1, return_index=True)
            _, control_idx = filter_df(data, treatment, 0, return_index=True)
            pre_matching_idx[treatment] = {'treated': treated_idx, 'control': control_idx}

        return pre_matching_idx

    def temperature_scaling(self, prob_clf_positiveclass):
        logit_clf_positiveclass = np.log(prob_clf_positiveclass / (1 - prob_clf_positiveclass))
        return 1 / (1 + np.exp(-logit_clf_positiveclass / self.temperature))

    def generate_propensity_score(self):
        """
        Generate the propensity score predicting each treatment variable independently training a logistic
        regression on ResNet features or dataset labels.
        """

        # Init result variables
        # Propensity score for each unit-treatment
        propensity_score = {data_flag: pd.DataFrame(columns=self.subgroup_vars, index=self.label_df[data_flag].index)
                            for data_flag in ["train", "val", "test"]}

        # Propensity score Accuracy on test set for each treatment
        propensity_score_accuracy = {data_flag: {} for data_flag in ["train", "val", "test"]}

        X = {data_flag: self.features[data_flag] for data_flag in ["train", "val", "test"]}

        for treatment in self.subgroup_vars:
            print(f'Computing Propensity Score for {treatment}: {datetime.datetime.now().time()}')
            y = {data_flag: self.label_df[data_flag][treatment] for data_flag in ["train", "val", "test"]}

            clf = LogisticRegression(solver='liblinear', random_state=1, tol=1e-12, max_iter=1000, C=1)

            # Fit on training set
            clf.fit(X["train"], y["train"], sample_weight=self.REWEIGHT[self.reweight](treatment, train_val=False))

            for data_flag in ["train", "val", "test"]:
                prob_clf = clf.predict_proba(X[data_flag])
                y_hat = clf.predict(X[data_flag])
                # Probability of treatment = 1 - here correct (specified with clf.classes_.tolist().index(1))
                propensity_score[data_flag][treatment] = list(prob_clf[:, clf.classes_.tolist().index(1)])
                # Temperature adjustment
                if self.temperature != 1:
                    print("Actually rescaling with temperature")
                    propensity_score[data_flag][treatment] = self.temperature_scaling(
                        propensity_score[data_flag][treatment])

                propensity_score_accuracy[data_flag][treatment] = accuracy_score(y[data_flag], y_hat)

                print(
                    f'Accuracy score on {data_flag}: {propensity_score_accuracy[data_flag][treatment]}: Time: {datetime.datetime.now().time()}')

        return propensity_score

    def not_nan_idx(self, matching_idx, group_possible_idx, treatment, treatment_group, data_flag):
        """
        Treatment or controls unit indices after applying matching. It removes the not-possible-to-match units.

        Args:
            matching_idx: sorted list with the corresponding matched (control) unit for each treated unit
            group_possible_idx: treated units corresponding to matching_idx
            treatment: treatment variable
            treatment_group: flag, whether we are working in the treatment (True) or control (False) group
            data_flag: whether we are want to compute it on train or test set
        """
        if treatment_group:
            # After using the sd-based caliper, only take the (position of) not-deleted units in the treatment group
            idx_position = [e for e, x in enumerate(matching_idx) if x == x]
        else:
            # After using the sd-based caliper, only take the (position of) not-deleted units in the control group
            idx_position = [x for e, x in enumerate(matching_idx) if x == x]

        # Take the index (image id) corresponding to the correct positions
        idx_ = self.propensity_score[data_flag].loc[group_possible_idx, treatment].index.to_numpy()[
            idx_position].tolist()
        return idx_

    def generate_post_matching_idx_single(self, data_flag, direction):
        """
        Matching on Propensity Score, compute distances, apply required calipers and return matching pairs.
        Args:
            data_flag: whether we are want to compute it on train or test set
            direction: only one way matching, i.e. either treatment -> control (direction=True) or control -> treatment (direction=False)
        """
        # Post-Matching: Save matched units in treatments and control group
        post_matching_idx = {}

        pre_matching_idx_ = self.pre_matching_idx[data_flag]
        propensity_score_ = self.propensity_score[data_flag]
        features_ = self.features[data_flag]

        for treatment in self.subgroup_vars:
            # Specify the 'direction' of matching
            # From treatment to control group
            if direction == "tc":
                # Needed information for specific treatment variable
                treated_idx, control_idx = pre_matching_idx_[treatment]['treated'], \
                                           pre_matching_idx_[treatment][
                                               'control']
            # From control to treatment group
            elif direction == "ct":
                treated_idx, control_idx = pre_matching_idx_[treatment]['control'], \
                                           pre_matching_idx_[treatment][
                                               'treated']
            else:
                raise NameError(f'Argument {direction} non valid.')

            # Apply the 'fixed caliper'
            possible_to_match_fixed_c = propensity_score_[
                propensity_score_[treatment].between(self.fixed_caliper_interval[0],
                                                     self.fixed_caliper_interval[1],
                                                     inclusive=False)].index.tolist()

            treated_idx_fixed_c = [x for x in treated_idx if x in possible_to_match_fixed_c]
            control_idx_fixed_c = control_idx

            # Define std-caliper
            # Compute Propensity Score standard deviation; needed for the std-distance caliper
            var_treated_ps = np.var(propensity_score_.loc[treated_idx_fixed_c, treatment])
            var_control_ps = np.var(propensity_score_.loc[control_idx_fixed_c, treatment])
            std_ps = np.sqrt((var_treated_ps + var_control_ps) / 2)

            # Compute Euclidean distances for each treatment-control pair
            distances_ps = pairwise_distances(
                propensity_score_.loc[treated_idx_fixed_c, treatment].values.reshape(-1, 1),
                propensity_score_.loc[control_idx_fixed_c, treatment].values.reshape(-1, 1), metric="euclidean")

            # If the std_caliper is defined (not None), use it do define a threshold on the propensity score distance
            if self.std_caliper:
                std_caliper_t = self.std_caliper * std_ps
            # Otherwise, do not eliminate possible treatment units based on it; N.B. all the distances are <= than
            # the overall max distance
            else:
                std_caliper_t = max(np.max(distances_ps, axis=0))

            distances = pairwise_distances(features_.loc[treated_idx_fixed_c],
                                           features_.loc[control_idx_fixed_c], metric="euclidean")
            # Propensity score caliper for resnet distances matching. Distance (Euclidean on ResNet) set
            # to inf if the propensity score distance is above the threshold
            distances[distances_ps > std_caliper_t] = np.inf
            matching_idx = [np.random.choice(np.where(x == min(x))[0]) if min(x) != np.inf else np.nan for x in
                            distances]

            # Treatment units idx after applying matching
            treated_idx_dist_c = self.not_nan_idx(matching_idx, treated_idx_fixed_c, treatment,
                                                  treatment_group=True, data_flag=data_flag)
            # Corresponding control units idx after applying matching
            matched_idx_dist_c = self.not_nan_idx(matching_idx, control_idx_fixed_c, treatment,
                                                  treatment_group=False, data_flag=data_flag)

            # Save idx after matching
            # As we can define the direction, it is more correct to have real/matched rather than treated/control
            post_matching_idx[treatment] = {'real': treated_idx_dist_c, 'matched': matched_idx_dist_c}

        self.post_matching_idx_by_direction[data_flag][direction] = post_matching_idx

    def generate_post_matching_idx(self, data_flag_list: Union[List[str], str], direction_list: Union[List[str], str]):
        if isinstance(data_flag_list, str):
            data_flag_list = [data_flag_list]
        if isinstance(direction_list, str):
            direction_list = [direction_list]

        for data_flag in data_flag_list:
            for direction in direction_list:
                self.generate_post_matching_idx_single(data_flag, direction)

            # Union of the two directions if both are give - bidirectional matching
            if len(direction_list) > 1:
                post_matching_idx_flat = {t: {
                    g: list(itertools.chain.from_iterable([self.post_matching_idx_by_direction[data_flag]["tc"][t][g],
                                                           self.post_matching_idx_by_direction[data_flag]["ct"][t][g]]))
                    for g in ['real', 'matched']}
                    for t in self.subgroup_vars}
            elif len(direction_list) == 1:
                post_matching_idx_flat = self.post_matching_idx_by_direction[data_flag][direction]

            self.post_matching_idx[data_flag] = post_matching_idx_flat
