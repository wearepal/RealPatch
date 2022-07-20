from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from matching.Matching_class import Matching

RULE_OF_THUMB = {'SMD': ['<= 0.1', '0.1, 0.2', '>= 0.2'],
                 'VR': ['<= 4/5', '4/5, 5/4', '>= 5/4']}

RULE_OF_THUMB_MINMAX = {'SMD': [0.1, 0.2],
                        'VR': [4 / 5, 5 / 4]}


def range_rule_of_thumb(x, _min, _max):
    return [sum([x <= _min for x in x]), sum([_min < x < _max for x in x]), sum([x >= _max for x in x])]


def distribution_statistics(x):
    return np.percentile(x, [25, 50, 75]).tolist() + [np.mean(x), np.var(x)]


def _smd_discrete(p_t, p_c):
    # No difference in means if the prevalence in the two groups is the same
    if p_t == p_c:
        return 0
    num = (p_t - p_c)
    den = np.sqrt((p_t * (1 - p_t) + p_c * (1 - p_c)) / 2)
    return abs(num / den)


def _smd_continuous(mu_t, mu_c, sigma_t, sigma_c):
    num = mu_t - mu_c
    den = np.sqrt((sigma_t + sigma_c) / 2)
    return abs(num / den)


def propensity_score_distributions_comparison(x_t_pre, x_c_pre, x_t_post, x_c_post, treatment="", results_dir=""):
    ps_stack = pd.DataFrame(columns=['PropensityScore', 'Group', 'Col'])
    ps_stack['PropensityScore'] = pd.concat([x_t_pre, x_c_pre, x_t_post, x_c_post], ignore_index=True)
    # Add group (treated or control) and method (pre or post matching)
    ps_stack['Group'] = ['Treated' for x in x_t_pre] + ['Control' for x in x_c_pre] + ['Treated' for x in x_t_post] + [
        'Control' for x in x_c_post]
    ps_stack['Col'] = ['Pre-Matching' for x in x_t_pre] + ['Pre-Matching' for x in x_c_pre] + ['Post-Matching' for x in
                                                                                               x_t_post] + [
                          'Post-Matching' for x in x_c_post]

    sns.set(font_scale=1.5, style="whitegrid")
    g = sns.displot(ps_stack, x="PropensityScore", hue="Group", col='Col', kind="kde", height=7, aspect=1)

    g.set_axis_labels("Propensity Score", "Density")
    g.set_titles(f"{col_name} Propensity Score Distributions")
    plt.subplots_adjust(top=.85)

    plt.savefig(results_dir / f"propensity_score_distributions_comparison_{treatment}.png", dpi=300, facecolor='white',
                transparent=True, bbox_inches='tight')
    plt.close()


def smd(x_t, x_c):
    """
    Compute standardised difference in means of two distributions
    TO-DO add reference paper!
    """
    # Transform to array
    x_t, x_c = np.array(x_t), np.array(x_c)
    # Take mean
    mu_t, mu_c = np.mean(x_t), np.mean(x_c)

    # Check both x_t and x_c are binary arrays
    if np.array_equal(x_t, x_t.astype(bool)) and np.array_equal(x_c, x_c.astype(bool)):
        return _smd_discrete(mu_t, mu_c)

    sigma_t, sigma_c = np.var(x_c), np.var(x_t)
    return _smd_continuous(mu_t, mu_c, sigma_t, sigma_c)


def variance_ratio(x_t, x_c):
    """
    Compute variance ratio of two distributions
    """
    # No difference (vr = 1) if the distribution in the two groups is the same
    if len(x_t) == len(x_c) and (x_t.values == x_c.values).all():
        return 1
    elif np.var(x_c) == 0:
        return 0
    return np.var(x_t) / np.var(x_c)


class MatchingEvaluation:
    def __init__(self,
                 matching_obj: Matching,
                 results_dir: Path = None, ):
        self.matching_obj = matching_obj
        self.results_dir = results_dir / "Evaluation"
        self.results_dir.mkdir(exist_ok=True)

        # Useful in all the evaluations - test is always a possible key
        self.direction_list = list(self.matching_obj.post_matching_idx_by_direction['test'].keys())

    def number_units(self, data_flag: str):
        """
        Number of Units: Pre and Post Matching comparison
        """
        number_units = pd.DataFrame(columns=['Treated', 'Control', 'Matched'],
                                    index=self.matching_obj.subgroup_vars)

        for treatment in self.matching_obj.subgroup_vars:
            number_units.loc[treatment, 'Treated'] = len(
                self.matching_obj.pre_matching_idx[data_flag][treatment]['treated'])
            number_units.loc[treatment, 'Control'] = len(
                self.matching_obj.pre_matching_idx[data_flag][treatment]['control'])

            number_units.loc[treatment, 'Matched'] = len(
                self.matching_obj.post_matching_idx[data_flag][treatment]['real'])
            assert len(self.matching_obj.post_matching_idx[data_flag][treatment]['real']) == len(
                self.matching_obj.post_matching_idx[data_flag][treatment]['matched'])

        number_units_df = number_units.copy()
        number_units_df["Original"] = number_units_df['Treated'] + number_units_df['Control']
        number_units_df['Matched'] = number_units_df['Matched']*2
        number_units_df = number_units_df.drop(['Treated', 'Control'], axis=1)
        number_units_df.to_csv(self.results_dir / f'number_units_{data_flag}.csv', index=True)

    def evaluation_features(self, data_flag: str):
        # Number of Variables (up to 2048 resnet features) with SMD and VR value in certain range
        evaluation_rule_of_thumb_ = {k: pd.DataFrame(
            index=pd.MultiIndex.from_product([self.matching_obj.subgroup_vars, ["Pre-Matching", "Post-Matching"]],
                                             names=["Treatment", k]),
            columns=RULE_OF_THUMB[k])
            for k in ['SMD', 'VR']}

        for treatment in self.matching_obj.subgroup_vars:
            # Needed information for specific treatment variable
            features_full, features_matched, features_matched_by_direction = {}, {}, defaultdict(dict)

            for group_full, group_matched in zip(['treated', 'control'], ['real', 'matched']):
                features_full[group_full] = self.matching_obj.features[data_flag].loc[
                    self.matching_obj.pre_matching_idx[data_flag][treatment][group_full]]
                features_matched[group_matched] = self.matching_obj.features[data_flag].loc[
                    self.matching_obj.post_matching_idx[data_flag][treatment][group_matched]]

                for direction in self.direction_list:
                    features_matched_by_direction[direction][group_matched] = \
                        self.matching_obj.features[data_flag].loc[
                            self.matching_obj.post_matching_idx_by_direction[data_flag][direction][treatment][
                                group_matched]]

            # Standardised Difference in Means, Variance Ratio
            for features_, group, name in zip([features_full, features_matched],
                                              [['treated', 'control'], ['real', 'matched']],
                                              ['Pre-Matching', 'Post-Matching']):
                for metric_name, metric_f in zip(['SMD', 'VR'], [smd, variance_ratio]):
                    metric_temp = [metric_f(features_[group[0]][c], features_[group[1]][c]) for c in
                                   features_[group[0]].columns]
                    evaluation_rule_of_thumb_[metric_name].loc[treatment, name] = range_rule_of_thumb(metric_temp, *
                    RULE_OF_THUMB_MINMAX[metric_name])

        # Write and Save results
        for metric_name in ['SMD', 'VR']:
            evaluation_rule_of_thumb_[metric_name].to_csv(
                self.results_dir / f'evaluation_{metric_name}_rule_of_thumb_features_{data_flag}.csv', index=True)

    def run_evaluation(self):
        self.number_units(data_flag="train")
        self.evaluation_features(data_flag="train")
