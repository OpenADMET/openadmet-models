import openadmet_models
import json
import pandas as pd
from openadmet_models.comparison import ComparisonBase, comparisons

@comparisons.register("PostHoc")
class PostHocComparison(ComparisonBase):

    _metrics_df : pd.DataFrame = {}


    @property
    def methods(self):
        pass


    def levene_test():
        pass

    def normality_test():
        pass

    def normality_plots():
        pass

    def anova():
        pass

    def anova_plots():
        pass

    def tukey_hsd():
        pass

    def mcs_plots():
        pass

    def mean_diff_plots():
        pass

    def report():
        pass

    def write_report():
        pass
