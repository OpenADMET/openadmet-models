

def _make_stat_caption(
        data:dict,
        task_name:str,
        metric_names:list,
        metrics:dict,
        bootstrap_confidence_level:float,
        evaluated:bool,
        cv:bool
) -> str:
    """A function to generate the stat caption (string) to be printed on the regplot

    Parameters
    ----------
    data : dict
        a dict of task_names, metric_names, metrics, bootstrap_confidence_level, evaluated parameters filled out after evaluating the regression model
    task_names : list or str
        a list of all the tasks, i.e. the predicted target columns, OR str for single task
    metric_names : dict
        a dict of the names of the metrics
    metrics : dict
        a dict of tuples of (metric value, whether the value is a scipy statistic, name of metric to use in the report)
    bootstrap_confidence_level : float
        conficence level for the bootstrap
    evaluated : bool
        whether or not the model has been evaluated
    cv : bool
        whether or not you are doing cross validation for evaluation

    Returns
    -------
    str
        a string containing all the formatted metrics and their labels to be printed on a plot

    Raises
    ------
    ValueError
        cannot make stat caption unless the model has been evaluated first
    """
    if not evaluated:
        raise ValueError("Must evaluate before making a caption")
    stat_caption = ""

    value_key = "mean" if cv else "value"

    stat_caption += f"## {task_name} ##\n"
    for metric in metric_names:
        value = data[task_name][metric][value_key]
        lower_ci = data[task_name][metric]["lower_ci"]
        upper_ci = data[task_name][metric]["upper_ci"]
        # confidence_level = data[task_name][metric]["confidence_level"]
        stat_caption += f"{metrics[metric][2]}: {value:.2f}$_{{{lower_ci:.2f}}}^{{{upper_ci:.2f}}}$\n"
    stat_caption += "\n"
    stat_caption += f"Confidence level: {bootstrap_confidence_level} \n"
    return stat_caption

def _make_stat_dict(
        data:dict,
        task_name:str,
        metric_names:list,
        metrics:dict,
        bootstrap_confidence_level:float,
        evaluated:bool,
        cv:bool
):
    """A function to generate a dict of formatted metrics and names to be printed into a table on regplot

    Parameters
    ----------
    data : dict
        a dict of task_names, metric_names, metrics, bootstrap_confidence_level, evaluated parameters filled out after evaluating the regression model
    task_names : list or str
        a list of all the tasks, i.e. the predicted target columns, OR str for single task
    metric_names : dict
        a dict of the names of the metrics
    metrics : dict
        a dict of tuples of (metric value, whether the value is a scipy statistic, name of metric to use in the report)
    bootstrap_confidence_level : float
        conficence level for the bootstrap
    evaluated : bool
        whether or not the model has been evaluated

    Returns
    -------
    dict
        a dict containing all the formatted metrics and their labels to be printed on a plot

    Raises
    ------
    ValueError
        cannot make stat caption unless the model has been evaluated first
    """
    if not evaluated:
        raise ValueError("Must evaluate before making a caption")

    stat_dict = {
        "metrics": [],
        "means": [],
        "lower_ci": [],
        "upper_ci": [],
        "conf_level": None,
        "task_name" : task_name
    }

    value_key = "mean" if cv else "value"

    for metric in metric_names:
        value = data[task_name][metric][value_key]
        lower_ci = data[task_name][metric]["lower_ci"]
        upper_ci = data[task_name][metric]["upper_ci"]
        # confidence_level = data[task_name][metric]["confidence_level"]

        # Save in dict
        stat_dict["metrics"].append(metrics[metric][2])
        stat_dict["means"].append(float(value))
        stat_dict["lower_ci"].append(float(lower_ci))
        stat_dict["upper_ci"].append(float(upper_ci))
        stat_dict["conf_level"] = bootstrap_confidence_level
    return stat_dict