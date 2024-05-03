import functools
import itertools
import json
import os
import threading
import time
import uuid
from collections import defaultdict
from pprint import pprint
from typing import Optional

import matplotlib.pyplot as plt
import pebble

from HierarchicMDP import HierarchicMDP

from evaluate.utils.pickling import cached, base_path
from evaluate.utils.log import logging
from evaluate.utils.combine_mdps import generate_combined_mdp, segment_traces_of_combined_mdps
from evaluate.utils.base import get_traces_from_mdp, no_printing, save_model, ensure_input_complete, frange
from evaluate.utils.variables import *
logger = logging.getLogger()


PLOTTABLE_VARIABLES = dict(
    # MDP generation
    num_individual_mdps=PlottableVariable(
        "-nm", type=int, default=3,
        description="Number of individual MDPs to combine",
        target=VariableTarget.MDP.Individual,
        exact_description="{value} individual MDPs combined",
    ),
    num_states_per_individual_mdp=PlottableVariable(
        "-ns", type=int, default=5,
        description="Number of states per individual MDPs",
        target=VariableTarget.MDP.Individual,
        exact_description="{value} states per individual MDP",
    ),
    num_inputs=PlottableVariable(
        "-ni", type=int, default=4,
        description="Number of inputs of the combined MDP",
        target=VariableTarget.MDP.Combined,
        exact_description="{value} inputs",
    ),
    num_outputs=PlottableVariable(
        "-no", type=int, default=4,
        description="Number of outputs of the combined MDP",
        target=VariableTarget.MDP.Combined,
        exact_description="{value} outputs",
    ),
    num_transitions_per_connected_mdps=PlottableVariable(
        "-ntrans", type=int, default=None,
        description="Number of transitions as a 'connection' between individual MDPs. ",
        target=VariableTarget.MDP.Individual,
        exact_description="{value} transitions for the 'connection' MDP A -> MDP B",
        only_relevant_if=functools.partial(arg_true, "num_transitions_per_connected_mdps")
    ),

    # Algorithms
    epsilon=PlottableVariable(
        "-eps", type=float, default=0.05,
        description="Epsilon value for IOAlergia algorithm",
        target=VariableTarget.ALGORITHM.IOAlergia,
        exact_description="IOAlergia Epsilon: {value}",
    ),
    alpha=PlottableVariable(
        "-alp", type=float, default=0.05,
        description="Alpha value for LikelihoodRatio algorithm",
        target=VariableTarget.ALGORITHM.LikelihoodRatio,
        exact_description="LikelihoodRatio Alpha: {value}"
    ),

    # Metrics
    discounting_factor_max=PlottableVariable(
        "-disc_max", type=float, default=0.95,
        description="Discounting factor for BisimilarityMax",
        target=VariableTarget.METRIC.BisimilarityMax,
        exact_description="BisimilarityMax discounting factor: {value}",
    ),
    discounting_factor_mean=PlottableVariable(
        "-disc_mean", type=float, default=0.95,
        description="Discounting factor for BisimilarityMean",
        target=VariableTarget.METRIC.BisimilarityMean,
        exact_description="BisimilarityMean discounting factor: {value}",
    ),

    # Traces
    num_traces=PlottableVariable(
        "-nt", type=int, default=200,
        description="Number of traces to learn from",
        target=VariableTarget.TRACES.Generation,
        exact_description="{value} traces to learn from",
    ),
    len_traces=PlottableVariable(
        "-lt", type=int, default=200,
        description="Length of traces to learn from",
        target=VariableTarget.TRACES.Generation,
        exact_description="Traces of length {value}",
        only_relevant_if=functools.partial(arg_false, "uniform_trace_length_distribution")
    ),

)

STATIC_VARIABLES = dict(
    # General
    num_runs=StaticVariable(
        "-nr", type=int, default=10,
        description="Number of runs",
        target=VariableTarget.GENERAL.Evaluation,
        exact_description="Over {value} runs"
    ),
    log_scale=StaticVariable(
        "-ls", type=bool, default=False,
        description="Use logarithmic scaling",
        target=VariableTarget.GENERAL.Evaluation,
        exact_description="Used a logarithmic scale",
        num_choices="flag",
        only_relevant_if=functools.partial(arg_true, "log_scale")
    ),

    # MDPs
    connection_type=StaticVariable(
        "-ct", type=str, default="two-way",
        description="How the individual MDPs are connected.",
        target=VariableTarget.MDP.Combined,
        choices=("one-way", "two-way", "complete"),
        exact_description="Connected in a {value} connection"
    ),

    # Algorithms
    learning_algorithm=StaticVariable(
        "-la", type=str, default='IOAlergia',
        description="Learning algorithm to compare against",
        target=VariableTarget.GENERAL.Learning,
        choices=tuple(a.name for a in LearningAlgorithms),
        exact_description="Learning algorithm: {value}"
    ),
    hyperparameter_search=StaticVariable(
        '-hs', type=bool, default=False,
        description="Perform hyperparameter search at each step",
        target=VariableTarget.GENERAL.Learning,
        num_choices="flag",
        exact_description="Performed hyperparameter search",
        only_relevant_if=functools.partial(arg_true, "hyperparameter_search")
    ),
    hyperparameter_min=StaticVariable(
        "-hmin", type=float, default=0.0,
        description="Minimum value of the algorithm parameter for hyperparameter search.",
        target=VariableTarget.GENERAL.Learning,
        num_choices="single",
        exact_description="Hyperparameter search minimum value: {value}",
        only_relevant_if=functools.partial(arg_true, "hyperparameter_search")
    ),
    hyperparameter_max=StaticVariable(
        "-hmax", type=float, default=0.5,
        description="Maximum value of the algorithm parameter for hyperparameter search.",
        target=VariableTarget.GENERAL.Learning,
        num_choices="single",
        exact_description="Hyperparameter search maximum value: {value}",
        only_relevant_if=functools.partial(arg_true, "hyperparameter_search")
    ),
    hyperparameter_step=StaticVariable(
        "-hstep", type=float, default=0.1,
        description="Step value for the hyperparameter search.",
        target=VariableTarget.GENERAL.Learning,
        num_choices="single",
        exact_description="Hyperparameter search step: {value}",
        only_relevant_if=functools.partial(arg_true, "hyperparameter_search")
    ),
    hyperparameter_log=StaticVariable(
        "-hlog", type=bool, default=False,
        description="Use a logarithmic scale for the hyperparameter search.",
        target=VariableTarget.GENERAL.Learning,
        num_choices="flag",
        exact_description="Hyperparameter search used a log scale",
        only_relevant_if=functools.partial(arg_true, "hyperparameter_log")
    ),

    # Metrics
    metrics=StaticVariable(
        "-m", type=str, default=tuple(m.name for m in Metrics),
        description="Metrics to compare",
        target=VariableTarget.GENERAL.Evaluation,
        choices=tuple(m.name for m in Metrics),
        num_choices="multiple",
        exact_description="Metrics: {value}"
    ),

    # Traces
    uniform_trace_length_distribution=StaticVariable(
        "-uni", type=bool, default=False,
        description="Setting this flag means that the length of traces will be based on a uniform distribution "
                    "between --uniform_min and --uniform_max. --len_traces will be ignored.",
        target=VariableTarget.TRACES.Generation,
        num_choices="flag",
        exact_description="Length of traces to learn from based on a uniform distribution",
        only_relevant_if=functools.partial(arg_true, "uniform_trace_length_distribution")
    ),
    uniform_min=StaticVariable(
        "-umin", type=int, default=100,
        description="Minimum length of traces if --uniform_trace_length_distribution is specified.",
        target=VariableTarget.TRACES.Generation,
        num_choices="single",
        exact_description="Minimum trace length: {value}",
        only_relevant_if=functools.partial(arg_true, "uniform_trace_length_distribution")
    ),
    uniform_max=StaticVariable(
        "-umax", type=int, default=200,
        description="Maximum length of traces if --uniform_trace_length_distribution is specified.",
        target=VariableTarget.TRACES.Generation,
        num_choices="single",
        exact_description="Maximum trace length: {value}",
        only_relevant_if=functools.partial(arg_true, "uniform_trace_length_distribution")
    )

)

# Add name (key) to each variable
ALL_VARIABLES = PLOTTABLE_VARIABLES | STATIC_VARIABLES
for key, var in ALL_VARIABLES.items():
    var.name = key

# sort combined variables by target
def sort_by_target(variable):
    return variable.target._target_order()
ALL_VARIABLES = dict(sorted(ALL_VARIABLES.items(), key=lambda item: sort_by_target(item[1])))

@cached("mdps")
def generate_ground_truth(idx, *args, **kwargs):
    return generate_combined_mdp(*args, **kwargs)

get_traces_from_mdp = cached("traces")(get_traces_from_mdp)

# @concurrent.process
def execute_run_step(run: int, plot_step, algorithm_arg, plot_over: PlottableVariable, args: argparse.Namespace):
    logger.info(f"Starting run {run} plot step {plot_over.name} = {plot_step}") if plot_over else logger.info(f"Starting only plot step")
    # add_prefix_to_print(f"Plot Step {plot_step}: ")

    # CREATE MDP
    creation_kwargs = get_mdp_creation_kwargs(args)
    if var_concerns(plot_over, VariableTarget.MDP):
        creation_kwargs[plot_over.name] = plot_step
    logger.mdp_creation(f"Creating MDP with Parameters: {creation_kwargs}")
    combined_mdp = generate_ground_truth(run, **creation_kwargs)

    # GENERATE TRACES
    traces_generation_kwargs = get_traces_generation_kwargs(args)
    if var_concerns(plot_over, VariableTarget.TRACES):
        traces_generation_kwargs[plot_over.name] = plot_step
    logger.mdp_creation(f"Generating Traces with Parameters: {traces_generation_kwargs}")
    traces = get_traces_from_mdp(combined_mdp, **traces_generation_kwargs)

    # SETUP LEARNING ALGORITHM
    logger.learning("Setting up algorithms")
    selected_algorithm = getattr(LearningAlgorithms, args.learning_algorithm).value
    if var_concerns(plot_over, VariableTarget.ALGORITHM):
        assert not args.hyperparameter_search
        if selected_algorithm == plot_over.target.value:
            algorithm_arg = plot_step
        else:
            raise ValueError(
                f"The plot-over argument concerns the algorithm {plot_over.target.value.__name__}, "
                f"but the selected algorithm is {args.learning_algorithm}")
    if algorithm_arg is not None:
        learning_algorithm = selected_algorithm(algorithm_arg)
    else:
        learning_algorithm = selected_algorithm()

    # LEARN
    logger.learning(f"Learning with {learning_algorithm} ...")
    alg_str = str(learning_algorithm)
    learning_algorithm_baseline = cached("learn_base/"+alg_str)(learning_algorithm)
    learning_algorithm_hal = cached("learn_hal/"+alg_str)(learning_algorithm)
    with no_printing():
        learnt_model_alg = learning_algorithm_baseline(traces)
        learnt_model_hal = HierarchicMDP.learn_with_function(traces, segment_traces_of_combined_mdps,
                                                             learning_algorithm_hal).to_MDP()

    mdp_name = f"mdp{creation_kwargs}_run{run}"
    save_model(combined_mdp, f"models/{mdp_name}/ground_truth")
    for prefix, model in [("base", learnt_model_alg), ("hal", learnt_model_hal)]:
        save_model(model, f"models/{mdp_name}/{alg_str}_{prefix}_sampling{traces_generation_kwargs}")

    ensure_input_complete(learnt_model_hal, "HAL")
    ensure_input_complete(learnt_model_alg, args.learning_algorithm)

    # CALCULATE METRICS
    logger.evaluating("Calculating statistics")
    metric_dict = dict()
    for metric_name in args.metrics:
        metric_kwargs = get_default_metric_kwargs(metric_name)
        if var_concerns(plot_over, VariableTarget.METRIC):
            metric_kwargs[plot_over.name] = plot_step

        metric_type = getattr(Metrics, metric_name).value

        @cached(metric_name)
        def evaluate_metric(ground_truth, model1, model2):
            metric = metric_type(ground_truth, model1, model2, args.learning_algorithm)
            return metric.calculate(**metric_kwargs)
        metric_dict[metric_name] = evaluate_metric(combined_mdp, learnt_model_hal, learnt_model_alg)

    if any(result["HAL"] < result[args.learning_algorithm] for result in metric_dict.values()):
        path = f"worse_models/{uuid.uuid4().hex}"
        for prefix, model in [("gt", combined_mdp), ("base", learnt_model_alg), ("hal", learnt_model_hal)]:
            # save_model(model, f"{path}/{prefix}", file_type="svg")
            save_model(model, f"{path}/{prefix}_model")
        with open(base_path / path / "info.json", "w") as file:
            json.dump(metric_dict, file)

    return metric_dict

def main(argv=None, pool=None, defer_plotting=True):
    parser = argparse.ArgumentParser(description='Run evaluation for hierarchical-automata-learning with random mdps')

    # ADD VARIABLES TO CLI
    for key, var in ALL_VARIABLES.items():
        kwargs = dict(default=var.default,
                      type=var.type,
                      help=var.description)
        if var.choices is not None:
            kwargs["choices"] = var.choices
        if isinstance(var, StaticVariable):
            if var.num_choices == "multiple":
                kwargs["nargs"] = "*"
            if var.num_choices == "flag":
                del kwargs["type"]
                del kwargs["default"]
                if var.default == False:
                    kwargs["action"] = "store_true"
                else:
                    kwargs["action"] = "store_false"

        parser.add_argument(f"--{key}", var.abbreviation, **kwargs)

    # ADD PLOT SETTINGS TO CLI
    parser.add_argument('--plot_over', '-po', choices=PLOTTABLE_VARIABLES,
                        help='The variable to plot over'
                             'If this variable is explicitly set, the set value is ignored.')
    parser.add_argument('--plot_from', '-pf', type=float, default=5)
    parser.add_argument('--plot_to', '-pt', type=float, default=100)
    parser.add_argument('--plot_step', '-ps', type=float, default=5)
    parser.add_argument('--save_plot', '-sp', type=str, default=None)

    # PARSE AND VALIDATE ARGS
    args = parser.parse_args(argv)
    plot_over: PlottableVariable = PLOTTABLE_VARIABLES[args.plot_over] if args.plot_over is not None else None

    # uniform distribution
    if args.uniform_trace_length_distribution and not is_default_value("len_traces", args):
        parser.error("Setting --len_traces and --uniform_trace_length_distribution at the same time is forbidden!")
    elif args.uniform_trace_length_distribution and plot_over.name == "len_traces":
        parser.error("Setting --uniform_trace_length_distribution and plotting over len_traces is forbidden!")
    elif not args.uniform_trace_length_distribution and (not is_default_value("uniform_max", args)
                                                       or not is_default_value("uniform_min", args)):
        parser.error("Setting --uniform_max or --uniform_min without --uniform_trace_length_distribution is forbidden!")

    # hyperparameter search
    if args.hyperparameter_search:
        if not is_default_value("epsilon", args) or not is_default_value("alpha", args) or (plot_over is not None and plot_over.name in ("epsilon", "alpha")):
            parser.error("Hyperparameter search conflicts with explicitly set algorithm arguments")
    elif not args.hyperparameter_search:
        if not is_default_value("hyperparameter_max", args) or not is_default_value("hyperparameter_min", args):
            parser.error("Setting hmax/hmin without hyperparameter search enabled is meaningless")

    # GET AND PRINT DESCRIPTION
    description = get_variables_description(plot_over, args)
    print(f"##############################################\n" +
          description +
          f"##############################################\n")

    # RUN
    def finalize(result_list):
        results_per_run = defaultdict(dict)

        if args.hyperparameter_search:
            for (run_idx, plot_step, algorithm_arg), result in result_list:
                if plot_step in results_per_run[run_idx]:
                    for metric in result:
                        for alg in result[metric]:
                            results_per_run[run_idx][plot_step][metric][alg] = max(results_per_run[run_idx][plot_step][metric][alg], result[metric][alg])
                else:
                    results_per_run[run_idx][plot_step] = result
        else:
            for (run_idx, plot_step, algorithm_arg), result in result_list:
                results_per_run[run_idx][plot_step] = result

        print_results(results_per_run, plot_over, args.metrics, args.learning_algorithm)
        if plot_over:
            plot_results(results_per_run, plot_over, args.metrics, args.learning_algorithm, description, args.save_plot,
                         args.log_scale)
        return results_per_run

    fun = functools.partial(execute_run_step, plot_over=plot_over, args=args)
    fun_args = list(itertools.product(range(args.num_runs), get_plot_step(args), get_algorithm_arg(args)))
    if pool:
        futures = pool.map(fun, *zip(*fun_args))
        def concurrent_finalize():
            return finalize(list(zip(fun_args, futures.result())))

        if defer_plotting:
            # use non-interactive backend
            plt.switch_backend("AGG")
            threading.Thread(target=concurrent_finalize).start()
        else:
            return concurrent_finalize()
    else:
        tmp_results = [(fun_arg, fun(*fun_arg)) for fun_arg in fun_args]
        return finalize(tmp_results)


def get_mdp_creation_kwargs(args: argparse.Namespace):
    creation_kwargs = dict()
    for key, value in vars(args).items():
        if key not in ALL_VARIABLES:
            continue
        if ALL_VARIABLES[key].target in (VariableTarget.MDP.Combined, VariableTarget.MDP.Individual):
            creation_kwargs[key] = value
    return creation_kwargs


def get_traces_generation_kwargs(args: argparse.Namespace):
    generation_kwargs = dict()
    for key, value in vars(args).items():
        if key not in ALL_VARIABLES:
            continue
        if ALL_VARIABLES[key].target in (VariableTarget.TRACES.Generation,):
            generation_kwargs[key] = value
    return generation_kwargs


def get_default_algorithm_arg(selected_algorithm: type):
    for key, var in PLOTTABLE_VARIABLES.items():
        if var.target.value == selected_algorithm:
            return var.default


def get_default_metric_kwargs(metric_name: str):
    kwargs = dict()
    for key, var in ALL_VARIABLES.items():
        if var.target.name == metric_name:
            kwargs[key] = var.default
    return kwargs


def get_plot_step(args: argparse.Namespace):
    if args.plot_over is not None:
        yield from frange(args.plot_from, args.plot_to, args.plot_step, args.log_scale)
    else:
        yield 0


def get_algorithm_arg(args: argparse.Namespace):
    if args.hyperparameter_search:
        assert args.plot_over not in ("epsilon", "alpha")
        yield from frange(args.hyperparameter_min, args.hyperparameter_max, args.hyperparameter_step, args.hyperparameter_log)
    else:
        selected_algorithm = getattr(LearningAlgorithms, args.learning_algorithm).value
        if selected_algorithm == IOAlergia:
            yield args.epsilon
        elif selected_algorithm == LikelihoodRatio:
            yield args.alpha
        elif selected_algorithm == AkaikeInformationCriterion:
            yield None



def var_concerns(var: PlottableVariable, target: TargetEnum | type[TargetEnum]) -> bool:
    if var is None:
        return False

    if isinstance(target, type(TargetEnum)):
        targets = [t for t in target]
    else:
        targets = [target]

    return var.target in targets


def is_default_value(var_name: str, args: argparse.Namespace) -> bool:
    """ Whether the variable var_name has the default value in args """
    return ALL_VARIABLES[var_name].default == getattr(args, var_name)


Run = int
PlotStep = float | int
MetricName = str
AlgorithmName = str
ResultsDict = dict[Run, dict[PlotStep, dict[MetricName, dict[AlgorithmName, float]]]]


def print_results(results: ResultsDict, plot_over: PlottableVariable, metrics: list[str], algorithm_name: str):
    pprint(results)
    # table = prettytable.PrettyTable(["System", "Metric", "HAL", algorithm_name] )
    #
    # for run in results:
    #     for i, plot_step in enumerate(results[run]):
    #         first = i == 0
    #         last = i == len(results[run]) - 1
    #         row = [f"Run {run}" if first else "", algorithm]
    #         for metric in metric_names:
    #             row.append(round(results[system_name][metric][algorithm],2))
    #         table.add_row(row, divider=last)

    # print(table)


def plot_results(results: ResultsDict, plot_over: PlottableVariable, metrics: list[str], algorithm_name: str,
                 description: str, save_plot: Optional[str], logarithmic_x_axis: bool):
    COLORS = {"HAL": "blue",
              "ALG": "orange"}
    num_runs = len(list(results.keys()))
    alpha = 1 / num_runs

    label_added = False

    dpi = 96

    def create_plot_layout():
        match len(metrics):
            case 1:
                return [["text", metrics[0]]]
            case 2:
                return [[metrics[0], metrics[1]],
                        ["text",     "text"]]
            case 3:
                return [["text",     metrics[0]],
                        [metrics[1], metrics[2]]]
            case 4:
                return [[metrics[0], metrics[1]],
                        [metrics[2], metrics[3]],
                        ["text", "text"]]

    def get_figsize(plot_layout):
        # i want one plot to be 300x400
        row_len = len(plot_layout[0])
        col_len = len(plot_layout)
        x = row_len * 400/dpi
        y = col_len * 300/dpi
        return (x, y)


    plot_layout = create_plot_layout()
    figsize = get_figsize(plot_layout)

    x_values = list(results[0].keys())  # all plot_steps
    fig, axd = plt.subplot_mosaic(plot_layout, sharey=True, layout="constrained", figsize=figsize)
    for run in results:
        y_values = {m: dict() for m in metrics}
        for plot_step in results[run]:
            for metric_name in results[run][plot_step]:
                for algorithm_name in results[run][plot_step][metric_name]:
                    if algorithm_name not in y_values[metric_name]:
                        y_values[metric_name][algorithm_name] = []
                    y_values[metric_name][algorithm_name].append(results[run][plot_step][metric_name][algorithm_name])

        for metric_name in metrics:
            y_values_hal = [round(r, 2) for r in y_values[metric_name]["HAL"]]
            y_values_alg = [round(r, 2) for r in y_values[metric_name][algorithm_name]]

            plot = axd[metric_name].semilogx if logarithmic_x_axis else axd[metric_name].plot
            plot(x_values, y_values_hal, marker="o", alpha=alpha, color=COLORS["HAL"], label="HAL" if not label_added else None)
            plot(x_values, y_values_alg, marker="o", alpha=alpha, color=COLORS["ALG"], label=algorithm_name if not label_added else None)
            label_added = True
            axd[metric_name].set_ylabel(metric_name)
            axd[metric_name].set_ylim(0, 1)
            axd[metric_name].set_xlabel(plot_over.description)

    axd["text"].text(0, 1, description.expandtabs(),
                     verticalalignment="top",
                     horizontalalignment='left',
                     fontsize='small',
                     wrap=True,
                     transform=axd["text"].transAxes)
    axd["text"].set_axis_off()

    fig.legend(loc='outside upper right')

    if save_plot is None:
        plt.show()
    else:
        if not os.path.isabs(save_plot):
            dirs, filename = os.path.split(save_plot)
            os.makedirs(dirs, exist_ok=True)
        logger.info(f"Saving plot to {save_plot}")
        plt.savefig(save_plot)
        plt.close()
    # table = prettytable.PrettyTable()
    # table.title = metric_name
    # table.field_names = [x_axis_name, "HAL", algorithm_name]
    # for key, hal, alg in zip(x_values, y_values_hal, y_values_alg):
    #     table.add_row([key, hal, alg])
    # print(table)

def does_not_concern_current_args(var: PlottableVariable, args: argparse.Namespace):
    if var_concerns(var, VariableTarget.ALGORITHM) and args.learning_algorithm != var.target.name:
        return True

    if var_concerns(var, VariableTarget.METRIC) and var.target.name not in args.metrics:
        return True

    return False


def get_variables_description(plot_over: Optional[PlottableVariable], args: argparse.Namespace):
    description = ""
    current_target = None
    for var in ALL_VARIABLES.values():
        if var == plot_over:
            continue
        if does_not_concern_current_args(var, args):
            continue
        if var.only_relevant_if is not None and not var.only_relevant_if(args):
            continue
        if var.target._description() != current_target:
            current_target = var.target._description()
            description += current_target + ":\n"

        description += "\t" + var.exact_description.format(value=getattr(args, var.name)) + "\n"

    return description


if __name__ == "__main__":
    start_time = time.time()
    parallelized = False
    if parallelized:
        pool = pebble.ProcessPool()
    else:
        pool = None
    main(pool=pool, defer_plotting=False)
    end_time = time.time()
    execution_time = round(end_time - start_time, 1)
    logger.info(f"Execution time: {execution_time} seconds")
