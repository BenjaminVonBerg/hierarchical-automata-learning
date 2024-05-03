import csv
import dataclasses
import itertools
import os
from dataclasses import dataclass
from pathlib import Path

import pebble

from evaluate.utils import logger
from evaluate.utils.pickling import cached, base_path
import evaluate.create_plot as plotter

get_statistics = plotter.main

# ----------------
# GENERAL
# ----------------

NUM_RUNS_PER_CONSTELLATION = 1
RESULTS_FILE = base_path / "results.csv"
ALGORITHM = "IOAlergia"

# ----------------
# MDP CREATION
# ----------------

MDP_NUM_STATES_PER_IND = [
    5, 15, 30
]

MDP_NUM_INPUTS = [
    2, 5, 10
]

MDP_NUM_OUTPUTS = [
    2, 5, 10
]

MDP_CONNECTION_TYPES = [
    "one-way",
    "two-way",
    "complete"
]

MDP_NUMBER_INDIVIDUAL_MDPS = [
    2, 3, 4, 5
]

MDP_NUM_TRANS_PER_CONN = [
    1, 5, 10
]

# ----------------
# TRACE GENERATION
# ----------------

NUM_TRACES = [
    100, 300, 900
]

LEN_TRACES = [
    100, 300,
]


@dataclass(frozen=True)
class ResultRow:
    num_individual_mdps: int
    num_states_per_individual_mdp: int
    num_inputs: int
    num_outputs: int
    num_transitions_per_connected_mdps: int
    connection_type: str
    num_traces: int
    len_traces: int
    hal_bisimilarity_mean: float
    hal_bisimilarity_max: float
    hal_compliance: float
    alg_bisimilarity_mean: float
    alg_bisimilarity_max: float
    alg_compliance: float

    def __iter__(self):
        for field in dataclasses.fields(self):
            yield getattr(self, field.name)


def constellation_to_argv_list(num_states_ind, num_inputs, num_outputs, connection_type, num_ind_mdps, num_trans, num_traces, len_traces):
    return [
            "--learning_algorithm", ALGORITHM,
            "--num_runs", str(NUM_RUNS_PER_CONSTELLATION),
            "--num_individual_mdps", str(num_ind_mdps),
            "--num_states_per_individual_mdp", str(num_states_ind),
            "--num_inputs", str(num_inputs),
            "--num_outputs", str(num_outputs),
            "--num_transitions_per_connected_mdps", str(num_trans),
            "--connection_type", connection_type,
            "--num_traces", str(num_traces),
            "--len_traces", str(len_traces),
            "--metrics", "BisimilarityMean", "BisimilarityMax", "ComplianceNotStepNormalized"
    ]

@pebble.synchronized
def write_row(row: ResultRow):
    logger.info("Writing")
    with open(RESULTS_FILE, "a", newline='') as file:
        writer = csv.writer(file)
        writer.writerow(list(row))

def execute_and_save_to_results(constellation):
    num_states_ind, num_inputs, num_outputs, connection_type, num_ind_mdps, num_trans, num_traces, len_traces = constellation
    logger.info("Calculating...")
    results = cached("statistics")(get_statistics)(constellation_to_argv_list(num_states_ind, num_inputs, num_outputs, connection_type, num_ind_mdps, num_trans, num_traces, len_traces), None, False)
    for run in results.values():
        run_results = run[0]  # only one plot step
        result_row = ResultRow(
            num_individual_mdps=num_ind_mdps,
            num_states_per_individual_mdp=num_states_ind,
            num_inputs=num_inputs,
            num_outputs=num_outputs,
            num_transitions_per_connected_mdps=num_trans,
            connection_type=connection_type,
            num_traces=num_traces,
            len_traces=len_traces,
            hal_bisimilarity_mean=run_results["BisimilarityMean"]["HAL"],
            hal_bisimilarity_max=run_results["BisimilarityMax"]["HAL"],
            hal_compliance=run_results["ComplianceNotStepNormalized"]["HAL"],
            alg_bisimilarity_mean=run_results["BisimilarityMean"][ALGORITHM],
            alg_bisimilarity_max=run_results["BisimilarityMax"][ALGORITHM],
            alg_compliance=run_results["ComplianceNotStepNormalized"][ALGORITHM],
        )

        write_row(result_row)


def generate_results():
    logger.info("Starting...")
    create_csv_file()

    all_constellations = list(itertools.product(MDP_NUM_STATES_PER_IND,
                                           MDP_NUM_INPUTS,
                                           MDP_NUM_OUTPUTS,
                                           MDP_CONNECTION_TYPES,
                                           MDP_NUMBER_INDIVIDUAL_MDPS,
                                           MDP_NUM_TRANS_PER_CONN,
                                           NUM_TRACES,
                                           LEN_TRACES))

    def filter_redundant_constellations(constellation):
        num_states_ind, num_inputs, num_outputs, connection_type, num_ind_mdps, num_trans, num_traces, len_traces = constellation

        if num_ind_mdps <= 2 and connection_type in ("two-way", "complete"):
            return False
        if num_ind_mdps <= 3 and connection_type == "complete":
            return False
        if num_states_ind < num_outputs:
            return False

        config_cols = ['num_individual_mdps', 'num_states_per_individual_mdp', 'num_inputs', 'num_outputs',
                       'num_transitions_per_connected_mdps', 'connection_type', 'num_traces', 'len_traces']
        this_config = [num_ind_mdps, num_states_ind, num_inputs, num_outputs,
                       num_trans, connection_type, num_traces, len_traces]

        return True

    filtered_constellations = list(filter(filter_redundant_constellations, all_constellations))
    chosen_constellations = filtered_constellations #random.sample(filtered_constellations, k=200)

    logger.info(f"Working through {len(chosen_constellations)}/{len(all_constellations)} constellations")
    pool = pebble.ProcessPool()
    futures = pool.map(execute_and_save_to_results, chosen_constellations)
    do_it = [f for f in futures.result()]


def create_csv_file():
    Path(RESULTS_FILE).parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_FILE, "w", newline='') as file:
        writer = csv.writer(file)
        writer.writerow(list(ResultRow.__dataclass_fields__.keys()))


if __name__ == "__main__":
    generate_results()
