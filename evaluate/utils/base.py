import dataclasses
import logging
import os
import random
from contextlib import contextmanager, redirect_stdout
from typing import Any

import numpy as np
from aalpy import save_automaton_to_file
from aalpy.automata import Mdp

from HierarchicMDP import Trace
from evaluate.utils.log import logger
from evaluate.utils.pickling import cached, base_path

date_format = "%Y%m%d_%H%M%S"

@contextmanager
def no_printing():
    with open(os.devnull, 'w') as null_file:
        with redirect_stdout(null_file):
            yield


def frange(start, stop, step, log_scale=False, round_to="auto"):
    """Generate a sequence of floating-point numbers."""
    if log_scale and step <= 1:
        raise ValueError("Step must be greater than 1 when using logarithmic scaling.")

    while start < stop:
        if round_to is not None:
            r2 = 2 - int(np.log10(start if log_scale else step)) if round_to == "auto" else round_to
            start = round(start, r2)
        yield start
        if log_scale:
            start *= step
        else:
            start += step


def ensure_input_complete(mdp: Mdp, algorithm_name: str):
    if not mdp.is_input_complete():
        logger.debug(f"MDP from {algorithm_name} was not input complete!")
        mdp.make_input_complete()
    return mdp


def save_model(automaton, path, file_type="string"):
    path = base_path / path
    path.parent.mkdir(parents=True, exist_ok=True)
    dot_string = save_automaton_to_file(automaton, path=path, file_type=file_type)
    if dot_string:
        with open(path, "w") as file:
            file.write(dot_string)

@cached("traces")
def get_traces_from_mdp(mdp: Mdp, num_traces: int, len_traces: int, uniform_trace_length_distribution: bool = False,
                        uniform_min: int = 100, uniform_max: int = 200) -> list[Trace]:
    """
    Get a list of traces from a MDP.
    :param mdp          instance of an aalpy.automata.Mdp
    """

    if uniform_trace_length_distribution:
        length_callable = lambda: round(random.uniform(uniform_min, uniform_max))
    else:
        length_callable = lambda: len_traces

    traces = []
    for _ in range(int(num_traces)):
        mdp.reset_to_initial()
        trace = [mdp.current_state.output]
        length = length_callable()
        while len(trace) < length:
            in_sym = random.choice(mdp.get_input_alphabet())
            trace.append((in_sym, mdp.step(in_sym)))
        traces.append(trace)

    return traces

def print_table(data, level=logging.INFO):

    def format_numbers(data):
        formatted_data = []
        for row in data:
            formatted_row = [f'{cell:.2f}' if isinstance(cell, float) else cell for cell in row]
            formatted_data.append(formatted_row)
        return formatted_data

    data = format_numbers(data)

    try:
        from prettytable import PrettyTable
        table = PrettyTable()
        table.field_names = data[0]
        for row in data[1:]:
            table.add_row(row)
        logger.log(level, "\n" + str(table))
    except ImportError:
        logger.info("prettytable not installed. Falling back to manual printing. For prettier output, "
              "please install prettytable (pip install prettytable)")
        max_widths = [max(len(str(row[i])) for row in data) for i in range(len(data[0]))]
        separator_line = "+-" + "-+-".join(["-" * width for width in max_widths]) + "-+"
        print(separator_line)
        print("| " + " | ".join(str(header).ljust(max_widths[i]) for i, header in enumerate(data[0])) + " |")
        print(separator_line)
        for row in data[1:]:
            print("| " + " | ".join(str(cell).ljust(max_widths[i]) for i, cell in enumerate(row)) + " |")
        print(separator_line)


def is_deterministically_labeled(mdp: Mdp) -> bool:
    """
    Check whether the given MDP is deterministically labeled
    (i.e., for each combination of state and input, the resulting output uniquely
    identifies the resulting state
    """

    for state in mdp.states:
        for input_symbol, next_states in state.transitions.items():
            next_outputs = [n.output for (n, p) in next_states]
            if len(next_outputs) != len(set(next_outputs)):
                # print(f"{id(mdp)} is not deterministically labeled! Possible next outputs "
                #       f"from state {state.state_id} with input {input_symbol} are {next_outputs}")
                return False
    return True

import contextlib
import sys

class DummyFile(object):
    def write(self, x): pass

@contextlib.contextmanager
def nostdout():
    save_stdout = sys.stdout
    sys.stdout = DummyFile()
    yield
    sys.stdout = save_stdout


def get_non_default_attributes(dataclass_instance: Any) -> dict:
    non_default_attributes = {}
    for field in dataclasses.fields(dataclass_instance):
        if getattr(dataclass_instance, field.name) != field.default:
            non_default_attributes[field.name] = getattr(dataclass_instance, field.name)
    return non_default_attributes

