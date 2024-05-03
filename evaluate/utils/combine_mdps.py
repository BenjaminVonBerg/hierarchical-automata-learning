import random
from typing import Iterable

import aalpy.utils
from aalpy.automata import Mdp, MdpState

from HierarchicMDP import Trace, SegmentedTrace, LabeledSegment
from evaluate.utils.base import ensure_input_complete, logger


def create_entry_state_output(mdp: Mdp, entry_state: MdpState):
    assert entry_state in mdp.states
    entry_states_mdp = get_entry_states(mdp)
    return f"init{len(entry_states_mdp)}-mdp{get_mdp_index_from_state_label(mdp.states[0])}"


def get_mdp_name_from_entry_state_output(output: str):
    if output.startswith("init"):
        return output.split("-")[-1]
    else:
        return None

def get_mdp_index_from_state_label(state: MdpState) -> int:
    return int(state.state_id.split("mdp")[-1])


def get_entry_states(mdp: Mdp) -> list[MdpState]:
    return [s for s in mdp.states if s.output.startswith("init")]


def make_entry_state(mdp: Mdp, state: MdpState):
    assert state in mdp.states
    if not state.output.startswith("init"):
        state.output = create_entry_state_output(mdp, state)
    else:
        assert False, "State is already an entry state!"


def add_transitions_from_mdp_to_mdp(mdp1: Mdp, mdp2: Mdp, num_transitions: int):
    """ Add transitions from mdp1 to mdp2"""

    finished_transitions = 0
    while finished_transitions < num_transitions:
        # choose random state as leave state (a state from mdp1 from which we transition to mdp2)
        # note: the same state can be chosen multiple times!
        leave_state = random.choice(mdp1.states)

        # choose random transition of this state
        # 1) get one of the inputs
        leave_input = random.choice(list(leave_state.transitions.keys()))
        # 2) get a random transition index and the original transition (for the probability)
        orig_transitions = leave_state.transitions[leave_input]
        index = random.choice(list(range(len(orig_transitions))))
        orig_state, orig_probability = orig_transitions[index]
        if orig_state in mdp2.states:
            continue  # avoid overwriting transitions that already go to mdp2

        # add transition to some state of mdp2 which this state+input doesn't already transition to
        allowed_entry_states = list(set(get_entry_states(mdp2)).difference(s for s, p in leave_state.transitions[leave_input]))
        if not allowed_entry_states:
            assert False, "There should always be available transitions"
        entry_state = random.choice(allowed_entry_states)
        leave_state.transitions[leave_input][index] = (entry_state, orig_probability)
        finished_transitions += 1

def combine_mdps(mdps: Iterable[Mdp], initial: Mdp):
    all_states = []
    for mdp in mdps:
        all_states.extend(mdp.states)
    return Mdp(initial.initial_state, all_states)


def generate_mdp(index, num_states, input_size, output_size):
    mdp = aalpy.utils.generate_random_mdp(num_states, input_size, output_size)
    for state in mdp.states:
        state.state_id = state.state_id + f"_mdp{index}"
    make_entry_state(mdp, mdp.initial_state)
    return mdp


def generate_combined_mdp(num_individual_mdps, num_states_per_individual_mdp, num_inputs, num_outputs, connection_type: str,
                          num_transitions_per_connected_mdps: int = None):
    logger.mdp_creation(f"Generating combined MDP from {num_individual_mdps} individual MDPs with {num_states_per_individual_mdp} states each, "
          f"with {num_inputs} inputs and {num_outputs} outputs, in a '{connection_type}' connection. ")

    if num_transitions_per_connected_mdps is None:
        num_transitions_per_connected_mdps = int(num_states_per_individual_mdp / 5)
        logger.mdp_creation(f"Number of transitions per individual MDP was not specified, defaulting to num_states / 5 = {num_transitions_per_connected_mdps}")
    else:
        logger.mdp_creation(f"{num_transitions_per_connected_mdps} to a connected MDP")

    mdps = []
    for i in range(int(num_individual_mdps)):
        mdp = generate_mdp(index=i, num_states=num_states_per_individual_mdp, input_size=num_inputs, output_size=num_outputs)
        mdps.append(mdp)

    num_entry_states = num_outputs
    for mdp in mdps:
        entry_states = random.sample([s for s in mdp.states if s != mdp.initial_state], k=num_entry_states-1)
        for state in entry_states:
            make_entry_state(mdp, state)

    connections = []
    match connection_type:
        case "one-way":
            for i in range(len(mdps)):
                connections.append((i, (i+1) % len(mdps)))
        case "two-way":
            for i in range(len(mdps)):
                connections.append((i, (i + 1) % len(mdps)))
                connections.append(((i + 1) % len(mdps), i))
        case "complete":
            for i in range(len(mdps)):
                for j in range(len(mdps)):
                    if i != j:
                        connections.append((i, j))

    for src, tgt in connections:
        add_transitions_from_mdp_to_mdp(mdps[src], mdps[tgt], num_transitions_per_connected_mdps)

    combinedmdp = combine_mdps(mdps, mdps[0])

    ensure_input_complete(combinedmdp, "Combined MDP")

    return combinedmdp


def segment_traces_of_combined_mdps(trace: Trace) -> SegmentedTrace:
    """
    Segment traces of combined MDPs by going into a new segment once a 'init*' output appears.
    """

    def output_to_label(output):
        return get_mdp_name_from_entry_state_output(output)

    initial_output = trace[0]
    current_segment = LabeledSegment(output_to_label(initial_output), [initial_output])
    segments = [current_segment]

    for step in trace[1:]:
        input, output = step
        if output_to_label(output) is None or output_to_label(output) == current_segment.label:
            current_segment.segment.append(step)
        else:
            current_segment.segment.append(step)
            current_segment = LabeledSegment(output_to_label(output), [output])
            segments.append(current_segment)

    for segment in segments:
        segment.segment = tuple(segment.segment)

    return segments

# combined = generate_combined_mdp(num_individual_mdps=3,
#                                  num_states_per_individual_mdp=5,
#                                  num_inputs=3,
#                                  num_outputs=3,
#                                  connection_type="one-way",
#                                  entry_states_per_mdp=2)
# combined.visualize(path="combined", file_type='svg')
