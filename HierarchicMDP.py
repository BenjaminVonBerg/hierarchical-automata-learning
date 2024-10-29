from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Callable

from aalpy.automata import Mdp, MdpState

Input = Any
Output = Any
SegmentLabel = Any

IOPair = tuple[Input, Output]
Trace = list[IOPair | Output]

@dataclass(frozen=True)
class OutputPair:
  symbol : Output
  next : SegmentLabel

  def __lt__(self, other):
    return [self.symbol, self.next or ""] < [other.symbol, other.next or ""]

@dataclass
class LabeledSegment:
  label : SegmentLabel
  segment : Trace

SegmentedTrace = list[LabeledSegment]
SegmentationFunction = Callable[[Trace], SegmentedTrace]
LearningAlg = Callable[[list[Trace]], Mdp]

# TODO: add support for different trace structures (data on enter/exit/both, custom entry states)
@dataclass
class HierarchicMDP:
  initial_output : Output
  mdps : dict[SegmentLabel, Mdp]
  initial_prob : dict[SegmentLabel, float]

  @staticmethod
  def learn_with_function(traces : list[Trace], seg_fun : SegmentationFunction, learning_alg : LearningAlg) -> 'HierarchicMDP':
    return HierarchicMDP.learn([seg_fun(trace) for trace in traces], learning_alg)

  @staticmethod
  def learn(seg_traces : list[SegmentedTrace], learning_alg : LearningAlg) -> 'HierarchicMDP':
    """
      passive automata stitching with only a single entry point per cluster with data attached to both exit and entry states
      the algorithm expects that there the provided traces overlap by one sample
    """

    # Get initial output of the whole system
    initial_output = seg_traces[0][0].segment[0]

    # Perform sanity checks on segmentation function and initial output
    for trace in seg_traces:
      if any(c.segment[-1][1] != n.segment[0] for c, n in zip(trace[:-1], trace[1:])):
        raise ValueError("provided traces don't overlap")

      if initial_output != trace[0].segment[0]:
        raise ValueError("Expect initial output of all traces to be identical")

    # Calculate probability of starting with a certain sub-model
    # Note: assuming a unique initial output, we could simply map MDP identifier to their frequency
    # Here we just save the initial output because it nicely extends to multiple initial states (not implemented).
    initial_segment_freq = defaultdict[tuple[SegmentLabel, Output], float](float)
    for trace in seg_traces:
      initial_segment_freq[trace[0].label, trace[0].segment[0]] += 1
    for key in initial_segment_freq.keys():
      initial_segment_freq[key] /= len(seg_traces)

    # Group and preprocess labeled segments from all traces
    def io_pair(in_sym, out_sym=None, next_type=None):
      if out_sym is None:
        out_sym = in_sym
      return (in_sym, OutputPair(out_sym, next_type))

    segment_dict = defaultdict[SegmentLabel, list[Trace]](list)
    for trace in seg_traces:
      for idx, segment in enumerate(trace):
        # placeholder state for having several initial states
        training_trace = [OutputPair(None, None)]
        # entry state is reached with initial output
        training_trace.append(io_pair(segment.segment[0]))
        # main part of the trace
        training_trace.extend(io_pair(*iop) for iop in segment.segment[1:-1])
        # terminal state containing information about next segment
        if 1 < len(segment.segment):
          next_label  = trace[idx+1].label if idx+1 < len(trace) else None
          training_trace.append(io_pair(*segment.segment[-1], next_label))

        segment_dict[segment.label].append(training_trace)

    # Learning an automaton per segment label
    sub_models = dict()
    for start_label, traces in segment_dict.items():
      sub_models[start_label] = learning_alg(traces)

    return HierarchicMDP(initial_output, sub_models, initial_segment_freq)

  def prune_dead_terminal_states(self):
    for mdp in self.mdps.values():
      for state in mdp.states:
        for in_sym, transitions in state.transitions.items():
          # sort terminal states by output and compute probabilistic ratio
          terminal_child_dict = defaultdict(list)
          for idx, (child, _) in enumerate(transitions):
            if child.output.next is not None:
              terminal_child_dict[child.output.symbol].append(idx)
          for children in terminal_child_dict.values():
            total_prob = sum(transitions[idx][1] for idx in children)
            for idx_to_children, idx_to_transitions in enumerate(children):
              children[idx_to_children] = (idx_to_transitions, transitions[idx_to_transitions][1] / total_prob)

          # check for dead duplicates of terminal states and update probabilities of their terminal counterparts
          idxs_to_del = []
          for idx, (child, prob) in enumerate(transitions):
            is_non_terminal = child.output.next is None
            terminal_counterparts = terminal_child_dict.get(child.output.symbol)
            is_dead = len(child.transitions) == 0
            if is_non_terminal and terminal_counterparts is not None and is_dead:
              idxs_to_del.insert(0, idx)
              for counterpart_idx, ratio in terminal_counterparts:
                cp_target, cp_prob = transitions[counterpart_idx]
                transitions[counterpart_idx] = (cp_target, cp_prob + ratio * prob)
          for idx in idxs_to_del:
            del transitions[idx]

  def to_MDP(self, prune_unreachable_entry_states = True, prune_dead_terminal_states = True) -> Mdp:
    if prune_dead_terminal_states:
      self.prune_dead_terminal_states()

    # copy information of mdps
    state_dict : dict[tuple[SegmentLabel, str], MdpState] = dict()
    for name, mdp in self.mdps.items():
      for state in mdp.states:
        if state.output.next in self.mdps.keys():
          continue # ignore terminal states as they are merged with start states
        if state is mdp.initial_state:
          continue # ignore initial state which is only a dummy
        state_copy = MdpState(state.state_id, state.output)
        state_copy.transitions = state.transitions.copy()
        state_dict[(name, state.state_id)] = state_copy

    # fix transitions
    for (name, _), state in state_dict.items():
      for in_sym, transitions in state.transitions.items():
        for idx, (next_state, prob) in enumerate(transitions):
          next_output = next_state.output
          if next_output.next in self.mdps.keys():
            next_mdp = self.mdps[next_output.next]
            next_state = next_mdp.initial_state.transitions[next_output.symbol][0][0]
            next_state = state_dict[(next_output.next, next_state.state_id)]
          else:
            next_state = state_dict[(name, next_state.state_id)]
          transitions[idx] = (next_state, prob)

    # change out structure, make ids unique
    for (name, state_id), state in state_dict.items():
      state.output = state.output.symbol
      state.state_id = name + "_" + state_id

    # create initial state
    initial_state = MdpState("init", self.initial_output)
    initial_transitions = initial_state.transitions
    for (name, output), initial_prob in self.initial_prob.items():
      state = state_dict[(name, self.mdps[name].initial_state.transitions[output][0][0].state_id)]
      for in_sym, sub_transitions in state.transitions.items():
        scaled_transitions = [(state, initial_prob * prob) for state, prob in sub_transitions]
        initial_transitions[in_sym].extend(scaled_transitions)
    for transitions in initial_transitions.values():
      # renormalization is necessary if the initial state of a sub model is not input complete.
      total_prob = sum(prob for _, prob in transitions)
      for idx, (child, prob) in enumerate(transitions):
        transitions[idx] = (child, prob / total_prob)


    # create state list
    states = [initial_state]
    states.extend(state_dict.values())

    # post processing
    if prune_unreachable_entry_states:
      reached = {initial_state.state_id}
      for state in states:
        for trans in state.transitions.values():
          for n_state, _ in trans:
            reached.add(n_state.state_id)
      states = list(filter(lambda state: state.state_id in reached, states))

    return Mdp(initial_state, states)