from typing import Sequence

from HierarchicMDP import *
import random, os

from aalpy.learning_algs.stochastic_passive.Alergia import run_Alergia

# generate a very simple MDP with 4 states
def get_mdp():
  in_sym = "i"

  q_low = MdpState("q0", 0)
  q_rise = MdpState("q1", 1)
  q_high = MdpState("q3", 2)
  q_fall = MdpState("q2", 1)

  q_low.transitions[in_sym] = [(q_rise, 1)]
  q_rise.transitions[in_sym] = [(q_rise, 0.8), (q_high, 0.2)]
  q_high.transitions[in_sym] = [(q_fall, 1)]
  q_fall.transitions[in_sym] = [(q_fall, 0.2), (q_low, 0.8)]

  return Mdp(q_low, [q_low, q_rise, q_high, q_fall])

# draw samples a number of samples from an MDP with a fixed probability of terminating each step
def get_samples(mdp : Mdp, nr, term_prob):
  alphabet = mdp.get_input_alphabet()
  samples = []
  for _ in range(nr):
    mdp.reset_to_initial()
    trace = [mdp.current_state.output]
    while term_prob < random.random():
      in_sym = random.choice(alphabet)
      trace.append((in_sym, mdp.step(in_sym)))
    samples.append(trace)
  return samples

# naive implementation of a segmentation criterion based on extrema
def segment_rising_falling(trace : Trace) -> SegmentedTrace:
  o_trace = [entry[1] if isinstance(entry, Sequence) else entry for entry in trace]
  segments = []

  r_sym = "rising"
  f_sym = "falling"

  start_idx = 0
  current_mode = None
  for idx in range(1, len(trace)):
    v_p, v_c = o_trace[idx-1:idx+1]
    next_mode = None
    if v_p > v_c:
      next_mode = f_sym
    if v_p < v_c:
      next_mode = r_sym

    if next_mode is None or current_mode == next_mode:
      continue
    if current_mode is None:
      current_mode = next_mode
      continue

    segment = [o_trace[start_idx]] + trace[start_idx+1:idx+1]
    segments.append(LabeledSegment(current_mode, segment))
    start_idx = idx
    current_mode = next_mode
  if current_mode is None:
    current_mode = r_sym
  segment = [o_trace[start_idx]] + trace[start_idx+1:]
  segments.append(LabeledSegment(current_mode, segment))

  return segments

# use vanilla IOAlergia as learning algorithm
def learning_alg(data : list[Trace]) :
  return run_Alergia(data, "mdp")

# obtain ground truth and samples
mdp = get_mdp()
samples = get_samples(mdp, 100, 0.1)

# learn models
model_baseline = learning_alg(samples)
hmdp = HierarchicMDP.learn_with_function(samples, segment_rising_falling, learning_alg)
model_ours = hmdp.to_MDP()

# visualization
os.makedirs("output", exist_ok=True)
os.chdir("output")
file_type = "svg"
mdp.visualize("original", file_type)
model_baseline.visualize("baseline", file_type)
model_ours.visualize("ours", file_type)