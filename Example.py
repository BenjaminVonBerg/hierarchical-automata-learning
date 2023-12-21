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
  for idx in range(1, len(trace) - 1):
    v_p, v_c, v_n = o_trace[idx-1:idx+2]
    falling = v_c < v_p and v_c < v_n
    rising = v_p < v_c and v_n < v_c
    if falling or rising:
      label = r_sym if rising else f_sym
      segment = [o_trace[start_idx]] + trace[start_idx+1:idx+1]
      segments.append(LabeledSegment(label, segment))
      start_idx = idx
  label = r_sym if o_trace[start_idx] < o_trace[-1] else f_sym
  segment = [o_trace[start_idx]] + trace[start_idx+1:]
  segments.append(LabeledSegment(label, segment))

  return segments

# use vanilla IOAlergia as learning algorithm
def learning_alg(data : list[Trace]) :
  return run_Alergia(data, "mdp")

# obtain ground truth and samples
mdp = get_mdp()
samples = get_samples(mdp, 100, 0.1)

# learn models
model_baseline = learning_alg(samples)
model_ours = HierarchicMDP.learn_with_function(samples, segment_rising_falling, learning_alg).to_MDP()

# visualization
os.makedirs("output", exist_ok=True)
os.chdir("output")
file_type = "svg"
mdp.visualize("original", file_type)
model_baseline.visualize("baseline", file_type)
model_ours.visualize("ours", file_type)