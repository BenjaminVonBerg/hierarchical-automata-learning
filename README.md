# Hierarchical Learning of Generative Automaton Models from Sequential Data

This repository provides an implementation of the algorithm described in the paper "Hierarchical Learning of Generative Automaton Models from Sequential Data". The method can be used to aid passive learning algorithms for Markov decision processes (MDPs), such as IOAlergia, using domain knowledge and heuristics about system modes. 

## Setup

The code in this repository has been requires python version 3.10 or greater. Requirements are listed in `requirements.txt` and can be installed using
```sh
pip install -r requirements
```

The main file is `HierarchicMDP.py`. An example of how our algorithm can be used to learn a simple MDP is given in `Example.py`.

## Acknowledgements

This repository internally uses the [AALpy](https://github.com/DES-Lab/AALpy) automata learning library.