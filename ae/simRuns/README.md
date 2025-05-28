# Change log:

INFO: What is new?
- param_sweeps.py
  > Creating arbitrary sweeps for sensitivity and overall parameters
  > Next updated should be with more systematic configurations
  > Saperate Sweep for memory datarate, capacity, and interconnect

- param_sweeps_updated.py
  > Proper memory and interconnect configurations for sweeps
  > Removed sensitivity simulations since it will be in whole sweep
  > Changed results directory which was storing results in different simulation directory

- param_sweep_new.py
  >Running workload simulations:
  > Sweep for batch size, input seq length and output seq length

# Results directories
- results
    > No idea

- results_old
    > Results from param_sweeps.py

- results_stopped
    > Simulation from param_sweeps_updated.py with change to remove sensitivity
    > results

- results_updated
    > Results from param_sweeps_updated.py file
