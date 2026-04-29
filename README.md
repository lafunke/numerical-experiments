# Numerical Experiments: standard and enriched POD-ROM for two examples

1.  Heat equation example: linear example from pyMOR
2.  Brusselator system: a nonlinear reaction-diffusion system, implemented with NGSolve and pyMOR

# Structure

## heat_example
load the example from the pyMOR library, compute std snapshots and enriched snapshots, and generate ROMs

## Brusselator_example
1. fom.py: contains the implementation of the FOM using NGSolve
2. parameter_study: compute the customized initial conditions and period times for chosen parameters
3. Brusselator_snapshots: compute the standard and enriches sets of snapshots
4. rom_Brusselator: generate the ROMs using different snapshots sets, error analysis
