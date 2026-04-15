# Epidemiological simulation

Combines a discrete spatial infection process on a 2D grid (3D surface plot of infected fraction) with a textbook mean-field SIR model integrated via `scipy.integrate.solve_ivp`. Results are checked with `validate_simulation_solution` and wrapped in `SimulationResult`.
