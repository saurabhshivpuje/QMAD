# FMO Trajectory Simulation

This documentation describes how to simulate and visualize the **Fenna–Matthews–Olson (FMO)** complex trajectory using the `qflux` framework. The goal is to analyze excitation energy transfer within the FMO system and demonstrate time-dependent propagation of the reduced density matrix.

---

## Overview

The FMO complex serves as a benchmark system for studying quantum coherence in biological systems. The simulation computes time evolution trajectories for the exciton states within the seven-site FMO model.

The following modules are used throughout the documentation:

```python
import numpy as np
import matplotlib.pyplot as plt
import qflux.FMO.params as pa
import qflux.FMO.readwrite as rw
import qflux.FMO.dynamics as dy
```

These modules define key parameters (site energies, coupling constants, bath characteristics) and provide utilities for input/output and trajectory propagation.

---

## FMO Hamiltonian and System Setup

The Hamiltonian for the seven-site FMO model can be expressed as:

$$
\hat{H}*{\text{FMO}} = \sum*{i=1}^{7} \epsilon_i |i\rangle \langle i| + \sum_{i\ne j} J_{ij} |i\rangle \langle j|
$$

where $\epsilon_i$ are the site energies and $J_{ij}$ are electronic couplings between sites. The parameters are typically derived from spectroscopic measurements or literature data (see Adolphs & Renger, *Biophys. J.*, 2006).

```python
H_FMO = pa.H_FMO  # 7x7 Hamiltonian matrix
rho0 = np.zeros((7,7), dtype=np.complex128)
rho0[0,0] = 1.0  # initial excitation on site 1
```

The initial density matrix $\rho(0)$ represents full population on the donor site.

![FMO\_Structure](../images/FMO/FMO_structure.png)

---

## Propagation of the Trajectory

To simulate dynamics, the Liouville–von Neumann equation is solved:

$$
\frac{d\rho(t)}{dt} = -\frac{i}{\hbar}[H_{\text{FMO}}, \rho(t)] - \mathcal{D}[\rho(t)]
$$

where $\mathcal{D}$ denotes dissipative terms introduced by the environment.

The time propagation is implemented using a fourth-order Runge–Kutta (RK4) integrator within the `dynamics` module.

```python
traj = dy.TrajectoryFMO(H_FMO, rho0)
traj.setup_timestep(pa.DT, pa.TIME_STEPS)
rho_t = traj.propagate_RK4()
```

The simulation outputs a time series of density matrices $\rho(t)$. Each diagonal element corresponds to the site population, while off-diagonal elements represent coherence.

![FMO\_trajectory](../images/FMO/FMO_trajectory.png)

---

## Saving and Loading Trajectories

Computed trajectories can be stored and retrieved using the `readwrite` utility.

```python
rw.output_density_array(traj.timeVec, rho_t, 'qflux/data/FMO/Trajectory_Output/FMO_')

# To reload a saved trajectory
timeVec, rho_t = rw.read_density_array(pa.TIME_STEPS, 'qflux/data/FMO/Trajectory_Output/FMO_')
```

This ensures reproducibility and allows post-processing or comparison with experimental datasets.

![FMO\_save\_load](../images/FMO/FMO_save_load.png)

---

## Visualization

Populations can be visualized to study the energy transfer pathways among sites. A typical plot includes populations of all seven sites as functions of time.

```python
plt.figure(figsize=(8,4))
for i in range(7):
    plt.plot(traj.timeVec, rho_t[:, i].real, label=f'Site {i+1}')
plt.xlabel('Time (fs)')
plt.ylabel('Population')
plt.legend()
plt.tight_layout()
```

The resulting population dynamics reveal oscillations and relaxation trends characteristic of coherent transport.

![FMO\_population](../images/FMO/FMO_population.png)

---

## Analysis and Discussion

The oscillatory features in populations reflect quantum coherence effects between excitonic states. The damping of oscillations over time is attributed to environmental dephasing captured by the dissipator.

Quantitative analysis may involve computing coherence lifetimes, transfer rates, or comparing with Redfield or Lindblad models.

![FMO\_analysis](../images/FMO/FMO_analysis.png)

---

# Summary

This example demonstrates the workflow for simulating excitation energy transfer dynamics in the **FMO complex** using the `qflux` package:

* The system Hamiltonian defines site energies and couplings.
* Trajectories are propagated via RK4 integration of the Liouville–von Neumann equation.
* Time-dependent populations and coherences can be saved, loaded, and visualized.

The modular implementation enables straightforward extension to other pigment-protein complexes or open quantum system models.
