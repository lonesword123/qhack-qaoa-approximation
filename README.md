# QHack QAOA approximation

This repository contains my solution to a QHack challenge on applying the Quantum Approximate Optimization Algorithm (QAOA) to the Minimum Vertex Cover problem.

## Challenge overview

Given an arbitrary graph, the task is to find a small set of vertices such that every edge has at least one endpoint in the set. The challenge asks for a QAOA implementation and a comparison between the QAOA result and the exact classical minimum eigenvalue of the cost Hamiltonian.

The implementation uses four QAOA layers, as required by the challenge. It constructs the cost and mixer Hamiltonians, optimizes the QAOA parameters with gradient descent, and returns the ratio between the optimized QAOA expectation value and the exact minimum eigenvalue.

## Implementation

The solution is in [`solution.py`](solution.py). It uses:

- Python
- [PennyLane](https://pennylane.ai/)
- `default.qubit` for simulation
- A four-layer QAOA circuit
- Gradient descent for 300 optimization steps

The challenge Hamiltonians are implemented directly in `build_hamiltonians`. The cost Hamiltonian is built from the graph edges and vertex degrees, and the mixer Hamiltonian is the sum of Pauli-X operators over the graph vertices.

## Public test results

The original solution passed both public test cases locally:

| Test case | Input graph | Expected output | Result |
|---|---|---:|---|
| 0 | `[[0, 1], [1, 2], [0, 2], [2, 3]]` | `0.58160924` | Correct |
| 1 | `[[0, 1], [1, 2], [2, 3], [3, 0]]` | `0.94746734` | Correct |

The challenge checker used a relative tolerance of `5e-2` when comparing outputs.

## Running the solution

Create an environment and install the dependency:

```bash
python -m venv .venv
source .venv/bin/activate       # macOS/Linux
# .venv\Scripts\activate      # Windows PowerShell
pip install -r requirements.txt
```

Then run:

```bash
python solution.py
```

The script runs the two public test cases and prints the result for each one.

## Note

This repository documents the original working implementation and its reported public-test results. The optimization is stochastic or simulator/version-sensitive in practice, so small numerical differences may occur while remaining within the challenge tolerance.
