import json

import pennylane as qp
import pennylane.numpy as np


def build_hamiltonians(edges):
    """Build the cost and mixer Hamiltonians from a graph edge list."""
    vertices = sorted(list(set(v for edge in edges for v in edge)))
    degrees = {v: 0 for v in vertices}

    cost_coeffs = []
    cost_obs = []

    for i, j in edges:
        cost_coeffs.append(0.75)
        cost_obs.append(qp.PauliZ(i) @ qp.PauliZ(j))
        degrees[i] += 1
        degrees[j] += 1

    for i in vertices:
        cost_coeffs.append(0.75 * degrees[i] - 1.0)
        cost_obs.append(qp.PauliZ(i))

    cost_h = qp.Hamiltonian(cost_coeffs, cost_obs)
    mixer_h = qp.Hamiltonian([1.0] * len(vertices), [qp.PauliX(i) for i in vertices])

    return vertices, cost_h, mixer_h


def approximation_ratio(edges):
    """Calculate the QAOA approximation ratio for Minimum Vertex Cover."""
    vertices, cost_h, mixer_h = build_hamiltonians(edges)
    depth = 4

    dev = qp.device("default.qubit", wires=vertices)

    def qaoa_layer(gamma, alpha):
        qp.qaoa.cost_layer(gamma, cost_h)
        qp.qaoa.mixer_layer(alpha, mixer_h)

    @qp.qnode(dev)
    def cost_function(params):
        for w in vertices:
            qp.Hadamard(wires=w)
        qp.layer(qaoa_layer, depth, params[0], params[1])
        return qp.expval(cost_h)

    optimizer = qp.GradientDescentOptimizer()
    steps = 300
    params = np.array([[0.5] * depth, [0.5] * depth], requires_grad=True)

    for _ in range(steps):
        params = optimizer.step(cost_function, params)

    qaoa_min = cost_function(params)
    true_min = np.min(np.real(qp.eigvals(cost_h)))

    return np.tensor(np.real(qaoa_min / true_min))


def run(test_case_input: str) -> str:
    """Run the solution using the challenge's JSON input format."""
    ins = json.loads(test_case_input)
    output = approximation_ratio(ins)
    return str(output)


def check(solution_output: str, expected_output: str) -> None:
    """Check an output using the challenge tolerance."""
    solution_output = json.loads(solution_output)
    expected_output = json.loads(expected_output)
    assert np.allclose(
        solution_output, expected_output, rtol=5e-2
    ), "Your code does not yield the right approximation ratio"


if __name__ == "__main__":
    test_cases = [
        ("[[0, 1], [1, 2], [0, 2], [2, 3]]", "0.58160924"),
        ("[[0, 1], [1, 2], [2, 3], [3, 0]]", "0.94746734"),
    ]

    for i, (input_, expected_output) in enumerate(test_cases):
        print(f"Running test case {i} with input '{input_}'...")
        try:
            output = run(input_)
        except Exception as exc:
            print(f"Runtime Error. {exc}")
        else:
            try:
                check(output, expected_output)
            except AssertionError:
                print(f"Wrong Answer. Have: '{output}'. Want: '{expected_output}'.")
            else:
                print("Correct!")
