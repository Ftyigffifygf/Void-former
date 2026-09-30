"""Customizable Workflow CLI Dispatcher for Harness Package."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

repo_root = Path(__file__).parent.parent.resolve()
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from harness.node import AgentNode, QuantumBackendNode
from harness.workflow import WorkflowDAG
from harness.ab_benchmark import run_ab_benchmark
from harness.ablation_sweep import run_ablation_sweep
from harness.scaling_profile import profile_scaling


def run_custom_workflow():
    print("=" * 65)
    print("       N8N-STYLE VOIDFORMER TEAM WORKFLOW ENGINE RUNNER         ")
    print("=" * 65)

    dag = WorkflowDAG(name="Quantum_AI_Consensus_Team")

    # Add customizable agent nodes with high loop counts
    dag.add_node(AgentNode("node_1", "Quantum Algorithm Specialist", skill="Grover_Search", max_loops=10))
    dag.add_node(AgentNode("node_2", "Circuit Synthesis Agent", skill="CNOT_Entanglement", max_loops=15))
    dag.add_node(QuantumBackendNode("node_3", "Qiskit Aer QPU Bridge", backend_type="aer"))

    dag.add_edge("node_1", "node_2")
    dag.add_edge("node_2", "node_3")

    results = dag.run()
    for n_id, data in results.items():
        print(f"Node '{n_id}': {data}")

    return results


def main():
    parser = argparse.ArgumentParser(description="VoidFormer Evaluation Harness & Team Workflow CLI")
    parser.add_argument("mode", choices=["workflow", "benchmark", "sweep", "profile", "all"], help="Execution mode")
    parser.add_argument("--steps", type=int, default=10, help="Training steps")
    args = parser.parse_args()

    if args.mode in ["workflow", "all"]:
        run_custom_workflow()
    if args.mode in ["benchmark", "all"]:
        run_ab_benchmark(steps=args.steps)
    if args.mode in ["sweep", "all"]:
        run_ablation_sweep(steps=args.steps)
    if args.mode in ["profile", "all"]:
        profile_scaling()


if __name__ == "__main__":
    main()
