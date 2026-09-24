"""Root wrapper for voidformer.quantum_init."""

from voidformer.quantum_init import initialize_quantum_processor, get_quantum_processor

if __name__ == "__main__":
    print("Initializing Virtual Quantum Processor...")
    proc = initialize_quantum_processor()
    print(f"Initialized processor with {proc.n_qubits} qubits per token, state dim={proc.state_dim}")
