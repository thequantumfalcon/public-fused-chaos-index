from fused_chaos_index import StreamlinedFCIPipeline, compute_syk_collatz_fci_constant


def main() -> None:
    pipeline = StreamlinedFCIPipeline(k_neighbors=10, seed=42)
    results = pipeline.run(n_galaxies=1500)
    fci = results["fci"]

    print("Operational FCI")
    print("--------------")
    print(f"FCI (normalized): {fci.fci_normalized:.6f}")
    print(f"Regime: {fci.physical_regime}")

    c = compute_syk_collatz_fci_constant().C
    print("\nSYK–Collatz constant")
    print("--------------------")
    print(f"C: {c:.6f}")


if __name__ == "__main__":
    main()
