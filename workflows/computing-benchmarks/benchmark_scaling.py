#!/usr/bin/env python3
"""
Benchmark scaling analysis script.

This script performs scaling benchmarks by building and running the same domain
configuration with different process partitionings. It uses CstarSpecEngine to
generate and execute all domain configurations from a domains YAML file.

Usage:
    python benchmark_scaling.py [OPTIONS]

Example:
    python benchmark_scaling.py --domains-file domains-bm-scaling.yml --clobber-inputs
"""

import argparse
import sys
import time
import warnings
from pathlib import Path
from typing import Optional

import cson_forge
from cstar.execution.handler import ExecutionStatus


def main():
    """Main entry point for the benchmark scaling script."""
    parser = argparse.ArgumentParser(
        description="Run scaling benchmarks using CstarSpecEngine",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    
    parser.add_argument(
        "--domains-file",
        type=str,
        default="domains-bm-scaling.yml",
        help="Path to domains YAML file (default: domains-bm-scaling.yml)",
    )
    
    parser.add_argument(
        "--clobber-inputs",
        action="store_true",
        help="Clobber existing input files when generating domains",
    )
    
    parser.add_argument(
        "--generate-only",
        action="store_true",
        help="Only generate domains, do not run simulations",
    )
    
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose output",
    )
    
    parser.add_argument(
        "--ensemble-id",
        type=int,
        default=None,
        help="Ensemble ID to use for all domains (default: 1 if not specified)",
    )
    
    args = parser.parse_args()
    
    # Resolve domains file path
    domains_file = Path(args.domains_file)
    if not domains_file.is_absolute():
        # Try relative to current directory, then relative to cson_forge package
        if not domains_file.exists():
            domains_file = Path(cson_forge.config.paths.here) / domains_file
        else:
            domains_file = domains_file.resolve()
    
    if not domains_file.exists():
        print(f"Error: Domains file not found: {domains_file}", file=sys.stderr)
        sys.exit(1)
    
    if args.verbose:
        print(f"Using domains file: {domains_file}")
        print(f"Clobber inputs: {args.clobber_inputs}")
        print(f"Generate only: {args.generate_only}")
        if args.ensemble_id is not None:
            print(f"Ensemble ID: {args.ensemble_id}")
        print()
    
    try:
        # Initialize engine
        print(f"Initializing CstarSpecEngine with domains file: {domains_file}")
        engine = cson_forge.CstarSpecEngine(domains_file=str(domains_file))
        
        # Generate all domains
        print("\n" + "=" * 80)
        print("Generating all domain configurations...")
        print("=" * 80)
        builders = engine.generate_all(
            clobber_inputs=args.clobber_inputs,
            ensemble_id=args.ensemble_id
        )
        
        print(f"\n✓ Generated {len(builders)} domain configuration(s)")
        for grid_name in builders.keys():
            print(f"  - {grid_name}")
        
        if args.generate_only:
            print("\n✓ Generation complete. Exiting (--generate-only specified).")
            return 0
        
        total_domains = len(builders)
        
        # Run all simulations
        print("\n" + "=" * 80)
        print("Running all simulations...")
        print("=" * 80)
        
        
        for idx, (grid_name, builder) in enumerate(builders.items(), start=1):
            print(f"\n{'-'*80}")
            print(f"[{idx}/{total_domains}] Running simulation: {grid_name}")
            print(f"{'-'*80}")            
            execution_handler = builder.run()
            print(f"Execution handler: {execution_handler}")
                           
        print(f"\n{'='*80}")
        print("All simulations submitted.")
        print(f"{'='*80}")
        
        return 0
    
    except Exception as e:
        print(f"\n✗ Error running simulation {grid_name}: {e}")
        warnings.warn(
            f"Error running simulation {grid_name}: {e}",
            UserWarning,
            stacklevel=2
        )
        return 1

if __name__ == "__main__":
    sys.exit(main())
