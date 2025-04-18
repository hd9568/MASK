#!/usr/bin/env python3
import argparse
import os
import re

def parse_mtx_header(mtx_path):
    """Parse the header of an MTX file to get matrix dimensions and nnz."""
    with open(mtx_path, 'r') as f:
        # Skip comments
        line = f.readline()
        while line.startswith('%'):
            line = f.readline()
        
        # First non-comment line should be the header
        header = line.strip().split()
        if len(header) != 3:
            raise ValueError(f"Invalid MTX header in {mtx_path}")
        
        rows, cols, nnz = map(int, header)
        return rows, cols, nnz

def calculate_sparsity(rows, cols, nnz):
    """Calculate sparsity as a percentage."""
    total_elements = rows * cols
    if total_elements == 0:
        return 0.0
    return (1 - nnz / total_elements) * 100

def main():
    parser = argparse.ArgumentParser(description='Analyze MTX matrices and output statistics in LaTeX format')
    parser.add_argument('-i', '--mat-list', required=True, help='Path to file containing list of MTX matrices')
    args = parser.parse_args()

    with open(args.mat_list, 'r') as f:
        matrix_paths = [line.strip() for line in f if line.strip()]

    for mtx_path in matrix_paths:
        if not os.path.exists(mtx_path):
            print(f"Warning: Matrix file {mtx_path} does not exist")
            continue

        try:
            rows, cols, nnz = parse_mtx_header(mtx_path)
            sparsity = calculate_sparsity(rows, cols, nnz)
            
            # Extract matrix name from path and escape underscores
            matrix_name = os.path.splitext(os.path.basename(mtx_path))[0].replace('_', '\\_')
            
            # Format output in LaTeX with line continuation
            print(f" & {matrix_name} & {rows}$\\times${cols} & {nnz} ({sparsity:.2f}\\%) \\\\")
        except Exception as e:
            print(f"Error processing {mtx_path}: {str(e)}")

if __name__ == '__main__':
    main()
