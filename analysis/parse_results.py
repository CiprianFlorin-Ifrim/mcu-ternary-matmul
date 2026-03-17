"""
parse_results.py

Reads benchmark CSV output from the ESP32-P4 serial port (or a saved file),
parses the results, and produces a set of plots comparing the four approaches
across all matrix sizes and zero fractions.

Usage:
    # From a saved CSV file
    python parse_results.py --file results.csv

    # Live from serial port (captures until Ctrl+C)
    python parse_results.py --port /dev/cu.usbserial-XXXX --baud 921600

Output:
    plots/  -- directory of PNG plots
    results_clean.csv -- cleaned and validated CSV
"""

import argparse
import os
import sys
import time

import numpy  as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

# ---------------------------------------------------------------------------
# Plot style
# ---------------------------------------------------------------------------

plt.rcParams.update({
    'figure.dpi'       : 120,
    'axes.spines.top'  : False,
    'axes.spines.right': False,
    'axes.grid'        : True,
    'grid.alpha'       : 0.3,
    'font.size'        : 10,
})

APPROACH_COLOURS = {
    'dense_c'       : '#2b7bb9',
    'dense_pie'     : '#d95f02',
    'sparse_bitmask': '#1b9e77',
    'sparse_pie'    : '#7570b3',
}

APPROACH_LABELS = {
    'dense_c'       : 'Dense C',
    'dense_pie'     : 'Dense PIE',
    'sparse_bitmask': 'Sparse Bitmask',
    'sparse_pie'    : 'Block-Sparse PIE',
}

# ---------------------------------------------------------------------------
# CSV parsing
# ---------------------------------------------------------------------------

EXPECTED_COLS = [
    'approach', 'rows', 'cols', 'label',
    'zero_frac_req', 'zero_frac_act',
    'mean_us', 'min_us', 'max_us', 'n_runs'
]


def load_csv(path):
    """
    Load and validate benchmark CSV. Filters out ESP-IDF log lines that
    begin with 'I (' so the file can be captured directly from the serial
    monitor without pre-processing.
    """
    clean_lines = []
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # Skip ESP-IDF log lines
            if line.startswith('I (') or line.startswith('E (') or line.startswith('W ('):
                continue
            clean_lines.append(line)

    from io import StringIO
    clean_text = '\n'.join(clean_lines)
    df = pd.read_csv(StringIO(clean_text))

    # Validate columns
    for col in EXPECTED_COLS:
        if col not in df.columns:
            print(f'WARNING: expected column "{col}" not found in CSV')

    print(f'Loaded {len(df)} rows from {path}')
    return df


def capture_serial(port, baud, output_path):
    """
    Capture CSV lines from serial port until Ctrl+C, save to output_path.
    Filters ESP-IDF log lines automatically.
    """
    import serial
    print(f'Opening {port} at {baud} baud. Press Ctrl+C to stop.')

    with serial.Serial(port, baud, timeout=1) as ser, open(output_path, 'w') as f:
        try:
            while True:
                line = ser.readline().decode('utf-8', errors='replace').strip()
                if not line:
                    continue
                print(line)
                if not (line.startswith('I (') or line.startswith('E (') or line.startswith('W (')):
                    f.write(line + '\n')
        except KeyboardInterrupt:
            print(f'\nCapture stopped. Saved to {output_path}')


# ---------------------------------------------------------------------------
# Plotting functions
# ---------------------------------------------------------------------------

def plot_approach_comparison_by_zero_frac(df, out_dir):
    """
    For each matrix label, plot mean_us vs zero_frac for all four approaches.
    One subplot per matrix shape, grouped into figures by shape category.
    """
    labels = df['label'].unique()

    for label in labels:
        sub = df[df['label'] == label]
        rows_val = sub['rows'].iloc[0]
        cols_val = sub['cols'].iloc[0]

        fig, ax = plt.subplots(figsize=(8, 4))

        for approach in APPROACH_LABELS:
            app_data = sub[sub['approach'] == approach].sort_values('zero_frac_req')
            if app_data.empty:
                continue
            ax.plot(
                app_data['zero_frac_req'],
                app_data['mean_us'],
                marker   = 'o',
                linewidth= 1.8,
                markersize=4,
                color    = APPROACH_COLOURS[approach],
                label    = APPROACH_LABELS[approach],
            )
            ax.fill_between(
                app_data['zero_frac_req'],
                app_data['min_us'],
                app_data['max_us'],
                alpha= 0.15,
                color= APPROACH_COLOURS[approach],
            )

        ax.set_title(f'{label}  ({rows_val} x {cols_val})', fontweight='bold')
        ax.set_xlabel('Zero fraction')
        ax.set_ylabel('Time (us)')
        ax.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
        ax.legend(fontsize=8)

        plt.tight_layout()
        fname = os.path.join(out_dir, f'comparison_{label}.png')
        plt.savefig(fname, bbox_inches='tight')
        plt.close()

    print(f'Saved per-shape comparison plots to {out_dir}')


def plot_crossover_heatmap(df, out_dir):
    """
    For each matrix shape and zero fraction, show which approach is fastest.
    Produces a heatmap with matrix shape on one axis and zero fraction on the other.
    """
    pivot_data = []

    for (label, zf), group in df.groupby(['label', 'zero_frac_req']):
        fastest = group.loc[group['mean_us'].idxmin(), 'approach']
        pivot_data.append({'label': label, 'zero_frac': zf, 'fastest': fastest})

    pivot_df = pd.DataFrame(pivot_data)

    # Encode approaches as integers for heatmap colouring
    approach_order = ['dense_c', 'dense_pie', 'sparse_bitmask', 'sparse_pie']
    approach_int   = {a: i for i, a in enumerate(approach_order)}
    pivot_df['fastest_int'] = pivot_df['fastest'].map(approach_int)

    pivot_matrix = pivot_df.pivot(index='label', columns='zero_frac', values='fastest_int')

    fig, ax = plt.subplots(figsize=(10, max(6, len(pivot_matrix) * 0.35)))

    cmap   = plt.cm.get_cmap('Set2', len(approach_order))
    im     = ax.imshow(pivot_matrix.values, aspect='auto', cmap=cmap,
                       vmin=0, vmax=len(approach_order) - 1)

    ax.set_xticks(range(len(pivot_matrix.columns)))
    ax.set_xticklabels([f'{v:.0%}' for v in pivot_matrix.columns], fontsize=8)
    ax.set_yticks(range(len(pivot_matrix.index)))
    ax.set_yticklabels(pivot_matrix.index, fontsize=7)
    ax.set_xlabel('Zero fraction')
    ax.set_ylabel('Matrix shape')
    ax.set_title('Fastest approach per shape and zero fraction', fontweight='bold')

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=cmap(approach_int[a]), label=APPROACH_LABELS[a])
                       for a in approach_order]
    ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1, 1), fontsize=8)

    plt.tight_layout()
    fname = os.path.join(out_dir, 'crossover_heatmap.png')
    plt.savefig(fname, bbox_inches='tight')
    plt.close()

    print(f'Saved crossover heatmap to {fname}')


def plot_speedup_over_dense_c(df, out_dir):
    """
    For each approach (excluding dense_c baseline), plot speedup ratio
    relative to dense_c at each zero fraction, averaged over all matrix shapes.
    """
    baseline = df[df['approach'] == 'dense_c'][['label', 'zero_frac_req', 'mean_us']]
    baseline = baseline.rename(columns={'mean_us': 'baseline_us'})

    merged = df[df['approach'] != 'dense_c'].merge(
        baseline, on=['label', 'zero_frac_req'], how='left'
    )
    merged['speedup'] = merged['baseline_us'] / merged['mean_us']

    fig, ax = plt.subplots(figsize=(9, 5))

    for approach in ['dense_pie', 'sparse_bitmask', 'sparse_pie']:
        sub = merged[merged['approach'] == approach].groupby('zero_frac_req')['speedup'].mean().reset_index()
        ax.plot(
            sub['zero_frac_req'],
            sub['speedup'],
            marker    = 'o',
            linewidth = 1.8,
            markersize= 4,
            color     = APPROACH_COLOURS[approach],
            label     = APPROACH_LABELS[approach],
        )

    ax.axhline(1.0, color='grey', linestyle='--', linewidth=0.8, label='Dense C baseline')
    ax.set_title('Mean speedup over Dense C (averaged over all matrix shapes)', fontweight='bold')
    ax.set_xlabel('Zero fraction')
    ax.set_ylabel('Speedup (x)')
    ax.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    ax.legend(fontsize=8)

    plt.tight_layout()
    fname = os.path.join(out_dir, 'speedup_over_dense_c.png')
    plt.savefig(fname, bbox_inches='tight')
    plt.close()

    print(f'Saved speedup plot to {fname}')


def plot_size_scaling(df, out_dir):
    """
    For square matrices only, plot mean_us vs matrix size (rows=cols) at
    a fixed zero fraction (0.70) for all four approaches.
    Shows how each approach scales with matrix size.
    """
    square = df[df['rows'] == df['cols']].copy()
    square = square[square['zero_frac_req'].between(0.69, 0.71)]

    if square.empty:
        print('No square matrix data at 0.70 zero fraction -- skipping size scaling plot')
        return

    square = square.sort_values('rows')

    fig, ax = plt.subplots(figsize=(8, 5))

    for approach in APPROACH_LABELS:
        sub = square[square['approach'] == approach]
        if sub.empty:
            continue
        ax.plot(
            sub['rows'],
            sub['mean_us'],
            marker    = 'o',
            linewidth = 1.8,
            markersize= 4,
            color     = APPROACH_COLOURS[approach],
            label     = APPROACH_LABELS[approach],
        )

    ax.set_title('Scaling with matrix size -- square matrices at 70% zero fraction',
                 fontweight='bold')
    ax.set_xlabel('Matrix dimension (N for N x N)')
    ax.set_ylabel('Time (us)')
    ax.set_xscale('log', base=2)
    ax.legend(fontsize=8)

    plt.tight_layout()
    fname = os.path.join(out_dir, 'size_scaling_square.png')
    plt.savefig(fname, bbox_inches='tight')
    plt.close()

    print(f'Saved size scaling plot to {fname}')


def print_summary_table(df):
    """
    Print a summary table showing the best approach for each matrix category
    at the highest zero fraction tested.
    """
    max_zf  = df['zero_frac_req'].max()
    high_zf = df[df['zero_frac_req'] == max_zf]

    print(f'\nFastest approach at zero fraction {max_zf:.0%}:')
    print(f'{"Label":<30}  {"Rows":>6}  {"Cols":>6}  {"Fastest":<18}  {"Mean us":>10}')
    print('-' * 75)

    for label in sorted(high_zf['label'].unique()):
        sub     = high_zf[high_zf['label'] == label]
        best    = sub.loc[sub['mean_us'].idxmin()]
        print(f'{label:<30}  {int(best["rows"]):>6}  {int(best["cols"]):>6}  '
              f'{best["approach"]:<18}  {best["mean_us"]:>10.2f}')


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Parse and plot ESP32-P4 benchmark results')
    group  = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--file',  type=str, help='Path to saved CSV file')
    group.add_argument('--port',  type=str, help='Serial port to capture from')
    parser.add_argument('--baud', type=int, default=921600, help='Serial baud rate')
    parser.add_argument('--out',  type=str, default='plots', help='Output directory for plots')
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    if args.port:
        raw_path = 'raw_capture.csv'
        capture_serial(args.port, args.baud, raw_path)
        df = load_csv(raw_path)
    else:
        df = load_csv(args.file)

    # Save cleaned version
    clean_path = os.path.join(args.out, 'results_clean.csv')
    df.to_csv(clean_path, index=False)
    print(f'Cleaned CSV saved to {clean_path}')

    # Generate all plots
    plot_approach_comparison_by_zero_frac(df, args.out)
    plot_crossover_heatmap(df, args.out)
    plot_speedup_over_dense_c(df, args.out)
    plot_size_scaling(df, args.out)
    print_summary_table(df)

    print(f'\nAll plots saved to {args.out}/')


if __name__ == '__main__':
    main()
