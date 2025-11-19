"""
Generate publication-quality comparison plots for SPARC fits.

Plots:
1. Chi-squared scatter plot (coherence vs NFW)
2. Histogram of chi-squared ratio
3. Parameter trend plots (R_c vs M_disk, etc.)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))


def load_results(csv_file='../outputs/sparc_fit_summary.csv'):
    """
    Load SPARC fit results from CSV.
    
    Parameters:
    -----------
    csv_file : str
        Path to CSV file
        
    Returns:
    --------
    df : DataFrame
        Results dataframe
    """
    if not os.path.exists(csv_file):
        raise FileNotFoundError(f"Results file not found: {csv_file}")
    
    df = pd.read_csv(csv_file)
    print(f"Loaded {len(df)} galaxy fits from {csv_file}")
    
    return df


def plot_chi_squared_scatter(df, savefig='../outputs/chi_squared_scatter.png'):
    """
    Plot chi-squared scatter: coherence vs NFW.
    
    Parameters:
    -----------
    df : DataFrame
        Results dataframe
    savefig : str
        Save figure filename
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Color by winner
    colors = {'Coherence': 'C0', 'NFW': 'C2', 'Tie': 'C1'}
    
    for winner in df['winner'].unique():
        mask = df['winner'] == winner
        ax.scatter(df.loc[mask, 'chi2_red_nfw'], 
                  df.loc[mask, 'chi2_red_coherence'],
                  label=f'{winner} wins', 
                  s=100, alpha=0.7, color=colors.get(winner, 'gray'),
                  edgecolors='black', linewidths=1.5)
    
    # Add galaxy labels
    for idx, row in df.iterrows():
        ax.annotate(row['galaxy'], 
                   (row['chi2_red_nfw'], row['chi2_red_coherence']),
                   fontsize=9, alpha=0.7,
                   xytext=(5, 5), textcoords='offset points')
    
    # 1:1 line
    max_val = max(df['chi2_red_coherence'].max(), df['chi2_red_nfw'].max())
    ax.plot([0, max_val], [0, max_val], 'k--', linewidth=2, 
           alpha=0.5, label='1:1 line')
    
    # Ratio lines
    for ratio in [0.5, 2.0]:
        ax.plot([0, max_val], [0, max_val * ratio], 'k:', 
               linewidth=1, alpha=0.3)
    
    ax.set_xlabel('NFW $\\chi^2_{\\rm red}$', fontsize=14, fontweight='bold')
    ax.set_ylabel('Coherence $\\chi^2_{\\rm red}$', fontsize=14, fontweight='bold')
    ax.set_title('SPARC Galaxy Fits: Coherence vs NFW Dark Matter', 
                fontsize=16, fontweight='bold')
    ax.legend(fontsize=11, loc='upper left')
    ax.grid(alpha=0.3, which='both')
    ax.set_xlim(0, max_val * 1.1)
    ax.set_ylim(0, max_val * 1.1)
    ax.set_aspect('equal')
    
    # Add statistics
    coherence_wins = (df['winner'] == 'Coherence').sum()
    nfw_wins = (df['winner'] == 'NFW').sum()
    ties = (df['winner'] == 'Tie').sum()
    
    stats_text = f"Coherence wins: {coherence_wins}/{len(df)} ({100*coherence_wins/len(df):.0f}%)\n"
    stats_text += f"NFW wins: {nfw_wins}/{len(df)} ({100*nfw_wins/len(df):.0f}%)\n"
    if ties > 0:
        stats_text += f"Ties: {ties}/{len(df)}"
    
    ax.text(0.05, 0.95, stats_text, transform=ax.transAxes,
           fontsize=10, verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(savefig, dpi=300, bbox_inches='tight')
    print(f"Saved: {savefig}")
    plt.show()


def plot_ratio_histogram(df, savefig='../outputs/chi_squared_ratio_histogram.png'):
    """
    Plot histogram of chi-squared ratio (coherence / NFW).
    
    Parameters:
    -----------
    df : DataFrame
        Results dataframe
    savefig : str
        Save figure filename
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ratios = df['ratio'].values
    
    # Histogram
    bins = np.logspace(np.log10(ratios.min() * 0.5), np.log10(ratios.max() * 2), 20)
    ax.hist(ratios, bins=bins, alpha=0.7, color='C0', edgecolor='black', linewidth=1.5)
    
    # Statistics
    mean_ratio = np.mean(ratios)
    median_ratio = np.median(ratios)
    frac_below_1 = (ratios < 1.0).sum() / len(ratios)
    frac_below_0_5 = (ratios < 0.5).sum() / len(ratios)
    
    # Vertical lines
    ax.axvline(1.0, color='k', linestyle='--', linewidth=2, 
              label='1:1 (equal fit)', alpha=0.7)
    ax.axvline(mean_ratio, color='C1', linestyle='-', linewidth=2,
              label=f'Mean: {mean_ratio:.3f}', alpha=0.7)
    ax.axvline(median_ratio, color='C2', linestyle='-', linewidth=2,
              label=f'Median: {median_ratio:.3f}', alpha=0.7)
    
    ax.set_xlabel('$\\chi^2_{\\rm red}$ Ratio (Coherence / NFW)', 
                 fontsize=14, fontweight='bold')
    ax.set_ylabel('Number of Galaxies', fontsize=14, fontweight='bold')
    ax.set_title('Distribution of $\\chi^2_{\\rm red}$ Ratio: Coherence vs NFW',
                fontsize=16, fontweight='bold')
    ax.set_xscale('log')
    ax.legend(fontsize=11, loc='upper left')
    ax.grid(alpha=0.3, which='both', axis='x')
    
    # Statistics text
    stats_text = "chi^2_co / chi^2_NFW\n"
    stats_text += f"Mean: {mean_ratio:.3f}\n"
    stats_text += f"Median: {median_ratio:.3f}\n"
    stats_text += f"< 1.0: {100*frac_below_1:.0f}%\n"
    stats_text += f"< 0.5: {100*frac_below_0_5:.0f}%"
    
    ax.text(0.95, 0.95, stats_text, transform=ax.transAxes,
           fontsize=11, verticalalignment='top', horizontalalignment='right',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(savefig, dpi=300, bbox_inches='tight')
    print(f"Saved: {savefig}")
    plt.show()


def plot_parameter_trends(df, savefig_prefix='../outputs/parameter_trends'):
    """
    Plot parameter trend plots.
    
    Parameters:
    -----------
    df : DataFrame
        Results dataframe
    savefig_prefix : str
        Prefix for saved figures
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # 1. R_c vs M_disk
    ax = axes[0, 0]
    ax.scatter(df['M_disk_co'], df['R_c'], s=100, alpha=0.7, 
              c=df['chi2_red_coherence'], cmap='viridis_r',
              edgecolors='black', linewidths=1.5)
    ax.set_xlabel('$M_{\\rm disk}$ (M$_\\odot$)', fontsize=12, fontweight='bold')
    ax.set_ylabel('$R_c$ (kpc)', fontsize=12, fontweight='bold')
    ax.set_title('Coherence Halo Core Radius vs Disk Mass', 
                fontsize=13, fontweight='bold')
    ax.set_xscale('log')
    ax.grid(alpha=0.3)
    cbar = plt.colorbar(ax.collections[0], ax=ax)
    cbar.set_label('$\\chi^2_{\\rm red}$', fontsize=11)
    
    # Add galaxy labels
    for idx, row in df.iterrows():
        ax.annotate(row['galaxy'], 
                   (row['M_disk_co'], row['R_c']),
                   fontsize=8, alpha=0.6,
                   xytext=(3, 3), textcoords='offset points')
    
    # 2. R_c / R_disk ratio
    ax = axes[0, 1]
    ratio_rc_rd = df['R_c'] / df['R_disk_co']
    ax.scatter(df['M_disk_co'], ratio_rc_rd, s=100, alpha=0.7,
              c=df['chi2_red_coherence'], cmap='viridis_r',
              edgecolors='black', linewidths=1.5)
    ax.set_xlabel('$M_{\\rm disk}$ (M$_\\odot$)', fontsize=12, fontweight='bold')
    ax.set_ylabel('$R_c / R_{\\rm disk}$', fontsize=12, fontweight='bold')
    ax.set_title('Halo-to-Disk Scale Ratio', fontsize=13, fontweight='bold')
    ax.set_xscale('log')
    ax.grid(alpha=0.3)
    ax.axhline(1.0, color='k', linestyle='--', alpha=0.5, label='R_c = R_disk')
    ax.axhline(2.0, color='k', linestyle=':', alpha=0.3, label='R_c = 2 R_disk')
    ax.legend(fontsize=9)
    cbar = plt.colorbar(ax.collections[0], ax=ax)
    cbar.set_label('$\\chi^2_{\\rm red}$', fontsize=11)
    
    # 3. rho_c0 vs M_disk
    ax = axes[1, 0]
    ax.scatter(df['M_disk_co'], df['rho_c0'], s=100, alpha=0.7,
              c=df['chi2_red_coherence'], cmap='viridis_r',
              edgecolors='black', linewidths=1.5)
    ax.set_xlabel('$M_{\\rm disk}$ (M$_\\odot$)', fontsize=12, fontweight='bold')
    ax.set_ylabel('$\\rho_{c0}$ (dimensionless)', fontsize=12, fontweight='bold')
    ax.set_title('Coherence Density vs Disk Mass', fontsize=13, fontweight='bold')
    ax.set_xscale('log')
    ax.grid(alpha=0.3)
    cbar = plt.colorbar(ax.collections[0], ax=ax)
    cbar.set_label('$\\chi^2_{\\rm red}$', fontsize=11)
    
    # 4. Ratio vs M_disk (which model wins?)
    ax = axes[1, 1]
    colors_winner = {'Coherence': 'C0', 'NFW': 'C2', 'Tie': 'C1'}
    for winner in df['winner'].unique():
        mask = df['winner'] == winner
        ax.scatter(df.loc[mask, 'M_disk_co'], df.loc[mask, 'ratio'],
                  label=f'{winner} wins', s=100, alpha=0.7,
                  color=colors_winner.get(winner, 'gray'),
                  edgecolors='black', linewidths=1.5)
    ax.set_xlabel('$M_{\\rm disk}$ (M$_\\odot$)', fontsize=12, fontweight='bold')
    ax.set_ylabel('$\\chi^2_{\\rm co} / \\chi^2_{\\rm NFW}$', 
                 fontsize=12, fontweight='bold')
    ax.set_title('Fit Quality Ratio vs Disk Mass', fontsize=13, fontweight='bold')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.axhline(1.0, color='k', linestyle='--', linewidth=2, 
              alpha=0.5, label='Equal fit')
    ax.grid(alpha=0.3, which='both')
    ax.legend(fontsize=10, loc='best')
    
    plt.tight_layout()
    plt.savefig(f'{savefig_prefix}.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {savefig_prefix}.png")
    plt.show()
    
    return ratio_rc_rd


def create_comparison_summary(df, savefig='../outputs/sparc_comparison_summary.png'):
    """
    Create comprehensive comparison summary figure.
    
    Parameters:
    -----------
    df : DataFrame
        Results dataframe
    savefig : str
        Save figure filename
    """
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # Panel 1: Chi-squared scatter
    ax1 = fig.add_subplot(gs[0, :2])
    colors = {'Coherence': 'C0', 'NFW': 'C2', 'Tie': 'C1'}
    for winner in df['winner'].unique():
        mask = df['winner'] == winner
        ax1.scatter(df.loc[mask, 'chi2_red_nfw'], 
                   df.loc[mask, 'chi2_red_coherence'],
                   label=f'{winner} wins', s=120, alpha=0.7,
                   color=colors.get(winner, 'gray'),
                   edgecolors='black', linewidths=1.5)
    max_val = max(df['chi2_red_coherence'].max(), df['chi2_red_nfw'].max())
    ax1.plot([0, max_val], [0, max_val], 'k--', linewidth=2, alpha=0.5)
    ax1.set_xlabel('NFW $\\chi^2_{\\rm red}$', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Coherence $\\chi^2_{\\rm red}$', fontsize=12, fontweight='bold')
    ax1.set_title('(a) Chi-Squared Comparison', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(alpha=0.3)
    ax1.set_xlim(0, max_val * 1.1)
    ax1.set_ylim(0, max_val * 1.1)
    ax1.set_aspect('equal')
    
    # Panel 2: Ratio histogram
    ax2 = fig.add_subplot(gs[0, 2])
    ratios = df['ratio'].values
    bins = np.logspace(np.log10(ratios.min() * 0.5), np.log10(ratios.max() * 2), 15)
    ax2.hist(ratios, bins=bins, alpha=0.7, color='C0', edgecolor='black', linewidth=1.5)
    ax2.axvline(1.0, color='k', linestyle='--', linewidth=2, alpha=0.7)
    ax2.set_xlabel('Ratio (Coherence / NFW)', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Count', fontsize=11, fontweight='bold')
    ax2.set_title('(b) Ratio Distribution', fontsize=13, fontweight='bold')
    ax2.set_xscale('log')
    ax2.grid(alpha=0.3, which='both', axis='x')
    
    # Panel 3: R_c vs M_disk
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.scatter(df['M_disk_co'], df['R_c'], s=100, alpha=0.7,
               c=df['chi2_red_coherence'], cmap='viridis_r',
               edgecolors='black', linewidths=1.5)
    ax3.set_xlabel('$M_{\\rm disk}$ (M$_\\odot$)', fontsize=11, fontweight='bold')
    ax3.set_ylabel('$R_c$ (kpc)', fontsize=11, fontweight='bold')
    ax3.set_title('(c) $R_c$ vs $M_{\\rm disk}$', fontsize=12, fontweight='bold')
    ax3.set_xscale('log')
    ax3.grid(alpha=0.3)
    
    # Panel 4: R_c / R_disk ratio
    ax4 = fig.add_subplot(gs[1, 1])
    ratio_rc_rd = df['R_c'] / df['R_disk_co']
    ax4.scatter(df['M_disk_co'], ratio_rc_rd, s=100, alpha=0.7,
               c=df['chi2_red_coherence'], cmap='viridis_r',
               edgecolors='black', linewidths=1.5)
    ax4.set_xlabel('$M_{\\rm disk}$ (M$_\\odot$)', fontsize=11, fontweight='bold')
    ax4.set_ylabel('$R_c / R_{\\rm disk}$', fontsize=11, fontweight='bold')
    ax4.set_title('(d) Halo-to-Disk Ratio', fontsize=12, fontweight='bold')
    ax4.set_xscale('log')
    ax4.grid(alpha=0.3)
    ax4.axhline(1.0, color='k', linestyle='--', alpha=0.5)
    ax4.axhline(2.0, color='k', linestyle=':', alpha=0.3)
    
    # Panel 5: rho_c0 vs M_disk
    ax5 = fig.add_subplot(gs[1, 2])
    ax5.scatter(df['M_disk_co'], df['rho_c0'], s=100, alpha=0.7,
               c=df['chi2_red_coherence'], cmap='viridis_r',
               edgecolors='black', linewidths=1.5)
    ax5.set_xlabel('$M_{\\rm disk}$ (M$_\\odot$)', fontsize=11, fontweight='bold')
    ax5.set_ylabel('$\\rho_{c0}$', fontsize=11, fontweight='bold')
    ax5.set_title('(e) $\\rho_{c0}$ vs $M_{\\rm disk}$', fontsize=12, fontweight='bold')
    ax5.set_xscale('log')
    ax5.grid(alpha=0.3)
    
    # Panel 6: Ratio vs M_disk
    ax6 = fig.add_subplot(gs[2, :])
    for winner in df['winner'].unique():
        mask = df['winner'] == winner
        ax6.scatter(df.loc[mask, 'M_disk_co'], df.loc[mask, 'ratio'],
                   label=f'{winner} wins', s=120, alpha=0.7,
                   color=colors.get(winner, 'gray'),
                   edgecolors='black', linewidths=1.5)
    ax6.set_xlabel('$M_{\\rm disk}$ (M$_\\odot$)', fontsize=12, fontweight='bold')
    ax6.set_ylabel('$\\chi^2_{\\rm co} / \\chi^2_{\\rm NFW}$', 
                  fontsize=12, fontweight='bold')
    ax6.set_title('(f) Fit Quality Ratio vs Galaxy Mass', 
                 fontsize=13, fontweight='bold')
    ax6.set_xscale('log')
    ax6.set_yscale('log')
    ax6.axhline(1.0, color='k', linestyle='--', linewidth=2, alpha=0.5)
    ax6.grid(alpha=0.3, which='both')
    ax6.legend(fontsize=11, loc='best')
    
    # Add overall title
    fig.suptitle('SPARC Galaxy Fits: Coherence Field vs NFW Dark Matter', 
                fontsize=16, fontweight='bold', y=0.995)
    
    plt.savefig(savefig, dpi=300, bbox_inches='tight')
    print(f"Saved: {savefig}")
    plt.show()


def main():
    """Main function to generate all comparison plots."""
    print("=" * 80)
    print("SPARC COMPARISON PLOTS")
    print("=" * 80)
    
    # Load results
    csv_file = '../outputs/sparc_fit_summary.csv'
    df = load_results(csv_file)
    
    print(f"\nDataset summary:")
    print(f"  Total galaxies: {len(df)}")
    print(f"  Coherence wins: {(df['winner'] == 'Coherence').sum()}")
    print(f"  NFW wins: {(df['winner'] == 'NFW').sum()}")
    print(f"  Mean ratio: {df['ratio'].mean():.3f}")
    print(f"  Median ratio: {df['ratio'].median():.3f}")
    
    # Generate plots
    print("\n" + "=" * 80)
    print("Generating comparison plots...")
    print("=" * 80)
    
    plot_chi_squared_scatter(df)
    plot_ratio_histogram(df)
    plot_parameter_trends(df)
    create_comparison_summary(df)
    
    print("\n" + "=" * 80)
    print("All plots generated successfully!")
    print("=" * 80)


if __name__ == '__main__':
    main()

