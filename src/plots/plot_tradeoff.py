import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple, Dict, List


def get_plot_mappings(all_filters: List[str]) -> Tuple[Dict, Dict, Dict]:
    """Colors are grouped by the base filter name; markers/dashes cycle per config index."""
    available_marks = ['s', 'o', '^', 'D', 'v', 'p', '*', 'X', 'h', 'P', 'd', 'H', '<', '>']
    available_dashes = [(1, 1), (4, 1.5), (3, 1, 1.5, 1), (5, 2, 1, 2), (2, 2), (4, 1, 1, 1, 1, 1), (1, 3), (6, 1), (3, 3, 1, 3)]
    base_names = [c.split(' + ')[0] for c in all_filters]
    unique_bases = list(dict.fromkeys(base_names))
    base_palette = sns.color_palette("tab10", n_colors=len(unique_bases))
    base_color_map = dict(zip(unique_bases, base_palette))

    color_map = {config: base_color_map[config.split(' + ')[0]] for config in all_filters}

    marker_map = {config: available_marks[i % len(available_marks)] for i, config in enumerate(all_filters)}
    dash_map = {config: available_dashes[i % len(available_dashes)] for i, config in enumerate(all_filters)}
    
    return color_map, marker_map, dash_map


def load_tradeoff_results(folder: str = "plots/tradeoff_static") -> Tuple[List, List]:
    """Reconstructs sphere_results/memory_results from tradeoff_evaluations.csv
    (written as a side effect of plot_tradeoff -- see legacy.plot_tradeoff_results),
    splitting back on the 'Mode' column plot_tradeoff_results itself stamps onto
    each half before concatenating them for that CSV."""
    df = pd.read_csv(f"{folder}/tradeoff_evaluations.csv")
    sphere_results = df[df["Mode"] == "sphere_test"].to_dict("records")
    memory_results = df[df["Mode"] == "memory_test"].to_dict("records")
    return sphere_results, memory_results


def plot_tradeoff_results(spheres_results: list, memory_results: list, folder: str = "plots/tradeoff_static") -> None:
    """Plots tradeoff evaluation results, comparing FPR, FNR, Precision, F1-Score, QPS,
    and ConstructTime across both geometric (spheres) and hardware (memory) sweeps."""
    os.makedirs(folder, exist_ok=True)

    df_spheres = pd.DataFrame(spheres_results) if spheres_results else pd.DataFrame()
    df_memory = pd.DataFrame(memory_results) if memory_results else pd.DataFrame()

    if df_spheres.empty and df_memory.empty:
        print("Both spheres results and memory results are empty. No plots generated.")
        return

    if not df_spheres.empty and 'Mode' not in df_spheres.columns:
        df_spheres.insert(0, 'Mode', 'sphere_test')
    if not df_memory.empty and 'Mode' not in df_memory.columns:
        df_memory.insert(0, 'Mode', 'memory_test')

    if not df_memory.empty:
        df_memory['MemoryBudget_MB'] = df_memory['MemoryBudget'] / (1024 * 1024)

    sns.set_theme(style="whitegrid")

    spheres_configs = list(df_spheres['Config'].unique()) if not df_spheres.empty and 'Config' in df_spheres.columns else []
    memory_configs = list(df_memory['Config'].unique()) if not df_memory.empty and 'Config' in df_memory.columns else []
    all_filters = list(dict.fromkeys(spheres_configs + memory_configs))

    color_map, marker_map, dash_map = get_plot_mappings(all_filters)

    def _add_bottom_legend(ax_obj):
        handles, labels = ax_obj.get_legend_handles_labels()
        if ax_obj.get_legend() is not None:
            ax_obj.get_legend().remove()
        ax_obj.legend(
            handles, labels, 
            loc='upper center', bbox_to_anchor=(0.5, -0.18), 
            ncol=min(3, len(labels)), fontsize=11, frameon=True
        )

    plot_kwargs = {
        'palette': color_map,
        'markers': marker_map,
        'dashes': dash_map,
        'linewidth': 2.5,
        'markersize': 10,
        'markeredgewidth': 1.5,
        'alpha': 0.75
    }

    if not df_spheres.empty:
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        (ax1, ax2), (ax3, ax4) = axes
        
        metrics = [
            ('FPR', 'False Positive Rate (FPR)', ax1, 'FPR across sphere Count'),
            ('FNR', 'False Negative Rate (FNR)', ax2, 'FNR across sphere Count'),
            ('Precision', 'Precision', ax3, 'Precision across sphere Count'),
            ('F1-Score', 'F1-Score', ax4, 'F1-Score across sphere Count')
        ]
        
        for col, ylabel, ax, title in metrics:
            sns.lineplot(
                data=df_spheres, x='NumSpheres', y=col, hue='Config', style='Config',
                ax=ax, **plot_kwargs
            )
            ax.set_xlabel('Number of Spheres', fontsize=12, fontweight='bold')
            ax.set_ylabel(ylabel, fontsize=12, fontweight='bold')
            ax.set_ylim(-0.05, 1.05)
            ax.set_title(title, fontsize=13)
            if ax.get_legend() is not None:
                ax.get_legend().remove()
        
        handles, labels = ax1.get_legend_handles_labels()
        fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, 0.02), ncol=min(3, len(labels)), fontsize=12, frameon=True)
        
        plt.suptitle('Filter Accuracy Metrics across Sphere Allocations', fontsize=16, y=0.96)
        fig.subplots_adjust(bottom=0.12, top=0.92, hspace=0.3)
        
        filename1 = "accuracy_vs_spheres_plot.png"
        plt.savefig(f"{folder}/{filename1}", bbox_inches='tight', dpi=300)
        plt.close()
        print(f"Saved {folder}/{filename1}")

    if not df_memory.empty:
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        (ax1, ax2), (ax3, ax4) = axes
        
        metrics = [
            ('FPR', 'False Positive Rate (FPR)', ax1, 'FPR across Memory Size'),
            ('FNR', 'False Negative Rate (FNR)', ax2, 'FNR across Memory Size'),
            ('Precision', 'Precision', ax3, 'Precision across Memory Size'),
            ('F1-Score', 'F1-Score', ax4, 'F1-Score across Memory Size')
        ]
        
        for col, ylabel, ax, title in metrics:
            sns.lineplot(
                data=df_memory, x='MemoryBudget_MB', y=col, hue='Config', style='Config',
                ax=ax, **plot_kwargs
            )
            ax.set_xlabel('Memory Allocation (MB)', fontsize=12, fontweight='bold')
            ax.set_ylabel(ylabel, fontsize=12, fontweight='bold')
            ax.set_ylim(-0.05, 1.05)
            ax.set_title(title, fontsize=13)
            if ax.get_legend() is not None:
                ax.get_legend().remove()
        
        handles, labels = ax1.get_legend_handles_labels()
        fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, 0.02), ncol=min(3, len(labels)), fontsize=12, frameon=True)
        
        plt.suptitle('Filter Accuracy Metrics across Memory Budgets', fontsize=16, y=0.96)
        fig.subplots_adjust(bottom=0.12, top=0.92, hspace=0.3)
        
        filename2 = "accuracy_vs_memory_plot.png"
        plt.savefig(f"{folder}/{filename2}", bbox_inches='tight', dpi=300)
        plt.close()
        print(f"Saved {folder}/{filename2}")

    if not df_spheres.empty and 'QPS' in df_spheres.columns:
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.lineplot(
            data=df_spheres, x='NumSpheres', y='QPS', hue='Config', style='Config',
            ax=ax, **plot_kwargs
        )
        ax.set_xlabel('Number of Spheres', fontsize=12, fontweight='bold')
        ax.set_ylabel('Queries per Second (QPS)', fontsize=12, fontweight='bold')
        ax.set_title('Query Throughput across Sphere Count', fontsize=13)
        _add_bottom_legend(ax)
        
        filename3 = "qps_vs_spheres_plot.png"
        plt.savefig(f"{folder}/{filename3}", bbox_inches='tight', dpi=300)
        plt.close()
        print(f"Saved {folder}/{filename3}")

    if not df_memory.empty and 'QPS' in df_memory.columns:
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.lineplot(
            data=df_memory, x='MemoryBudget_MB', y='QPS', hue='Config', style='Config',
            ax=ax, **plot_kwargs
        )
        ax.set_xlabel('Memory Allocation (MB)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Queries per Second (QPS)', fontsize=12, fontweight='bold')
        ax.set_title('Query Throughput across Memory Size', fontsize=13)
        _add_bottom_legend(ax)
        
        filename4 = "qps_vs_memory_plot.png"
        plt.savefig(f"{folder}/{filename4}", bbox_inches='tight', dpi=300)
        plt.close()
        print(f"Saved {folder}/{filename4}")
        
    if not df_spheres.empty and 'ConstructTime' in df_spheres.columns:
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.lineplot(
            data=df_spheres, x='NumSpheres', y='ConstructTime', hue='Config', style='Config',
            ax=ax, **plot_kwargs
        )
        ax.set_xlabel('Number of Spheres', fontsize=12, fontweight='bold')
        ax.set_ylabel('Construct Time (Seconds)', fontsize=12, fontweight='bold')
        ax.set_title('Filter Build Time across Sphere Count', fontsize=13)
        _add_bottom_legend(ax)
        
        filename5 = "construct_time_vs_spheres_plot.png"
        plt.savefig(f"{folder}/{filename5}", bbox_inches='tight', dpi=300)
        plt.close()
        print(f"Saved {folder}/{filename5}")

    if not df_memory.empty and 'ConstructTime' in df_memory.columns:
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.lineplot(
            data=df_memory, x='MemoryBudget_MB', y='ConstructTime', hue='Config', style='Config',
            ax=ax, **plot_kwargs
        )
        ax.set_xlabel('Memory Allocation (MB)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Construct Time (Seconds)', fontsize=12, fontweight='bold')
        ax.set_title('Filter Build Time across Memory Size', fontsize=13)
        _add_bottom_legend(ax)
        
        filename6 = "construct_time_vs_memory_plot.png"
        plt.savefig(f"{folder}/{filename6}", bbox_inches='tight', dpi=300)
        plt.close()
        print(f"Saved {folder}/{filename6}")

    if not df_spheres.empty or not df_memory.empty:
        final_df = pd.concat([df_spheres, df_memory], ignore_index=True)

        filename7 = "tradeoff_evaluations.csv"
        os.makedirs(folder, exist_ok=True)
        final_df.to_csv(f"{folder}/{filename7}", index=False)
        print(f"Saved {folder}/{filename7}")

        print("\n=== Tradeoff Evaluation Results ===")
        print(final_df.to_string(index=False))