import os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.style as style
from matplotlib.backends.backend_pdf import PdfPages

def moving_average(data, window_size=5):
    """Calculate moving average with given window size"""
    weights = np.ones(window_size) / window_size
    return np.convolve(data, weights, mode='valid')

def plot_metrics(output_dir):
    """
    Plot training metrics from training_logs directory.
    Creates PDF with separate plots for each metric over training steps.
    Uses a modern, professional style with custom color palette.
    """
    # Load training logs
    train_logs_path = os.path.join(output_dir, 'training_logs', 'train_logs.json')
    with open(train_logs_path, 'r') as f:
        train_logs = json.load(f)

    # Load evaluation logs
    eval_logs = {}
    eval_logs_dir = os.path.join(output_dir, 'eval_logs')
    for filename in os.listdir(eval_logs_dir):
        if filename.startswith('metrics_') and filename.endswith('.json'):
            step = int(filename.split('_')[1].split('.')[0])
            with open(os.path.join(eval_logs_dir, filename), 'r') as f:
                eval_logs[step] = json.load(f)

    # Set style and color palette
    plt.style.use('bmh')  # Using 'bmh' style which is a modern, clean style
    colors = ['#2ecc71', '#e74c3c', '#3498db', '#f1c40f', '#9b59b6', '#1abc9c', '#e67e22', '#34495e']
    
    # Create PDF to save all plots
    pdf_path = os.path.join(output_dir, 'training_plots.pdf')
    with PdfPages(pdf_path) as pdf:
        
        # Plot reward metrics
        reward_metrics = [
            'rewards/correctness_reward_func',
            'rewards/int_reward_func',
            'rewards/strict_format_reward_func',
            'rewards/soft_format_reward_func',
            'rewards/xmlcount_reward_func',
            'reward'
        ]
        
        for metric, color in zip(reward_metrics, colors):
            plt.figure(figsize=(12,7))
            steps = [int(x) for x in train_logs.keys()]
            values = [metrics[metric] for metrics in train_logs.values()]
            
            # Plot raw data with low alpha
            plt.plot(steps, values, color=color, alpha=0.3, linewidth=1.5, label='Raw data')
            
            # Calculate and plot moving average if we have enough data points
            if len(values) > 5:
                ma_values = moving_average(values)
                ma_steps = steps[len(steps)-len(ma_values):]
                plt.plot(ma_steps, ma_values, color=color, linewidth=2.5, label='Moving average')
            
            plt.xlabel('Training Steps', fontsize=12)
            plt.ylabel(f'{metric.split("/")[-1].replace("_", " ").title()}', fontsize=12)
            plt.title(f'{metric.split("/")[-1].replace("_", " ").title()}', fontsize=14, pad=20)
            plt.grid(True, alpha=0.3)
            plt.legend()
            pdf.savefig(bbox_inches='tight')
            plt.close()

        # Plot learning rate
        plt.figure(figsize=(12,7))
        steps = [int(x) for x in train_logs.keys()]
        lr_values = [metrics['learning_rate'] for metrics in train_logs.values()]

        plt.plot(steps, lr_values, color='#e74c3c', linewidth=2.0, label='Learning Rate')
        
        plt.xlabel('Training Steps', fontsize=12)
        plt.ylabel('Learning Rate', fontsize=12)
        plt.title('Learning Rate Schedule', fontsize=14, pad=20)
        plt.grid(True, alpha=0.3)
        plt.legend()
        pdf.savefig(bbox_inches='tight')
        plt.close()

        # Plot reward standard deviation
        plt.figure(figsize=(12,7))
        reward_std = [metrics['reward_std'] for metrics in train_logs.values()]

        plt.plot(steps, reward_std, color='#3498db', alpha=0.3, linewidth=1.5, label='Reward Std (Raw)')
        if len(reward_std) > 5:
            ma_std = moving_average(reward_std)
            ma_steps = steps[len(steps)-len(ma_std):]
            plt.plot(ma_steps, ma_std, color='#3498db', linewidth=2.5, label='Reward Std (MA)')

        plt.xlabel('Training Steps', fontsize=12)
        plt.ylabel('Standard Deviation', fontsize=12)
        plt.title('Reward Standard Deviation', fontsize=14, pad=20)
        plt.grid(True, alpha=0.3)
        plt.legend()
        pdf.savefig(bbox_inches='tight')
        plt.close()

        # Plot loss
        plt.figure(figsize=(12,7))
        loss_values = [metrics['loss'] for metrics in train_logs.values()]

        plt.plot(steps, loss_values, color='#e67e22', alpha=0.3, linewidth=1.5, label='Loss (Raw)')
        if len(loss_values) > 5:
            ma_loss = moving_average(loss_values)
            ma_steps = steps[len(steps)-len(ma_loss):]
            plt.plot(ma_steps, ma_loss, color='#e67e22', linewidth=2.5, label='Loss (MA)')

        plt.xlabel('Training Steps', fontsize=12)
        plt.ylabel('Loss', fontsize=12)
        plt.title('Training Loss', fontsize=14, pad=20)
        plt.grid(True, alpha=0.3)
        plt.legend()
        pdf.savefig(bbox_inches='tight')
        plt.close()

        # Plot KL divergence
        plt.figure(figsize=(12,7))
        kl_values = [metrics['kl'] for metrics in train_logs.values()]

        plt.plot(steps, kl_values, color='#9b59b6', alpha=0.3, linewidth=1.5, label='KL Divergence (Raw)')
        if len(kl_values) > 5:
            ma_kl = moving_average(kl_values)
            ma_steps = steps[len(steps)-len(ma_kl):]
            plt.plot(ma_steps, ma_kl, color='#9b59b6', linewidth=2.5, label='KL Divergence (MA)')

        plt.xlabel('Training Steps', fontsize=12)
        plt.ylabel('KL Divergence', fontsize=12)
        plt.title('KL Divergence', fontsize=14, pad=20)
        plt.grid(True, alpha=0.3)
        plt.legend()
        pdf.savefig(bbox_inches='tight')
        plt.close()

        # Plot correlation metrics (both on same plot for comparison)
        plt.figure(figsize=(12,7))
        corr_length = [metrics.get('corr_advantage_length', 0) for metrics in train_logs.values()]
        corr_kl = [metrics.get('corr_advantage_kl', 0) for metrics in train_logs.values()]

        plt.plot(steps, corr_length, color='#1abc9c', alpha=0.3, linewidth=1.5)
        plt.plot(steps, corr_kl, color='#e74c3c', alpha=0.3, linewidth=1.5)

        if len(corr_length) > 5:
            ma_corr_length = moving_average(corr_length)
            ma_corr_kl = moving_average(corr_kl)
            ma_steps = steps[len(steps)-len(ma_corr_length):]
            plt.plot(ma_steps, ma_corr_length, color='#1abc9c', linewidth=2.5, label='Corr(Advantage, Length)')
            plt.plot(ma_steps, ma_corr_kl, color='#e74c3c', linewidth=2.5, label='Corr(Advantage, KL)')

        plt.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        plt.xlabel('Training Steps', fontsize=12)
        plt.ylabel('Correlation', fontsize=12)
        plt.title('Advantage Correlations', fontsize=14, pad=20)
        plt.ylim(-1, 1)
        plt.grid(True, alpha=0.3)
        plt.legend()
        pdf.savefig(bbox_inches='tight')
        plt.close()

        # Plot evaluation metrics
        if eval_logs:
            eval_steps = sorted(eval_logs.keys())
            
            # Plot accuracy
            plt.figure(figsize=(12,7))
            accuracy_values = [eval_logs[step]['accuracy'] for step in eval_steps]
            plt.plot(eval_steps, accuracy_values, color='#2ecc71', linewidth=2.0, label='Accuracy')
            plt.xlabel('Training Steps', fontsize=12)
            plt.ylabel('Accuracy (%)', fontsize=12)
            plt.title('Evaluation Accuracy', fontsize=14, pad=20)
            plt.grid(True, alpha=0.3)
            plt.legend()
            pdf.savefig(bbox_inches='tight')
            plt.close()

            # Plot evaluation reward metrics
            eval_metrics = [key for key in eval_logs[eval_steps[0]]['metrics'].keys()]
            for metric, color in zip(eval_metrics, colors):
                plt.figure(figsize=(12,7))
                metric_values = [eval_logs[step]['metrics'][metric] for step in eval_steps]
                plt.plot(eval_steps, metric_values, color=color, linewidth=2.0, label=metric)
                plt.xlabel('Training Steps', fontsize=12)
                plt.ylabel(metric.replace('_', ' ').title(), fontsize=12)
                plt.title(f'Evaluation {metric.replace("_", " ").title()}', fontsize=14, pad=20)
                plt.grid(True, alpha=0.3)
                plt.legend()
                pdf.savefig(bbox_inches='tight')
                plt.close()

def compare_runs(output_dirs: list[str], labels: list[str] = None, output_path: str = "comparison_plots.pdf"):
    """
    Compare training curves from multiple runs on the same axes.

    Args:
        output_dirs: List of output directories containing training logs
        labels: Optional list of labels for each run (defaults to directory names)
        output_path: Path to save the comparison PDF
    """
    if labels is None:
        labels = [os.path.basename(d.rstrip('/')) for d in output_dirs]

    # Color palette for different runs
    run_colors = ['#2ecc71', '#e74c3c', '#3498db', '#f1c40f', '#9b59b6', '#1abc9c', '#e67e22', '#34495e']

    # Load training logs from all directories
    all_train_logs = {}
    all_eval_logs = {}

    for output_dir, label in zip(output_dirs, labels):
        train_logs_path = os.path.join(output_dir, 'training_logs', 'train_logs.json')
        if os.path.exists(train_logs_path):
            with open(train_logs_path, 'r') as f:
                all_train_logs[label] = json.load(f)

        eval_logs_dir = os.path.join(output_dir, 'eval_logs')
        if os.path.exists(eval_logs_dir):
            eval_logs = {}
            for filename in os.listdir(eval_logs_dir):
                if filename.startswith('metrics_') and filename.endswith('.json'):
                    step = int(filename.split('_')[1].split('.')[0])
                    with open(os.path.join(eval_logs_dir, filename), 'r') as f:
                        eval_logs[step] = json.load(f)
            if eval_logs:
                all_eval_logs[label] = eval_logs

    if not all_train_logs:
        print("No training logs found in any directory")
        return

    plt.style.use('bmh')

    # Metrics to plot
    training_metrics = [
        'reward', 'loss', 'kl', 'learning_rate', 'reward_std', 'grad_norm',
        'rewards/correctness_reward_func', 'rewards/int_reward_func',
        'rewards/strict_format_reward_func', 'rewards/soft_format_reward_func',
        'rewards/xmlcount_reward_func', 'corr_advantage_length', 'corr_advantage_kl'
    ]

    with PdfPages(output_path) as pdf:
        # Plot each training metric
        for metric in training_metrics:
            # Check if metric exists in at least one run
            metric_exists = any(
                metric in list(logs.values())[0]
                for logs in all_train_logs.values()
            )
            if not metric_exists:
                continue

            plt.figure(figsize=(12, 7))

            for (label, train_logs), color in zip(all_train_logs.items(), run_colors):
                steps = [int(x) for x in train_logs.keys()]

                # Get values, using None for missing metrics
                values = []
                for metrics in train_logs.values():
                    values.append(metrics.get(metric, None))

                # Skip if all None
                if all(v is None for v in values):
                    continue

                # Replace None with 0 for plotting
                values = [v if v is not None else 0 for v in values]

                # Plot with moving average
                plt.plot(steps, values, color=color, alpha=0.2, linewidth=1)
                if len(values) > 5:
                    ma_values = moving_average(values)
                    ma_steps = steps[len(steps)-len(ma_values):]
                    plt.plot(ma_steps, ma_values, color=color, linewidth=2.5, label=label)
                else:
                    plt.plot(steps, values, color=color, linewidth=2.5, label=label)

            # Add zero line for correlation metrics
            if 'corr' in metric and 'correct' not in metric:
                plt.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
                plt.ylim(-1, 1)

            metric_name = metric.split('/')[-1].replace('_', ' ').title()
            plt.xlabel('Training Steps', fontsize=12)
            plt.ylabel(metric_name, fontsize=12)
            plt.title(f'{metric_name} Comparison', fontsize=14, pad=20)
            plt.grid(True, alpha=0.3)
            plt.legend()
            pdf.savefig(bbox_inches='tight')
            plt.close()

        # Plot evaluation accuracy comparison
        if all_eval_logs:
            plt.figure(figsize=(12, 7))

            for (label, eval_logs), color in zip(all_eval_logs.items(), run_colors):
                eval_steps = sorted(eval_logs.keys())
                accuracy_values = [eval_logs[step]['accuracy'] for step in eval_steps]
                plt.plot(eval_steps, accuracy_values, color=color, linewidth=2.0,
                        marker='o', markersize=4, label=label)

            plt.xlabel('Training Steps', fontsize=12)
            plt.ylabel('Accuracy (%)', fontsize=12)
            plt.title('Evaluation Accuracy Comparison', fontsize=14, pad=20)
            plt.grid(True, alpha=0.3)
            plt.legend()
            pdf.savefig(bbox_inches='tight')
            plt.close()

    print(f"Comparison plots saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Plot training metrics from logs directory')
    parser.add_argument('--log_dir', type=str, help='Directory containing training logs')
    parser.add_argument('--compare', type=str, nargs='+', help='Compare multiple runs (list of output directories)')
    parser.add_argument('--labels', type=str, nargs='+', help='Labels for compared runs')
    parser.add_argument('--output', type=str, default='comparison_plots.pdf', help='Output path for comparison PDF')
    args = parser.parse_args()

    if args.compare:
        compare_runs(args.compare, args.labels, args.output)
    elif args.log_dir:
        plot_metrics(args.log_dir)
    else:
        print("Please provide either --log_dir or --compare")