"""
Evaluation script for polyp size classification

Loads trained model and evaluates on test set
Generates comprehensive reports and visualizations
"""

import torch
import yaml
import argparse
from pathlib import Path
import numpy as np
from tqdm import tqdm
import json
from datetime import datetime

from src.datasets.rgbd_dataset import create_dataloaders
from src.datasets.depth_only_dataset import create_depth_dataloaders
from src.models.bsenet import create_bsenet_from_config
from src.models.resnet_rgb import create_resnet_rgb_from_config
from src.models.vit_rgbd import create_vit_rgbd_from_config
from src.utils.metrics import (
    compute_metrics,
    compute_confusion_matrix,
    plot_confusion_matrix,
    plot_reliability_diagram,
    format_metrics_for_report
)


class Evaluator:
    """Evaluator for trained models"""

    def __init__(self, config: dict, checkpoint_path: str, split: str = 'test'):
        self.config = config
        self.checkpoint_path = Path(checkpoint_path)
        self.split = split

        # Setup device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")

        # Create output directory for evaluation results
        self.output_dir = Path(config['logging']['output_dir']) / config['logging']['experiment_name']
        self.eval_dir = self.output_dir / f'evaluation_{split}'
        self.eval_dir.mkdir(parents=True, exist_ok=True)

        self.scenario_id = config.get('scenario', {}).get('id', 1)
        self.arch = str(config.get('model', {}).get('architecture', 'vit_rgbd')).lower()

        # Create dataloaders
        print(f"\nCreating {split} dataloader...")
        data_cfg = config.get('data', {})
        loader_mode = str(data_cfg.get('loader', '')).lower()
        if loader_mode in {'depth', 'depth_only', 'bsenet'} or self.arch == 'bsenet':
            dataloaders = create_depth_dataloaders(config)
            datasets = dataloaders.get('datasets', {})
            dataloaders['train_dataset'] = datasets.get('train')
            dataloaders['val_dataset'] = datasets.get('val')
            dataloaders['test_dataset'] = datasets.get('test')
        else:
            dataloaders = create_dataloaders(config, scenario_id=self.scenario_id)
        self.dataloader = dataloaders[split]
        self.dataset = dataloaders[f'{split}_dataset']

        # Create model
        print("\nCreating model...")
        if self.arch == 'photometry_mlp':
            from src.models.size_models import create_photometry_mlp_from_config
            self.model = create_photometry_mlp_from_config(config)
        elif self.arch == 'bsenet':
            self.model = create_bsenet_from_config(config)
        elif self.arch in {'rgb_resnet', 'resnet_rgb'}:
            self.model = create_resnet_rgb_from_config(config)
        elif self.arch in {'vit_rgb', 'vit_rgbd'}:
            self.model = create_vit_rgbd_from_config(config)
        else:
            raise ValueError(f"Unsupported architecture '{self.arch}'")

        # Load checkpoint
        print(f"\nLoading checkpoint from {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model = self.model.to(self.device)
        self.model.eval()

        print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
        if 'metrics' in checkpoint:
            print(f"Checkpoint val metrics:")
            print(f"  Macro-F1: {checkpoint['metrics']['macro_f1']:.4f}")
            print(f"  Accuracy: {checkpoint['metrics']['accuracy']:.4f}")

        # Class names
        self.class_names = config['evaluation'].get('class_names', ['le_5mm', 'gt_5mm'])

    @torch.no_grad()
    def evaluate(self):
        """Run evaluation on test set"""
        print(f"\nEvaluating on {self.split} set...")

        all_preds = []
        all_labels = []
        all_probs = []
        all_image_names = []
        all_sizes_mm = []

        for batch in tqdm(self.dataloader, desc="Evaluating"):
            images = batch.get('image')
            if images is not None:
                images = images.to(self.device)
            labels = batch['label']
            features = batch.get('features')
            if features is not None:
                features = features.to(self.device)

            # Forward pass
            if self.arch == 'photometry_mlp':
                outputs = self.model(features)
            else:
                outputs = self.model(images)
                if isinstance(outputs, (tuple, list)):
                    outputs = outputs[0]
            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)

            # Collect
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs.cpu().numpy())
            all_image_names.extend(batch['image_name'])
            all_sizes_mm.extend(batch['size_mm'].numpy())

        # Convert to arrays
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        all_probs = np.array(all_probs)
        all_sizes_mm = np.array(all_sizes_mm)

        print(f"\nEvaluated {len(all_preds)} samples")

        # Compute metrics
        print("\nComputing metrics...")
        metrics = compute_metrics(all_labels, all_preds, all_probs, self.class_names)

        # Compute confusion matrix
        cm = compute_confusion_matrix(all_labels, all_preds, self.class_names)

        # Print metrics
        print("\n" + "="*80)
        print("Evaluation Results")
        print("="*80)
        print(f"Macro-F1: {metrics['macro_f1']:.4f}")
        print(f"Balanced Accuracy: {metrics['balanced_accuracy']:.4f}")
        print(f"Accuracy: {metrics['accuracy']:.4f}")

        for name in self.class_names:
            f1 = metrics.get(f'{name}_f1', 0)
            prec = metrics.get(f'{name}_precision', 0)
            rec = metrics.get(f'{name}_recall', 0)
            print(f"\n{name}:")
            print(f"  F1: {f1:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}")

        # Save results
        self.save_results(metrics, cm, all_preds, all_labels, all_probs,
                         all_image_names, all_sizes_mm)

        return metrics, cm

    def save_results(self, metrics, cm, preds, labels, probs, image_names, sizes_mm):
        """Save evaluation results"""
        # Save metrics as JSON
        metrics_file = self.eval_dir / 'metrics.json'
        with open(metrics_file, 'w') as f:
            # Convert numpy types to Python types
            metrics_serializable = {k: float(v) if isinstance(v, (np.floating, np.integer)) else v
                                   for k, v in metrics.items()}
            json.dump(metrics_serializable, f, indent=2)
        print(f"\nSaved metrics to {metrics_file}")

        # Save predictions as CSV
        predictions_file = self.eval_dir / 'predictions.csv'
        import pandas as pd
        prediction_data = {
            'image_name': image_names,
            'true_label': labels,
            'pred_label': preds,
            'true_class': [self.class_names[l] for l in labels],
            'pred_class': [self.class_names[p] for p in preds],
            'size_mm': sizes_mm,
            'correct': (labels == preds)
        }
        for idx, name in enumerate(self.class_names):
            prediction_data[f'prob_{name}'] = probs[:, idx]
        df = pd.DataFrame(prediction_data)
        df.to_csv(predictions_file, index=False)
        print(f"Saved predictions to {predictions_file}")

        # Save confusion matrix plot
        cm_file = self.eval_dir / 'confusion_matrix.png'
        plot_confusion_matrix(cm, self.class_names, save_path=cm_file, normalize=True)
        print(f"Saved confusion matrix to {cm_file}")

        # Save reliability diagram
        reliability_file = self.eval_dir / 'reliability_diagram.png'
        plot_reliability_diagram(labels, probs, save_path=reliability_file)
        print(f"Saved reliability diagram to {reliability_file}")

        # Save markdown report
        report = format_metrics_for_report(metrics, cm, self.class_names)
        report_file = self.eval_dir / 'report.md'
        with open(report_file, 'w') as f:
            f.write(f"# Evaluation Report\n\n")
            f.write(f"**Probe:** {self.config.get('scenario', {}).get('name', self.arch)}\n\n")
            f.write(f"**Split:** {self.split}\n\n")
            f.write(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(f"**Checkpoint:** {self.checkpoint_path.name}\n\n")
            f.write("---\n\n")
            f.write(report)
        print(f"Saved report to {report_file}")

        # Analyze errors
        self.analyze_errors(df)

    def analyze_errors(self, df):
        """Analyze common error patterns"""
        print("\n" + "="*80)
        print("Error Analysis")
        print("="*80)

        errors = df[~df['correct']]
        n_errors = len(errors)
        total = len(df)

        print(f"Total errors: {n_errors}/{total} ({n_errors/total*100:.1f}%)")

        if n_errors > 0:
            # Error breakdown by true class
            print("\nErrors by true class:")
            for name in self.class_names:
                class_errors = errors[errors['true_class'] == name]
                class_total = len(df[df['true_class'] == name])
                if class_total > 0:
                    print(f"  {name}: {len(class_errors)}/{class_total} ({len(class_errors)/class_total*100:.1f}%)")

            # Confusion pairs
            print("\nMost common confusion pairs:")
            confusion_pairs = errors.groupby(['true_class', 'pred_class']).size().sort_values(ascending=False)
            for (true_cls, pred_cls), count in confusion_pairs.head(5).items():
                print(f"  {true_cls} → {pred_cls}: {count}")

            # Size analysis for errors
            if 'size_mm' in df.columns:
                print("\nSize distribution of errors:")
                for name in self.class_names:
                    class_errors = errors[errors['true_class'] == name]
                    if len(class_errors) > 0 and class_errors['size_mm'].mean() > 0:
                        print(f"  {name}: mean={class_errors['size_mm'].mean():.2f}mm, "
                              f"std={class_errors['size_mm'].std():.2f}mm")


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate trained polyp classification model")

    parser.add_argument('--config', type=str, required=True,
                        help="Path to config YAML file")
    parser.add_argument('--checkpoint', type=str, required=True,
                        help="Path to model checkpoint")
    parser.add_argument('--split', type=str, default='test',
                        choices=['train', 'val', 'test'],
                        help="Which split to evaluate on")

    return parser.parse_args()


def main():
    # Parse arguments
    args = parse_args()

    # Load config
    with open(args.config) as f:
        config = yaml.safe_load(f)

    # Print header
    print("="*80)
    print(f"Polyp Size Classification Evaluation")
    print("="*80)
    print(f"Probe: {config.get('scenario', {}).get('name', config.get('model', {}).get('architecture', 'probe'))}")
    print(f"Config: {args.config}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Split: {args.split}")
    print("="*80)

    # Create evaluator
    evaluator = Evaluator(config, args.checkpoint, split=args.split)

    # Evaluate
    metrics, cm = evaluator.evaluate()

    print("\n✅ Evaluation completed!")
    print(f"Results saved to: {evaluator.eval_dir}")


if __name__ == "__main__":
    main()
