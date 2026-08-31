from __future__ import annotations

import argparse
import json

from qel_twin.training.lstm_noise import train_lstm_noise_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train LSTM/BiLSTM models on qel-ml NoiseDataset NPZ files."
    )
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output-dir", default="outputs/noise_ml_runs")
    parser.add_argument("--model", choices=["lstm", "bilstm"], default="lstm")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--train-fraction", type=float, default=0.60)
    parser.add_argument("--validation-fraction", type=float, default=0.20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--run-tag", default=None)
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = train_lstm_noise_model(
        dataset_path=args.dataset,
        output_dir=args.output_dir,
        model_name=args.model,
        seed=args.seed,
        train_fraction=args.train_fraction,
        validation_fraction=args.validation_fraction,
        batch_size=args.batch_size,
        epochs=args.epochs,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        patience=args.patience,
        grad_clip=args.grad_clip,
        run_tag=args.run_tag,
        device_name=args.device,
    )
    print(json.dumps({"run_dir": result["run_dir"]}, indent=2))


if __name__ == "__main__":
    main()
