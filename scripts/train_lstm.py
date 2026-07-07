#!/usr/bin/env python3

from __future__ import annotations

import argparse

from qel_twin.training.lstm import train_lstm_model


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train an LSTM model on a P3 Lindblad .npz dataset."
    )

    parser.add_argument(
        "--dataset",
        type=str,
        default="data/processed/dataset_p3_lindblad.npz",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/ml_runs",
    )

    parser.add_argument(
        "--model-name",
        type=str,
        default="lstm",
    )

    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--test-size", type=float, default=0.15)
    parser.add_argument("--val-size", type=float, default=0.15)

    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=200)

    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)

    parser.add_argument(
        "--bidirectional",
        action="store_true",
        help="Use bidirectional LSTM.",
    )

    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--grad-clip", type=float, default=1.0)

    parser.add_argument(
        "--run-tag",
        type=str,
        default=None,
    )

    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Example: cpu, cuda, cuda:0. Default chooses cuda if available.",
    )

    args = parser.parse_args()

    train_lstm_model(
        dataset_path=args.dataset,
        output_dir=args.output_dir,
        model_name=args.model_name,
        seed=args.seed,
        test_size=args.test_size,
        val_size=args.val_size,
        batch_size=args.batch_size,
        epochs=args.epochs,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout,
        bidirectional=args.bidirectional,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        patience=args.patience,
        grad_clip=args.grad_clip,
        run_tag=args.run_tag,
        device_name=args.device,
    )


if __name__ == "__main__":
    main()