from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def draw_line_plot(
    data,
    label_x: str,
    label_y: str,
    title: str,
    label_line: str = " ",
    img_path: str | Path = "",
):
    """Save a simple line plot and return its path."""
    out_dir = Path(img_path) if img_path else Path(".")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{title}.png"

    plt.figure()
    plt.plot(data, color="blue", label=label_line)
    plt.xlabel(label_x)
    plt.ylabel(label_y)
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    return out_path


def draw_curves(
    train_values,
    val_values,
    x=None,
    label_x: str = "step",
    label_y: str = "loss",
    title: str = "loss",
    img_path: str | Path = "",
):
    """Save train/validation curves on one figure."""
    out_dir = Path(img_path) if img_path else Path(".")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{title}.png"

    plt.figure()
    if x is None:
        plt.plot(train_values, color="blue", label="train")
        plt.plot(val_values, color="orange", label="val")
    else:
        plt.plot(x, train_values, color="blue", label="train")
        plt.plot(x, val_values, color="orange", label="val")
    plt.xlabel(label_x)
    plt.ylabel(label_y)
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    return out_path
