"""Geracao do relatorio comparativo do benchmark."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import psutil
import seaborn as sns

from benchmark_core import LOG_COLUMNS, cpu_name


REQUIRED_MODEL_FILES = {
    "MediaPipe": "log_mediapipe.csv",
    "YOLOv26 (PyTorch)": "log_yolo.csv",
    "YOLOv26 (ONNX Runtime)": "log_yolo_onnx.csv",
}
OPTIONAL_MODEL_FILES = {
    "YOLOv26 (OpenVINO FP16)": "log_yolo_openvino.csv",
}
MODEL_DISPLAY_NAMES = {
    "MediaPipe": "MediaPipe",
    "YOLOv26 (PyTorch)": "YOLOv26\n(PyTorch)",
    "YOLOv26 (ONNX Runtime)": "YOLOv26\n(ONNX Runtime)",
    "YOLOv26 (OpenVINO FP16)": "YOLOv26\n(OpenVINO FP16)",
}


def missing_logs(project_dir: Path) -> list[str]:
    project_dir = Path(project_dir)
    missing = []
    for model, filename in REQUIRED_MODEL_FILES.items():
        path = project_dir / filename
        if not path.exists() or path.stat().st_size == 0:
            missing.append(model)
    return missing


def _has_samples(path: Path) -> bool:
    try:
        return not pd.read_csv(path, nrows=1).empty
    except (OSError, pd.errors.ParserError):
        return False


def model_files_for_report(project_dir: Path) -> dict[str, str]:
    project_dir = Path(project_dir)
    files = dict(REQUIRED_MODEL_FILES)
    for model, filename in OPTIONAL_MODEL_FILES.items():
        if _has_samples(project_dir / filename):
            files[model] = filename
    return files


def _load_log(path: Path, model: str) -> pd.DataFrame:
    frame = pd.read_csv(path, encoding="utf-8-sig")
    absent = [column for column in LOG_COLUMNS if column not in frame.columns]
    if absent:
        raise ValueError(f"{path.name} nao possui as colunas: {', '.join(absent)}")
    numeric = ["FPS", "CPU_Usage_Percent", "RAM_Usage_MB", "Nose_X", "Nose_Y", "Head_Dist"]
    for column in numeric:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.dropna(subset=numeric).reset_index(drop=True)
    if frame.empty:
        raise ValueError(f"{path.name} nao contem amostras validas")
    # A coleta preserva o valor bruto. Para comparacao no relatorio, CPU e
    # normalizada para o intervalo percentual convencional de 0 a 100.
    frame["CPU_Usage_Percent"] = frame["CPU_Usage_Percent"].clip(lower=0, upper=100)
    frame["Modelo"] = model
    frame["Amostra"] = range(1, len(frame) + 1)
    frame["Jitter_Nose_Y"] = frame["Nose_Y"].diff().abs()
    return frame


def _summary(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for model, frame in frames.items():
        metrics = {
            "FPS": frame["FPS"],
            "CPU (%)": frame["CPU_Usage_Percent"],
            "RAM (MB)": frame["RAM_Usage_MB"],
            "Jitter": frame["Jitter_Nose_Y"].dropna(),
        }
        for metric, values in metrics.items():
            rows.append(
                {
                    "Modelo": model,
                    "Metrica": metric,
                    "Minimo": values.min(),
                    "Media": values.mean(),
                    "Maximo": values.max(),
                    "Desvio_Padrao": values.std(),
                    "Amostras": len(values),
                }
            )
    return pd.DataFrame(rows).round(4)


def generate_report(project_dir: Path, camera_fps_limit: float | None = None) -> dict[str, object]:
    project_dir = Path(project_dir)
    missing = missing_logs(project_dir)
    if missing:
        raise FileNotFoundError("Logs ausentes: " + ", ".join(missing))

    model_files = model_files_for_report(project_dir)
    frames = {
        model: _load_log(project_dir / filename, model)
        for model, filename in model_files.items()
    }
    combined = pd.concat(frames.values(), ignore_index=True)
    summary = _summary(frames)

    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(3, 2, figsize=(20, 17), gridspec_kw={"height_ratios": [1, 1, 0.58]})
    palette = {
        "MediaPipe": "#2ea8ff",
        "YOLOv26 (PyTorch)": "#9b59ff",
        "YOLOv26 (ONNX Runtime)": "#35b779",
        "YOLOv26 (OpenVINO FP16)": "#f5a623",
    }
    sns.lineplot(data=combined, x="Amostra", y="FPS", hue="Modelo", palette=palette, ax=axes[0, 0])
    axes[0, 0].set_title("Performance - FPS")
    sns.lineplot(data=combined, x="Amostra", y="CPU_Usage_Percent", hue="Modelo", palette=palette, ax=axes[0, 1])
    axes[0, 1].set_title("Eficiencia - uso de CPU (%)")
    ram_models = list(frames)
    display_names = [MODEL_DISPLAY_NAMES[model] for model in ram_models]
    ram_means = [frames[model]["RAM_Usage_MB"].mean() for model in ram_models]
    ram_mins = [frames[model]["RAM_Usage_MB"].min() for model in ram_models]
    ram_maxs = [frames[model]["RAM_Usage_MB"].max() for model in ram_models]
    ram_errors = [
        [mean - minimum for mean, minimum in zip(ram_means, ram_mins)],
        [maximum - mean for mean, maximum in zip(ram_means, ram_maxs)],
    ]
    ram_bars = axes[1, 0].bar(
        range(len(ram_models)),
        ram_means,
        color=[palette[model] for model in ram_models],
        edgecolor="#4b5563",
        linewidth=1,
        alpha=0.9,
        yerr=ram_errors,
        capsize=6,
    )
    axes[1, 0].bar_label(ram_bars, labels=[f"{value:.1f} MB" for value in ram_means], padding=4, fontweight="bold")
    axes[1, 0].set_ylim(0, max(ram_maxs) * 1.15)
    axes[1, 0].set_title("Consumo medio de RAM (MB)")
    axes[1, 0].set_xticks(range(len(ram_models)), display_names, fontsize=10)
    axes[1, 0].set_xlabel("Modelo")
    axes[1, 0].set_ylabel("RAM (MB)")
    jitter = combined.dropna(subset=["Jitter_Nose_Y"])
    sns.boxplot(data=jitter, x="Modelo", y="Jitter_Nose_Y", hue="Modelo", order=ram_models, palette=palette, legend=False, ax=axes[1, 1])
    axes[1, 1].set_title("Estabilidade - jitter do nariz")
    axes[1, 1].set_yscale("symlog", linthresh=1e-5)
    axes[1, 1].set_xticks(range(len(ram_models)), display_names, fontsize=10)

    for axis, metric, title in (
        (axes[2, 0], "FPS", "FPS — minimo, media e maximo"),
        (axes[2, 1], "CPU (%)", "CPU — minimo, media e maximo (%)"),
    ):
        metric_summary = summary[summary["Metrica"] == metric].set_index("Modelo").reindex(list(frames))
        axis.axis("off")
        table = axis.table(
            cellText=[
                [MODEL_DISPLAY_NAMES[model], f"{row.Minimo:.2f}", f"{row.Media:.2f}", f"{row.Maximo:.2f}"]
                for model, row in metric_summary.iterrows()
            ],
            colLabels=["Modelo", "Minimo", "Media", "Maximo"],
            colWidths=[0.38, 0.205, 0.205, 0.21],
            cellLoc="center",
            colLoc="center",
            loc="center",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        table.scale(1, 1.9)
        for column, model in enumerate(metric_summary.index):
            table[(column + 1, 0)].set_facecolor(palette[model])
        for column in range(4):
            table[(0, column)].set_facecolor("#dce6f2")
            table[(0, column)].set_text_props(weight="bold")
        axis.set_title(title, fontweight="bold", pad=8)

    total_ram_gb = psutil.virtual_memory().total / (1024**3)
    hardware = {
        "cpu": cpu_name(),
        "ram_gb": round(total_ram_gb, 1),
        "camera_fps_limit": camera_fps_limit,
    }
    camera_fps_text = (
        f"{camera_fps_limit:.0f} FPS" if camera_fps_limit is not None else "Nao informado"
    )
    fig.suptitle(
        f"Benchmark de postura\nCPU: {hardware['cpu']} | RAM: {hardware['ram_gb']} GB"
        f"\nLimite de FPS da Camera Usada no Teste - {camera_fps_text}",
        fontsize=14,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0.01, 0.01, 0.99, 0.90), h_pad=2.4, w_pad=2.2)
    image_path = project_dir / "comparativo_modelos.png"
    fig.savefig(image_path, dpi=220, bbox_inches="tight")
    plt.close(fig)

    csv_path = project_dir / "resumo_estatistico.csv"
    summary.to_csv(csv_path, index=False, encoding="utf-8-sig")
    markdown_path = project_dir / "relatorio_benchmark.md"
    table_header = "| Modelo | Metrica | Minimo | Media | Maximo | Desvio padrao | Amostras |"
    table_rule = "|---|---|---:|---:|---:|---:|---:|"
    table_rows = [
        f"| {row.Modelo} | {row.Metrica} | {row.Minimo:.4f} | {row.Media:.4f} | {row.Maximo:.4f} | {row.Desvio_Padrao:.4f} | {int(row.Amostras)} |"
        for row in summary.itertuples(index=False)
    ]
    cpu_fps_rows = [
        f"| {row.Modelo} | {row.Metrica} | {row.Minimo:.4f} | {row.Media:.4f} | {row.Maximo:.4f} |"
        for row in summary[summary["Metrica"].isin(["FPS", "CPU (%)"])].itertuples(index=False)
    ]
    markdown = [
        "# Relatorio do benchmark de postura",
        "",
        "## Hardware",
        "",
        f"- CPU: {hardware['cpu']}",
        f"- RAM instalada: {hardware['ram_gb']} GB",
        f"- Limite de FPS da camera usada no teste: {camera_fps_text}",
        "- CPU no relatorio: valores normalizados para 0–100%; os logs brutos permanecem intactos",
        "",
        "## FPS e CPU — minimo, media e maximo",
        "",
        "| Modelo | Metrica | Minimo | Media | Maximo |",
        "|---|---|---:|---:|---:|",
        *cpu_fps_rows,
        "",
        "## Resumo estatistico",
        "",
        table_header,
        table_rule,
        *table_rows,
        "",
        "Os logs sao aditivos. Arquive os resultados anteriores no launcher antes de iniciar um novo experimento independente.",
    ]
    markdown_path.write_text("\n".join(markdown), encoding="utf-8")

    return {
        "image": image_path,
        "summary_csv": csv_path,
        "markdown": markdown_path,
        "summary": summary,
        "hardware": hardware,
    }
