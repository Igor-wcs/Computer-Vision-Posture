"""Entrada de compatibilidade para gerar o novo relatorio pela linha de comando."""

from pathlib import Path

from reporting import generate_report, missing_logs


def generate_comparative_graphs():
    project_dir = Path(__file__).resolve().parent
    missing = missing_logs(project_dir)
    if missing:
        print("Nao foi possivel gerar o relatorio.")
        print("Logs ausentes: " + ", ".join(missing))
        return None
    result = generate_report(project_dir)
    print("Relatorio gerado com sucesso:")
    print(f"- {result['image'].name}")
    print(f"- {result['summary_csv'].name}")
    print(f"- {result['markdown'].name}")
    return result


if __name__ == "__main__":
    generate_comparative_graphs()
