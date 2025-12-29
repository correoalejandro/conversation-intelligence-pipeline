#!/usr/bin/env python3
"""
3.v7.uploadToCAS.py – Upload experiment to CAS with httpConnSwat
===============================================================
* Lista los 20 experimentos más recientes (o toma `-x`).
* **No** fusiona con master: sube el DataFrame tal cual.
* Conecta a CAS usando tu helper `app_refresh_tokenRafael.httpConnSwat()`.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import List, Optional
import sys
sys.path.append("c:/Projects/clasificador_mensajes")
import click
import joblib
from app_refresh_tokenRafael import httpConnSwat  # helper propio

DEF_EXP_DIR = Path("data/experiments")
CASLIB_NAME = "CASUSER"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def list_experiments(exp_dir: Path, n: int = 40) -> List[Path]:
    files = sorted(exp_dir.glob("*.joblib"), key=lambda p: p.stat().st_mtime, reverse=True)
    return files[:n]


def pick_experiment_menu(files: List[Path]) -> Path:
    click.echo("\nExperimentos recientes:")
    for i, p in enumerate(files, 1):
        ts = datetime.fromtimestamp(p.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
        click.echo(f"[{i}] {p.name:50} | {ts}")
    idx = click.prompt("Seleccione #", type=int)
    if 1 <= idx <= len(files):
        return files[idx - 1]
    raise click.ClickException("Número inválido.")


def build_table_name(stem: str) -> str:
    utc = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    return f"{stem[:8]}_{utc}"


def load_df(joblib_path: Path):
    obj = joblib.load(joblib_path)
    return obj["df"] if isinstance(obj, dict) and "df" in obj else obj

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.option("--experiment", "-x", type=click.Path(exists=True, dir_okay=False, path_type=Path),
              help="Archivo .joblib a subir (salta menú).")
@click.option("--exp-dir", default=DEF_EXP_DIR, show_default=True, type=click.Path(path_type=Path),
              help="Directorio donde buscar experimentos si no se usa -x.")
@click.option("--caslib", default=CASLIB_NAME, show_default=True, help="Caslib destino.")
@click.option("--table", help="Nombre de tabla CAS (si se omite se genera).")

def main(experiment: Optional[Path], exp_dir: Path, caslib: str, table: Optional[str]):
    """Sube un experimento .joblib a CAS con httpConnSwat."""

    # 1. Seleccionar experimento
    if experiment is None:
        files = list_experiments(exp_dir)
        if not files:
            raise click.ClickException("No se encontraron experimentos .joblib en el directorio.")
        experiment = pick_experiment_menu(files)

    click.echo(f"📦 Experimento seleccionado: {experiment.name}")
    df = load_df(experiment)
    click.echo(f"   → filas: {len(df):,}, columnas: {len(df.columns)}")

    # 2. Nombre de tabla CAS
    tbl_name = table or build_table_name(experiment.stem)

    # 3. Conexión CAS via helper
    try:
        conn = httpConnSwat()
        _ = conn.serverStatus()
        click.echo("🔗 Conexión CAS establecida.")
    except Exception as e:
        raise click.ClickException(f"Error de conexión a CAS: {e}")

    # 4. Subir DataFrame
    click.echo(f"⬆️  Subiendo a caslib '{caslib}' como '{tbl_name}'…")
    try:
        conn.upload_frame(df, casout={"name": tbl_name, "caslib": caslib, "replace": False})
        conn.table.promote(name=tbl_name, target=tbl_name, caslib=caslib, targetLib=caslib)
        click.echo("✅ Tabla subida y promovida.")
    finally:
        conn.terminate()
        click.echo("🔒 Sesión CAS cerrada.")


if __name__ == "__main__":
    main()
