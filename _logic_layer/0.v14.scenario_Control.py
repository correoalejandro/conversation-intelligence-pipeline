#!/usr/bin/env python
"""
run_showcase.py ─ Lanza varios lotes sintéticos con parámetros fijos
y deja todo registrado (prompt ▸ batch ▸ conversaciones).

Lotes predefinidos
──────────────────
A) 300 clientes – estrategia VIEJA   – con semillas
B) 300 clientes – estrategia NUEVA   – con semillas
C) 100 clientes – estrategia NUEVA   – SIN semillas (stress-test)

Se guardan en una carpeta timestamp dentro de showcase_runs/.
"""

import subprocess, pathlib, datetime, sys

# Ruta al script generador (ajusta si lo moviste)
GENERATOR = "synthetic_cobranza_generator.py"

JOBS = [
    # (nombre_carpeta, n_clients, estrategia, use_seeds)
    ("demo_old",         300, "vieja", 1),
    ("demo_new",         300, "nueva", 1),
    ("demo_new_noSeeds", 100, "nueva", 0),
]

def main():
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    root = pathlib.Path("showcase_runs") / ts
    root.mkdir(parents=True, exist_ok=True)

    for folder, n, estr, seeds in JOBS:
        out_dir = root / folder
        print(f"▶ Generando {folder}  ({n} clientes, estr. {estr}, seeds={seeds})")
        cmd = [
            sys.executable, GENERATOR,
            "--n_clients", str(n),
            "--out", str(out_dir)
        ]
        # Pasamos seeds mediante variable de entorno (override rápido)
        env = dict(**os.environ, USE_SEEDS=str(seeds))
        subprocess.run(cmd, check=True, env=env)

    print(f"\n✅ Showcase completo en {root}\n"
          "Todos los lotes quedaron registrados en registries/.")

if __name__ == "__main__":
    import os
    main()
