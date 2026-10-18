"""Comparacion de las tres recompensas en el tramo de validacion, sin reentrenar.

  uv run python scripts/analisis_recompensas.py

Carga los modelos de saves/ (2 algoritmos x 3 recompensas x 5 semillas, mas PPO con la
version 4 de la recompensa), los recorre en validacion y produce:

  - validacion_costos.csv       : metricas por corrida, ahora con costos de transaccion
  - validacion_costos_resumen.csv : mediana por configuracion
  - recompensas_posiciones.png  : exposicion diaria de PPO (semilla 0) con cada recompensa

Solo usa el tramo de validacion: no toca los tramos de prueba.
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from agente_inversor import config, datos, envs, evaluacion, registry

RES = Path("resultados")
ALGOS = ("dqn", "ppo")
RECOMPENSAS = tuple(envs.RECOMPENSAS)
SEMILLAS = range(5)


def _env():
    e = config.ENTORNO
    return envs.make("validacion", ventana=e["ventana"], costo=e["costo"])


def tabla() -> dict:
    filas, exposiciones = [], {}
    for algo in ALGOS:
        for rec in RECOMPENSAS:
            etiqueta = f"{algo}_{rec}"
            if not registry.save_path(etiqueta, 0).exists():  # la version 4 solo con PPO
                continue
            for s in SEMILLAS:
                modelo = registry.load(algo, etiqueta, s)
                sim = evaluacion.simular(_env(), evaluacion.politica_modelo(modelo))
                filas.append({"etiqueta": etiqueta, "semilla": s, **evaluacion.metricas(sim)})
                exposiciones[(etiqueta, s)] = sim["exposicion"]
    df = pd.DataFrame(filas)
    df.to_csv(RES / "validacion_costos.csv", index=False)
    resumen = df.groupby("etiqueta").median(numeric_only=True).drop(columns="semilla")
    base = evaluacion.baselines("validacion", config.ENTORNO)
    for nombre in ("comprar_y_mantener", "aleatorio"):
        resumen.loc[f"({nombre})"] = pd.Series(base[nombre])
    resumen = resumen.sort_values("retorno_pct", ascending=False)
    resumen.to_csv(RES / "validacion_costos_resumen.csv")
    cols = ["retorno_pct", "max_caida_pct", "costos_pct", "exposicion_media", "operaciones"]
    print(resumen[cols].round(2).to_string())
    return exposiciones


def figura(exposiciones: dict) -> None:
    serie, t = datos.tramo("validacion", config.ENTORNO["ventana"])
    fechas = serie.index[t.inicio + 1 : t.fin]
    recompensas = [r for r in RECOMPENSAS if ("ppo_" + r, 0) in exposiciones]
    fig, ejes = plt.subplots(len(recompensas) + 1, 1, figsize=(9, 2 + 1.4 * len(recompensas)),
                             sharex=True,
                             gridspec_kw={"height_ratios": [1.6] + [1] * len(recompensas)})
    ejes[0].plot(fechas, serie.iloc[t.inicio + 1 : t.fin], color="black", lw=1)
    ejes[0].set_ylabel("Precio GOOGL")
    ejes[0].set_title("Validacion: exposicion diaria de PPO (semilla 0) segun la recompensa")
    for ax, rec in zip(ejes[1:], recompensas, strict=True):
        x = exposiciones[(f"ppo_{rec}", 0)]
        ax.fill_between(fechas, x, step="post", color="tab:blue", alpha=0.5)
        ax.set_ylim(-0.05, 1.05)
        ax.set_yticks([0, 0.5, 1])
        ax.set_ylabel(rec.replace("_", "\n"))
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(RES / "recompensas_posiciones.png", dpi=150)


if __name__ == "__main__":
    figura(tabla())
