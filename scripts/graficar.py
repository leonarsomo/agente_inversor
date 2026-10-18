"""Figuras del informe, a partir de los archivos de resultados/.

  uv run python scripts/graficar.py

  - validacion.png : retorno en validacion de cada semilla, por configuracion
  - prueba.png     : curva de capital del agente y de comprar y mantener en prueba
  - prueba_extendida.png : lo mismo en la segunda prueba, 2018-08-30 a 2025-12-31
  - posiciones_prueba.png : precio y exposicion diaria de cada semilla en prueba
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import ticker

from agente_inversor import config, datos, evaluacion

RES = Path("resultados")


def validacion() -> None:
    df = pd.read_csv(RES / "validacion.csv")
    base = evaluacion.baselines("validacion", config.ENTORNO)
    orden = df.groupby("etiqueta")["validacion_retorno_pct"].median().sort_values().index

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for i, etiqueta in enumerate(orden):
        y = df.loc[df["etiqueta"] == etiqueta, "validacion_retorno_pct"]
        ax.scatter([i] * len(y), y, color="tab:blue", alpha=0.7, zorder=3)
        ax.hlines(y.median(), i - 0.25, i + 0.25, color="black", zorder=4)
    estilos = {"comprar_y_mantener": "tab:green", "aleatorio": "tab:red", "efectivo": "gray"}
    for nombre, color in estilos.items():
        ax.axhline(base[nombre]["retorno_pct"], color=color, ls="--", label=nombre)
    ax.set_xticks(range(len(orden)), orden, rotation=25, ha="right")
    ax.set_ylabel("Retorno acumulado en validacion (%)")
    ax.set_title("Cinco semillas por configuracion; la raya negra es la mediana")
    ax.legend()
    fig.tight_layout()
    fig.savefig(RES / "validacion.png", dpi=150)


def prueba() -> None:
    res = json.loads((RES / "prueba.json").read_text())
    serie = datos.cargar()
    t = datos.particion(len(serie), config.ENTORNO["ventana"])["prueba"]
    fechas = serie.index[t.inicio : t.fin]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for nombre, curva in res["curvas"].items():
        if nombre == "comprar_y_mantener":
            ax.plot(fechas, curva, color="tab:green", lw=2, label="comprar y mantener")
        elif nombre.startswith("semilla_"):  # las exposicion_semilla_* van en posiciones()
            ax.plot(fechas, curva, color="tab:blue", alpha=0.6, label=None)
    ax.plot([], [], color="tab:blue", label=f"agente {res['etiqueta']} (5 semillas)")
    ax.axhline(1.0, color="gray", lw=0.8)
    ax.set_ylabel("Capital (inicial = 1)")
    ax.set_title("Tramo de prueba")
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(RES / "prueba.png", dpi=150)


def prueba_extendida() -> None:
    res = json.loads((RES / "prueba_extendida.json").read_text())
    serie, t = datos.tramo("prueba_extendida", config.ENTORNO["ventana"])
    fechas = serie.index[t.inicio : t.fin]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for nombre, curva in res["curvas"].items():
        if nombre == "comprar_y_mantener":
            ax.plot(fechas, curva, color="tab:green", lw=2, label="comprar y mantener")
        elif nombre.startswith("semilla_"):
            ax.plot(fechas, curva, color="tab:blue", alpha=0.6, label=None)
    ax.plot([], [], color="tab:blue", label=f"agente {res['etiqueta']} (5 semillas)")
    for a, b in (("2020-02-19", "2020-03-23"), ("2021-11-18", "2022-11-03")):
        ax.axvspan(pd.Timestamp(a), pd.Timestamp(b), color="gray", alpha=0.15, lw=0)
    ax.set_yscale("log")
    ax.yaxis.set_major_locator(ticker.FixedLocator([0.6, 0.8, 1, 1.5, 2, 3, 4, 6]))
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{v:g}"))
    ax.yaxis.set_minor_locator(ticker.NullLocator())
    ax.axhline(1.0, color="gray", lw=0.8)
    ax.set_ylabel("Capital (inicial = 1, escala logaritmica)")
    ax.set_title("Segunda prueba, 2018 a 2025; en gris, las caidas de 2020 y 2022")
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(RES / "prueba_extendida.png", dpi=150)


def posiciones() -> None:
    res = json.loads((RES / "prueba.json").read_text())
    serie = datos.cargar()
    t = datos.particion(len(serie), config.ENTORNO["ventana"])["prueba"]
    fechas = serie.index[t.inicio + 1 : t.fin]  # la exposicion i se cobra el dia i + 1
    semillas = sorted(k for k in res["curvas"] if k.startswith("exposicion_semilla_"))

    fig, ejes = plt.subplots(
        len(semillas) + 1, 1, figsize=(9, 8), sharex=True,
        gridspec_kw={"height_ratios": [1.8] + [1] * len(semillas)},
    )
    ejes[0].plot(fechas, serie.iloc[t.inicio + 1 : t.fin], color="black", lw=1)
    ejes[0].set_ylabel("Precio GOOGL")
    ejes[0].set_title("Tramo de prueba: exposicion diaria del agente en cada semilla")
    for ax, clave in zip(ejes[1:], semillas, strict=True):
        x = res["curvas"][clave]
        ax.fill_between(fechas, x, step="post", color="tab:blue", alpha=0.5)
        ax.set_ylim(-0.05, 1.05)
        ax.set_yticks([0, 0.5, 1])
        ax.set_ylabel(f"s{clave.rsplit('_', 1)[1]}", rotation=0, labelpad=12)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(RES / "posiciones_prueba.png", dpi=150)


if __name__ == "__main__":
    validacion()
    if (RES / "prueba.json").exists():
        prueba()
        posiciones()
    if (RES / "prueba_extendida.json").exists():
        prueba_extendida()
