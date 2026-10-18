"""Dos comprobaciones de rigor sobre resultados ya guardados, sin reentrenar ni reajustar.

  uv run python scripts/analisis_rigor.py

Solo lee resultados/prueba.json y resultados/prueba_extendida.json, que ya fueron
reportados; no elige nada ni cambia ningun modelo. Produce:

  - rigor_rebalanceo.csv   : retorno con la contabilidad del entorno frente a una
                             contabilidad exacta que cobra el rebalanceo al 50 %
  - rigor_significancia.csv: intervalo de confianza del exceso de cada semilla sobre
                             comprar y mantener (bootstrap por bloques)

Rebalanceo. Cuando la exposicion es 50 %, el entorno aplica cada dia el retorno a la
mitad del capital: equivale a volver a dejar exactamente la mitad invertida cada
noche sin pagar por ello. En la contabilidad exacta la parte invertida deriva con el
precio y llevarla de nuevo a la exposicion elegida cuesta lo mismo que cualquier otra
operacion (0,1 % del capital movido).

Significancia. El exceso diario es la diferencia de retornos logaritmicos entre la
semilla y comprar y mantener. Se remuestrea por bloques de 20 dias para conservar la
agrupacion de la volatilidad y se reporta el intervalo del 95 % del exceso acumulado.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from agente_inversor import config, datos

RES = Path("resultados")
TRAMOS = ("prueba", "prueba_extendida")
COSTO = config.ENTORNO["costo"]
BLOQUE, REMUESTREOS = 20, 5000


def _precios(tramo: str) -> np.ndarray:
    serie, t = datos.tramo(tramo, config.ENTORNO["ventana"])
    return serie.to_numpy()[t.inicio : t.fin]


def capital_exacto(precios: np.ndarray, exposiciones: np.ndarray) -> float:
    """Capital final cobrando todo lo que se mueve, incluida la deriva del 50 %."""
    capital, invertido = 1.0, 0.0
    for e, r in zip(exposiciones, precios[1:] / precios[:-1], strict=True):
        costo = abs(e - invertido / capital) * COSTO
        capital *= 1.0 - costo
        invertido = e * capital
        capital += invertido * (r - 1.0)
        invertido *= r
    return capital


def rebalanceo() -> pd.DataFrame:
    filas = []
    for tramo in TRAMOS:
        res = json.loads((RES / f"{tramo}.json").read_text())
        p = _precios(tramo)
        estrategias = {
            f"semilla {k}": np.array(res["curvas"][f"exposicion_semilla_{k}"]) for k in range(5)
        }
        estrategias["50 % fijo"] = np.full(len(p) - 1, 0.5)
        for nombre, x in estrategias.items():
            entorno = (
                res["curvas"][f"semilla_{nombre[-1]}"][-1]
                if nombre.startswith("semilla")
                else 1.0 + res["baselines"]["mitad_fija"]["retorno_pct"] / 100
            )
            exacto = capital_exacto(p, x)
            filas.append({
                "tramo": tramo, "estrategia": nombre,
                "dias_al_50": int((x == 0.5).sum()),
                "retorno_entorno_pct": (entorno - 1) * 100,
                "retorno_exacto_pct": (exacto - 1) * 100,
                "diferencia_puntos": (exacto - entorno) * 100,
            })
    return pd.DataFrame(filas)


def significancia() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    filas = []
    for tramo in TRAMOS:
        res = json.loads((RES / f"{tramo}.json").read_text())
        bh = np.diff(np.log(res["curvas"]["comprar_y_mantener"]))
        for k in range(5):
            exceso = np.diff(np.log(res["curvas"][f"semilla_{k}"])) - bh
            n = len(exceso)
            inicios = rng.integers(0, n - BLOQUE + 1, size=(REMUESTREOS, -(-n // BLOQUE)))
            idx = (inicios[:, :, None] + np.arange(BLOQUE)).reshape(REMUESTREOS, -1)[:, :n]
            sumas = exceso[idx].sum(axis=1)
            bajo, alto = np.percentile(sumas, [2.5, 97.5])
            filas.append({
                "tramo": tramo, "semilla": k,
                "exceso_log_puntos": exceso.sum() * 100,
                "ic95_bajo": bajo * 100, "ic95_alto": alto * 100,
                "excluye_cero": bool(bajo > 0 or alto < 0),
            })
    return pd.DataFrame(filas)


if __name__ == "__main__":
    reb, sig = rebalanceo(), significancia()
    reb.to_csv(RES / "rigor_rebalanceo.csv", index=False)
    sig.to_csv(RES / "rigor_significancia.csv", index=False)
    print(reb.round(2).to_string(index=False), "\n")
    print(sig.round(2).to_string(index=False))
