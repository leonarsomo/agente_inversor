"""Entorno de inversion en un activo, compatible con Gymnasium.

Contenido:
  - EXPOSICIONES     : fraccion del capital invertida por cada accion
  - RECOMPENSAS      : las funciones de recompensa disponibles
  - EntornoInversion : el entorno
  - make()           : construye el entorno de un tramo (entrenamiento, validacion, prueba,
                       prueba_extendida)

Formulacion:
  Estado   : 6 indicadores del precio calculados con los ultimos `ventana` dias,
             mas la exposicion actual del portafolio.
  Acciones : Discrete(3): tener invertido el 0 %, el 50 % o el 100 % del capital.
  Costo    : `costo` por cada unidad de capital que cambia de manos.
  Tiempo   : en el dia t se observa hasta el cierre de t, se decide, y se cobra el
             retorno de t a t+1. Ningun indicador usa precios posteriores a t.
"""
from __future__ import annotations

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from stable_baselines3.common.monitor import Monitor

from agente_inversor import datos

EXPOSICIONES = np.array([0.0, 0.5, 1.0])
ESCALA = 100.0  # las recompensas van en puntos porcentuales diarios

# Penalizacion de la version 4, en las mismas unidades que la recompensa (puntos
# porcentuales) por cada unidad de capital que cambia de manos. Se fijo antes de
# entrenar y no se ajusto con validacion: cinco veces el costo real (0,1 puntos).
PENALIZACION_ROTACION = 0.005


def _log(r_cartera: float, r_activo: float, cambio: float) -> float:
    """Retorno logaritmico de la cartera, ya neto del costo."""
    return r_cartera


def _diferencial(r_cartera: float, r_activo: float, cambio: float) -> float:
    """Exceso sobre comprar y mantener: estar 100 % invertido sin operar vale 0."""
    return r_cartera - r_activo


def _aversion(r_cartera: float, r_activo: float, cambio: float) -> float:
    """Las perdidas pesan el doble que las ganancias."""
    return r_cartera if r_cartera >= 0 else 2.0 * r_cartera


def _diferencial_rotacion(r_cartera: float, r_activo: float, cambio: float) -> float:
    """Version 4: el diferencial menos una penalizacion por mover capital.

    El costo real (0,1 %) ya esta en r_cartera, pero es diminuto frente al ruido de
    un retorno diario (desviacion cercana a 1,5 puntos), asi que casi no deja senal
    en el aprendizaje. La penalizacion solo cambia la recompensa: el capital y todas
    las metricas siguen pagando unicamente el costo real.
    """
    return r_cartera - r_activo - PENALIZACION_ROTACION * cambio


RECOMPENSAS = {
    "log": _log,
    "diferencial": _diferencial,
    "aversion": _aversion,
    "diferencial_rotacion": _diferencial_rotacion,
}


class EntornoInversion(gym.Env):
    """Recorre precios[inicio:fin]. Con `episodio`, cada episodio es una ventana
    aleatoria de esa longitud (entrenamiento); sin el, una pasada completa (evaluacion).
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        precios: np.ndarray,
        inicio: int,
        fin: int,
        *,
        ventana: int = 20,
        costo: float = 0.001,
        recompensa: str = "log",
        episodio: int | None = None,
    ) -> None:
        super().__init__()
        if inicio < ventana:
            raise ValueError("inicio debe dejar `ventana` dias de historia")
        self.precios = np.asarray(precios, dtype=np.float64)
        self.inicio, self.fin = inicio, fin
        self.ventana, self.costo, self.episodio = ventana, costo, episodio
        self._recompensa = RECOMPENSAS[recompensa]
        # retornos[t] = log(p[t] / p[t-1]): conocido al cierre del dia t
        self.retornos = np.diff(np.log(self.precios), prepend=np.log(self.precios[0]))

        self.action_space = spaces.Discrete(len(EXPOSICIONES))
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(7,), dtype=np.float32)

    def _obs(self) -> np.ndarray:
        t, v = self.t, self.ventana
        r = self.retornos[t - v + 1 : t + 1]
        p = self.precios[t - v + 1 : t + 1]
        vol = r.std() + 1e-8
        vol_ventana = vol * np.sqrt(v)  # volatilidad acumulada en la ventana
        return np.array(
            [
                r[-1] / vol,  # ultimo retorno
                r[-5:].sum() / (vol * np.sqrt(5)),  # momento a 5 dias
                r.sum() / vol_ventana,  # momento de la ventana
                (p[-5:].mean() / p.mean() - 1.0) / vol_ventana,  # cruce de medias
                vol * np.sqrt(252),  # volatilidad anualizada
                (p[-1] / p.max() - 1.0) / vol_ventana,  # distancia al maximo
                self.exposicion,
            ],
            dtype=np.float32,
        )

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if self.episodio is None:
            self.t, self.t_fin = self.inicio, self.fin - 1
        else:
            self.t = int(self.np_random.integers(self.inicio, self.fin - self.episodio))
            self.t_fin = self.t + self.episodio
        self.exposicion = 0.0
        self.capital = 1.0
        return self._obs(), {}

    def step(self, action):
        nueva = float(EXPOSICIONES[int(action)])
        cambio = abs(nueva - self.exposicion)
        costo = cambio * self.costo
        costo_pagado = self.capital * costo  # en unidades de capital (inicial = 1)
        self.exposicion = nueva

        self.t += 1
        r_activo = self.retornos[self.t]
        factor = (1.0 - costo) * (1.0 + nueva * (np.exp(r_activo) - 1.0))
        self.capital *= factor

        recompensa = ESCALA * self._recompensa(float(np.log(factor)), float(r_activo), cambio)
        # El final de los datos no es un estado terminal del problema: es un corte.
        truncado = self.t >= self.t_fin
        info = {"capital": self.capital, "exposicion": nueva, "costo": costo_pagado}
        return self._obs(), recompensa, False, truncado, info


def make(
    tramo: str,
    *,
    ventana: int = 20,
    costo: float = 0.001,
    recompensa: str = "log",
    episodio: int | None = None,
) -> Monitor:
    """El entorno de un tramo. `episodio` solo tiene sentido en entrenamiento."""
    serie, t = datos.tramo(tramo, ventana)
    env = EntornoInversion(
        serie.to_numpy(), t.inicio, t.fin, ventana=ventana, costo=costo,
        recompensa=recompensa, episodio=episodio,
    )
    return Monitor(env)
