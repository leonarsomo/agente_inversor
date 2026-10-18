"""Los modelos guardados reproducen exactamente la tabla del tramo de prueba.

Si una actualizacion de dependencias, un cambio en el entorno o en las metricas
alterara el resultado reportado, estas pruebas fallan en la integracion continua.
"""
import json
from pathlib import Path

import numpy as np
import pytest

from agente_inversor import config, envs, evaluacion, registry

RAIZ = Path(__file__).resolve().parents[1]
REPORTADO = json.loads((RAIZ / "resultados" / "prueba.json").read_text())
ETIQUETA = REPORTADO["etiqueta"]  # ppo_diferencial, la configuracion elegida en validacion


@pytest.mark.parametrize("fila", REPORTADO["agente"], ids=lambda f: f"semilla_{f['semilla']}")
def test_modelo_guardado_reproduce_el_tramo_de_prueba(fila, monkeypatch):
    monkeypatch.chdir(RAIZ)  # registry usa rutas relativas a la raiz del repositorio
    algo = ETIQUETA.split("_", 1)[0]
    modelo = registry.load(algo, ETIQUETA, fila["semilla"])
    e = config.ENTORNO
    sim = evaluacion.simular(
        envs.make("prueba", ventana=e["ventana"], costo=e["costo"]),
        evaluacion.politica_modelo(modelo),
    )
    m = evaluacion.metricas(sim)
    assert m["retorno_pct"] == pytest.approx(fila["retorno_pct"], abs=1e-6)
    assert m["max_caida_pct"] == pytest.approx(fila["max_caida_pct"], abs=1e-6)
    assert m["operaciones"] == fila["operaciones"]
    curva = np.array(REPORTADO["curvas"][f"semilla_{fila['semilla']}"])
    np.testing.assert_allclose(sim["capital"], curva, rtol=0, atol=1e-9)
