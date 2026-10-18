import numpy as np
import pytest

from agente_inversor import datos, envs, evaluacion
from agente_inversor.envs import EntornoInversion


def _precios(n=300, semilla=0):
    rng = np.random.default_rng(semilla)
    return 100.0 * np.exp(np.cumsum(rng.normal(0.0005, 0.01, n)))


def test_espacios_y_forma_del_estado():
    env = EntornoInversion(_precios(), 20, 300)
    obs, _ = env.reset(seed=0)
    assert env.action_space.n == 3
    assert obs.shape == (7,) and obs.dtype == np.float32


def test_el_estado_no_usa_precios_futuros():
    """Cambiar los precios posteriores a t no puede cambiar el estado en t."""
    p = _precios()
    q = p.copy()
    q[101:] *= 3.0
    a, b = EntornoInversion(p, 20, 300), EntornoInversion(q, 20, 300)
    a.reset(seed=0)
    b.reset(seed=0)
    for _ in range(80):  # t llega a 100
        oa, *_ = a.step(1)
        ob, *_ = b.step(1)
    assert a.t == 100
    np.testing.assert_array_equal(oa, ob)


def test_comprar_y_mantener_reproduce_el_activo_menos_el_costo():
    p = _precios()
    env = EntornoInversion(p, 20, 300, costo=0.001)
    sim = evaluacion.simular(env, lambda _o: 2)
    assert sim["capital"][-1] == pytest.approx(p[-1] / p[20] * (1 - 0.001))


def test_costo_proporcional_al_capital_que_se_mueve():
    p = np.full(100, 50.0)  # precio plano: lo unico que cambia el capital es el costo
    env = EntornoInversion(p, 20, 100, costo=0.01)
    env.reset()
    env.step(1)  # 0 % -> 50 %
    assert env.capital == pytest.approx(1 - 0.005)
    env.step(1)  # sin cambio: no cuesta
    assert env.capital == pytest.approx(1 - 0.005)
    env.step(0)  # 50 % -> 0 %
    assert env.capital == pytest.approx((1 - 0.005) ** 2)


def test_en_efectivo_el_capital_no_cambia():
    env = EntornoInversion(_precios(), 20, 300)
    m = evaluacion.metricas(evaluacion.simular(env, lambda _o: 0))
    assert m["retorno_pct"] == 0.0 and m["max_caida_pct"] == 0.0 and m["operaciones"] == 0


def test_recompensa_diferencial_vale_cero_al_mantener_todo_invertido():
    env = EntornoInversion(_precios(), 20, 300, recompensa="diferencial")
    env.reset()
    env.step(2)  # paga el costo de entrada
    _, r, *_ = env.step(2)
    assert r == pytest.approx(0.0, abs=1e-9)


def test_recompensa_aversion_duplica_las_perdidas():
    p = np.array([100.0] * 21 + [90.0, 99.0])
    env = EntornoInversion(p, 20, 23, costo=0.0, recompensa="aversion")
    env.reset()
    _, r_baja, *_ = env.step(2)
    _, r_sube, *_ = env.step(2)
    assert r_baja == pytest.approx(2 * 100 * np.log(0.9))
    assert r_sube == pytest.approx(100 * np.log(1.1))


def test_rotacion_penaliza_mover_capital_sin_tocar_el_capital():
    """La version 4 resta la penalizacion solo en la recompensa, no en el capital."""
    p = np.full(30, 50.0)  # precio plano: el diferencial vale menos el costo real
    base = EntornoInversion(p, 20, 30, costo=0.001, recompensa="diferencial")
    rot = EntornoInversion(p, 20, 30, costo=0.001, recompensa="diferencial_rotacion")
    base.reset()
    rot.reset()
    _, r_base, *_ = base.step(2)  # 0 % -> 100 %
    _, r_rot, *_ = rot.step(2)
    assert r_rot == pytest.approx(r_base - 100 * envs.PENALIZACION_ROTACION)
    assert rot.capital == pytest.approx(base.capital)
    _, r_quieto, *_ = rot.step(2)  # sin cambio de exposicion: sin penalizacion
    assert r_quieto == pytest.approx(0.0, abs=1e-9)


def test_episodio_aleatorio_dura_lo_pedido_y_no_sale_del_tramo():
    env = EntornoInversion(_precios(), 20, 200, episodio=50)
    for semilla in range(20):
        env.reset(seed=semilla)
        assert env.t >= 20 and env.t_fin <= 199
        pasos, fin = 0, False
        while not fin:
            _, _, terminado, fin, _ = env.step(0)
            pasos += 1
            assert not terminado
        assert pasos == 50


def test_los_tramos_no_comparten_retornos():
    tramos = datos.particion(2335, 20)
    cobrados = [set(range(t.inicio + 1, t.fin)) for t in tramos.values()]
    assert not (cobrados[0] & cobrados[1]) and not (cobrados[1] & cobrados[2])
    assert cobrados[0] | cobrados[1] | cobrados[2] == set(range(21, 2335))


def test_maxima_caida():
    sim = {"capital": np.array([1.0, 1.2, 0.9, 1.1]), "exposicion": np.array([1.0, 1.0, 1.0])}
    assert evaluacion.metricas(sim)["max_caida_pct"] == pytest.approx(25.0)


def test_prueba_extendida_continua_la_serie_sin_compartir_retornos():
    original = datos.cargar()
    extendida, t = datos.tramo("prueba_extendida", 20)
    # primera decision el ultimo dia original; primer retorno cobrado, el dia siguiente
    assert extendida.index[t.inicio] == original.index[-1]
    assert extendida.index[t.inicio + 1] > original.index[-1]
    # en los dias comunes las dos fuentes dan los mismos retornos
    comunes = original.index.intersection(extendida.index)
    cociente = original[comunes] / extendida[comunes]
    assert cociente.max() / cociente.min() == pytest.approx(1.0, abs=1e-6)


def test_costos_reportados_coinciden_con_lo_que_pierde_el_capital():
    """Con precio plano, la suma de costos reportados es exactamente la perdida de capital."""
    p = np.full(100, 50.0)
    env = EntornoInversion(p, 20, 100, costo=0.01)
    acciones = iter([2, 0, 1, 1, 2] + [2] * 100)
    sim = evaluacion.simular(env, lambda _o: next(acciones))
    m = evaluacion.metricas(sim)
    assert m["costos_pct"] == pytest.approx((1 - sim["capital"][-1]) * 100)
    assert m["costos_monto"] == pytest.approx(m["costos_pct"] / 100 * evaluacion.CAPITAL_INICIAL)
