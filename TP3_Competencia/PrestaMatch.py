"""
PrestaMatch - Optimización de cartera P2P
=========================================
Estrategia: regla de puntaje. Se aprueba primero a quienes tengan
MÁS ingreso, MÁS interés potencial y MENOS días de atraso, hasta agotar el presupuesto.

Pipeline de principio a fin:
  1) Carga de datos y cruce solicitudes <-> comportamiento bancario (sin leakage temporal)
  2) Variables derivadas -> train_clean.csv / test_clean.csv
  3) Evaluación: baseline, hold-out estratificado y K-Fold estratificado
  4) Decisión sobre test dentro del presupuesto -> salida/submission.csv

Uso:
    DATA_DIR=data OUT_DIR=salida python solucion.py
    (DATA_DIR = carpeta con los 3 CSV; por defecto "data". OUT_DIR por defecto "salida")
"""
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split

SEED = 42
np.random.seed(SEED)

DATA_DIR = Path(os.environ.get("DATA_DIR", "data"))
OUT_DIR = Path(os.environ.get("OUT_DIR", "salida"))
PRESUPUESTO = 2_000_000
VENTANA = 12          # meses previos a la solicitud que se usan del banco

# Peso de cada criterio en el puntaje (mayor peso = pesa más en el orden)
PESOS_REGLA = {"ingreso": 1, "interes": 1, "atraso": 1}


# ======================================================================
# 1) CARGA Y VARIABLES
# ======================================================================
def cargar_datos():
    tr = pd.read_csv(DATA_DIR / "solicitudes_train.csv")
    te = pd.read_csv(DATA_DIR / "solicitudes_test.csv")
    cb = pd.read_csv(DATA_DIR / "comportamiento_bancario.csv")
    return tr, te, cb


def _mes_idx(s):
    """'2025-04' -> número entero de mes (para restar fechas)."""
    p = s.str.split("-", expand=True).astype(int)
    return p[0] * 12 + p[1]


def _agregar_banco(sol, cb):
    """Resume el comportamiento bancario de cada solicitud usando SOLO los
    VENTANA meses anteriores a la fecha de solicitud (evita leakage)."""
    m = sol[["ID_Solicitud", "ID_Usuario", "sol_idx"]].merge(cb, on="ID_Usuario")
    m["meses_antes"] = m["sol_idx"] - m["mes_idx"]       # 1 = mes inmediato anterior
    m = m[(m["meses_antes"] >= 1) & (m["meses_antes"] <= VENTANA)]
    g = m.groupby("ID_Solicitud")
    return pd.DataFrame({
        "n_meses_hist": g.size(),
        "saldo_medio": g["Saldo_Promedio_Mensual"].mean(),
        "atraso_medio": g["Dias_Atraso_Otras_Deudas"].mean(),
    }).reset_index()


def construir_features(sol, cb):
    sol = sol.copy()
    sol["sol_idx"] = _mes_idx(sol["Fecha_Solicitud"])
    d = sol.merge(_agregar_banco(sol, cb), on="ID_Solicitud", how="left")
    d["monto_sobre_ingreso"] = d["Monto_Solicitado"] / d["Ingreso_Mensual"]       # cuántos sueldos pide
    d["puede_pagar"] = (d["saldo_medio"] >= d["Monto_Solicitado"]).astype(int)    # 1 = su saldo cubre el monto
    d["interes_potencial"] = d["Monto_Solicitado"] * d["Tasa_Interes"]            # lo que gana la plataforma si paga
    return d.drop(columns=["Motivo_Prestamo", "Fecha_Solicitud", "sol_idx"])


def preparar(tr, te, cb):
    """Devuelve (tr_clean, te_clean, lista_de_features)."""
    tr_c, te_c = construir_features(tr, cb), construir_features(te, cb)
    feats = [c for c in te_c.columns if c not in ("ID_Solicitud", "ID_Usuario")]
    med = tr_c[feats].median()                    # medianas SOLO de train
    tr_c[feats] = tr_c[feats].fillna(med)
    te_c[feats] = te_c[feats].fillna(med)
    return tr_c, te_c, feats


# ======================================================================
# 2) REGLA DE DECISIÓN Y MÉTRICA DE NEGOCIO
# ======================================================================
def puntaje_regla(d, pesos=None):
    """Puntaje 0-1 por solicitud: promedio ponderado de tres percentiles (dentro de d):
       - Ingreso_Mensual   (más ingreso  -> mejor)
       - interes_potencial (más interés  -> mejor)
       - atraso_medio      (menos atraso -> mejor, por eso se invierte el signo)
    Los percentiles ponen las tres variables en la misma escala (miles de USD vs días)."""
    pesos = pesos or PESOS_REGLA
    pct = lambda v: pd.Series(v).rank(pct=True).values
    total = sum(pesos.values())
    return (pesos["ingreso"] * pct(d["Ingreso_Mensual"].values)
            + pesos["interes"] * pct(d["interes_potencial"].values)
            + pesos["atraso"] * pct(-d["atraso_medio"].values)) / total


def aprobar_por_puntaje(puntaje, monto, presupuesto):
    """Aprueba en orden de mayor puntaje mientras el acumulado entre en el presupuesto."""
    aprobado = np.zeros(len(puntaje), dtype=int)
    gastado = 0
    for i in np.argsort(-puntaje, kind="stable"):
        if gastado + monto[i] <= presupuesto:
            aprobado[i] = 1
            gastado += monto[i]
    return aprobado


def roi_por_puntaje(d, puntaje, frac_presup):
    """ROI real = ganancia neta / presupuesto simulado.
    Paga -> Monto x Tasa ; default -> -Monto ; rechazado -> 0."""
    monto, tasa = d["Monto_Solicitado"].values, d["Tasa_Interes"].values
    presup = frac_presup * monto.sum()
    ap = aprobar_por_puntaje(puntaje, monto, presup)
    g = np.where(d["Estado_Final"].values == 1, monto * tasa, -monto)
    return (g * ap).sum() / presup


def cv_roi(tr_c, y, frac_presup, fn_puntaje):
    """ROI promedio y desvío con K-Fold estratificado (5 folds).
    fn_puntaje(d) devuelve el puntaje de cada fila de d."""
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    r = [roi_por_puntaje(tr_c.iloc[b], fn_puntaje(tr_c.iloc[b]), frac_presup)
         for _, b in skf.split(tr_c, y)]
    return float(np.mean(r)), float(np.std(r))


def puntaje_baseline_tasa(d):
    """Baseline sin criterio de riesgo: prestar primero a quien paga más tasa."""
    return d["Tasa_Interes"].values


# ======================================================================
# 3) PROGRAMA PRINCIPAL
# ======================================================================
def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tr, te, cb = cargar_datos()
    cb["mes_idx"] = _mes_idx(cb["Mes_Referencia"])
    tr_c, te_c, FEATS = preparar(tr, te, cb)
    tr_c.to_csv(OUT_DIR / "train_clean.csv", index=False)
    te_c.to_csv(OUT_DIR / "test_clean.csv", index=False)
    y = tr_c["Estado_Final"]
    frac = PRESUPUESTO / te_c["Monto_Solicitado"].sum()   # presupuesto relativo en test (~29%)
    print(f"train {tr_c.shape} | test {te_c.shape} | {len(FEATS)} variables")

    res = {"n_train": len(tr_c), "n_test": len(te_c), "n_features": len(FEATS),
           "pct_pagan": float(y.mean()), "frac_presupuesto": float(frac),
           "pesos": PESOS_REGLA,
           "pct_puede_pagar_train": float(tr_c["puede_pagar"].mean()),
           "pct_puede_pagar_test": float(te_c["puede_pagar"].mean())}

    # ---- Hold-out estratificado 70/30 ----
    _, idx_va = train_test_split(tr_c.index, test_size=0.30, stratify=y, random_state=SEED)
    v = tr_c.loc[idx_va]
    res["holdout"] = {
        "baseline_tasa": float(roi_por_puntaje(v, puntaje_baseline_tasa(v), frac)),
        "regla": float(roi_por_puntaje(v, puntaje_regla(v), frac)),
    }
    print("\n=== Hold-out (30% de validación) ===")
    print(f"Baseline (prestar por tasa) ROI: {res['holdout']['baseline_tasa']:+.2%}")
    print(f"Regla ingreso/interés/atraso ROI: {res['holdout']['regla']:+.2%}")

    # ---- K-Fold estratificado (5 folds) ----
    mu_b, sd_b = cv_roi(tr_c, y, frac, puntaje_baseline_tasa)
    mu_r, sd_r = cv_roi(tr_c, y, frac, puntaje_regla)
    res["cv"] = {"baseline_tasa": {"roi": mu_b, "std": sd_b}, "regla": {"roi": mu_r, "std": sd_r}}
    print("\n=== K-Fold estratificado (5 folds) ===")
    print(f"Baseline (prestar por tasa) ROI: {mu_b:+.2%} (±{sd_b:.2%})")
    print(f"Regla {PESOS_REGLA}  ROI: {mu_r:+.2%} (±{sd_r:.2%})")

    # ---- Decisión sobre test ----
    aprobado = aprobar_por_puntaje(puntaje_regla(te_c), te_c["Monto_Solicitado"].values, PRESUPUESTO)
    sub = pd.DataFrame({"ID_Solicitud": te_c["ID_Solicitud"], "Aprobado": aprobado})
    sub.to_csv(OUT_DIR / "submission.csv", index=False)
    gastado = int(te_c.loc[aprobado == 1, "Monto_Solicitado"].sum())
    res.update({"aprobadas": int(aprobado.sum()), "monto_aprobado": gastado})
    print(f"\nAprobadas: {aprobado.sum()} de {len(te_c)} | Monto: {gastado:,} (límite {PRESUPUESTO:,})")
    assert gastado <= PRESUPUESTO, "¡Te pasaste del presupuesto!"
    assert set(sub["Aprobado"].unique()) <= {0, 1} and sub["ID_Solicitud"].is_unique
    assert len(sub) == len(te)

    with open(OUT_DIR / "resultados.json", "w") as f:
        json.dump(res, f, indent=2)


if __name__ == "__main__":
    main()
