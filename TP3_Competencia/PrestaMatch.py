"""PrestaMatch: árbol de decisión y selección de cartera orientada a rentabilidad.
Entradas: train_clean.csv y test_clean.csv en DATA_DIR (por defecto data/).
Salidas: solicitudes_test.csv y resultados.json en OUT_DIR (por defecto salida/).
"""
import json, os
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier

SEED = 42
PRESUPUESTO = 2_000_000
PESOS_REGLA = {"ingreso": 1, "interes": 1, "atraso": 1}
DATA_DIR = Path(os.environ.get("DATA_DIR", "data"))
OUT_DIR = Path(os.environ.get("OUT_DIR", "salida"))


def cargar_datos():
    train = pd.read_csv(DATA_DIR / "train_clean.csv")
    test = pd.read_csv(DATA_DIR / "test_clean.csv")
    if "Estado_Final" not in train or "Estado_Final" in test:
        raise ValueError("train debe tener Estado_Final y test no debe tenerlo.")
    for nombre, df in [("train", train), ("test", test)]:
        if "ID_Solicitud" not in df or df.ID_Solicitud.duplicated().any():
            raise ValueError(f"{nombre}: ID_Solicitud falta o tiene duplicados.")
    return train, test


def crear_modelo(X):
    categoricas = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

    numericas = [c for c in X.columns if c not in categoricas]

    prep = ColumnTransformer([("num", SimpleImputer(strategy="median"), numericas), ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),("onehot", OneHotEncoder(handle_unknown="ignore")) ]), categoricas), ], remainder="drop")

    return Pipeline([
        ("preprocesamiento", prep),
        ("arbol", RandomForestClassifier(
            n_estimators=200,
            max_depth=16,
            min_samples_leaf=60,
            max_features="sqrt",
            class_weight=None,
            random_state=SEED,
            n_jobs=-1
        ))
    ])


def preparar_xy(train, test):
    excluir = ["ID_Solicitud", "ID_Usuario", "Estado_Final"]
    X = train.drop(columns=excluir, errors="ignore")
    y = train["Estado_Final"].astype(int)
    Xt = test.drop(columns=excluir, errors="ignore")
    if set(X.columns) != set(Xt.columns):
        raise ValueError(f"Columnas distintas train/test: faltan {set(X)-set(Xt)}, sobran {set(Xt)-set(X)}")
    return X, y, Xt[X.columns]


def ganancia_esperada(prob_pago, monto, tasa):
    # Si paga: recupera capital + interés; si incumple: se pierde el capital.
    return monto * (prob_pago * (1.0 + tasa) - 1.0)


def calcular_puntaje_reglas(df):
    """Puntaje de reglas 0..1. Cada peso vale 1; menos atraso obtiene mayor puntaje."""
    cols = ["Ingreso_Mensual", "interes_potencial", "atraso_medio"]
    faltan = [c for c in cols if c not in df]
    if faltan:
        raise ValueError(f"Faltan variables para las reglas ponderadas: {faltan}")
    ingreso = pd.to_numeric(df[cols[0]], errors="coerce")
    interes = pd.to_numeric(df[cols[1]], errors="coerce")
    atraso = pd.to_numeric(df[cols[2]], errors="coerce")
    pct = {
        "ingreso": ingreso.rank(pct=True).fillna(0).to_numpy(),
        "interes": interes.rank(pct=True).fillna(0).to_numpy(),
        "atraso": (1 - atraso.rank(pct=True)).fillna(0).to_numpy(),
    }
    total = sum(PESOS_REGLA.values())
    return sum(PESOS_REGLA[k] * pct[k] for k in PESOS_REGLA) / total


def seleccionar_cartera(ganancia, monto, presupuesto, puntaje_reglas):
    """Maximiza ganancia esperada total; el puntaje ponderado desempata marginalmente.

    Solo considera solicitudes con ganancia esperada positiva. Si SciPy/MILP no está
    disponible, usa una alternativa voraz por ganancia esperada por dólar.
    """
    g = np.asarray(ganancia, float)
    m = np.asarray(monto, float)
    score = np.asarray(puntaje_reglas, float)
    aprobado = np.zeros(len(m), dtype=int)
    elegibles = np.where((m > 0) & np.isfinite(g) & (g > 0) & np.isfinite(score))[0]
    if not len(elegibles) or presupuesto <= 0:
        return aprobado
    try:
        from scipy.optimize import Bounds, LinearConstraint, milp
        # La contribución de las reglas es diminuta frente a los dólares de ganancia:
        # respeta los pesos como desempate sin reemplazar el objetivo financiero.
        max_score = max(float(np.max(score[elegibles])), 1.0)
        objetivo = g[elegibles] + 0.01 * score[elegibles] / max_score
        r = milp(c=-objetivo, integrality=np.ones(len(elegibles)), bounds=Bounds(np.zeros(len(elegibles)), np.ones(len(elegibles))),constraints=LinearConstraint(m[elegibles].reshape(1, -1), -np.inf, presupuesto), options={"time_limit": 60})
        if r.x is not None and r.success:
            aprobado[elegibles] = (r.x >= .5).astype(int)
            return aprobado
    except Exception:
        pass
    orden = elegibles[np.argsort(-(g[elegibles] / m[elegibles] + 1e-9 * score[elegibles]))]
    gastado = 0.0
    for i in orden:
        if gastado + m[i] <= presupuesto:
            aprobado[i] = 1
            gastado += m[i]
    return aprobado


def ganancia_real(y, monto, tasa, aprobado):
    return float((np.where(np.asarray(y) == 1, np.asarray(monto)*np.asarray(tasa), -np.asarray(monto)) * np.asarray(aprobado)).sum())


def evaluar(modelo, X, y, df, presupuesto):
    clases = list(modelo.named_steps["arbol"].classes_)
    prob = modelo.predict_proba(X)[:, clases.index(1)]
    monto = df.Monto_Solicitado.to_numpy(float)
    tasa = df.Tasa_Interes.to_numpy(float)
    ge = ganancia_esperada(prob, monto, tasa)
    score = calcular_puntaje_reglas(df)
    aprobado = seleccionar_cartera(ge, monto, presupuesto, score)
    gastado = float((monto * aprobado).sum())
    real = ganancia_real(y, monto, tasa, aprobado)
    return {"ganancia_neta": real, "roi_sobre_capital_prestado": real/gastado if gastado else 0.0,
            "aprobadas": int(aprobado.sum()), "monto_aprobado": gastado,
            "ganancia_esperada": float((ge * aprobado).sum())}


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    train, test = cargar_datos()
    X, y, Xt = preparar_xy(train, test)
    if set(y.unique()) != {0, 1}:
        raise ValueError("Estado_Final debe usar 0=default y 1=pagado.")
    for c in ["Monto_Solicitado", "Tasa_Interes"]:
        if c not in train or c not in test: raise ValueError(f"Falta {c}.")

    i_fit, i_val = train_test_split(np.arange(len(train)), test_size=.30, stratify=y, random_state=SEED)
    mv = crear_modelo(X); mv.fit(X.iloc[i_fit], y.iloc[i_fit])
    budget_val = min(PRESUPUESTO, float(train.iloc[i_val].Monto_Solicitado.sum()))
    holdout = evaluar(mv, X.iloc[i_val], y.iloc[i_val], train.iloc[i_val], budget_val)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    folds = []
    for a, b in skf.split(X, y):
        mf = crear_modelo(X); mf.fit(X.iloc[a], y.iloc[a])
        budget = min(PRESUPUESTO, float(train.iloc[b].Monto_Solicitado.sum()))
        folds.append(evaluar(mf, X.iloc[b], y.iloc[b], train.iloc[b], budget))
    cv = {"ganancia_neta_media": float(np.mean([f["ganancia_neta"] for f in folds])), 
          "roi_medio": float(np.mean([f["roi_sobre_capital_prestado"] for f in folds])),
          "detalle_folds": folds}

    final = crear_modelo(X); final.fit(X, y)
    cls = list(final.named_steps["arbol"].classes_)
    prob = final.predict_proba(Xt)[:, cls.index(1)]
    monto = test.Monto_Solicitado.to_numpy(float); tasa = test.Tasa_Interes.to_numpy(float)
    ge = ganancia_esperada(prob, monto, tasa)
    score = calcular_puntaje_reglas(test)
    aprobado = seleccionar_cartera(ge, monto, PRESUPUESTO, score)
    salida = pd.DataFrame({"ID_Solicitud": test.ID_Solicitud.to_numpy(), "Decision_Aprobacion": aprobado})
    salida.to_csv(OUT_DIR / "solicitudes_test.csv", index=False)
    gastado = float((monto*aprobado).sum())
    resultados = {"modelo":"RandomForestClassifier", "parametros":{"n_estimators": 200,"max_depth":16,"min_samples_leaf":60,"max_features": "sqrt","class_weight":None,"random_state": SEED},
      "n_train":len(train),"n_test":len(test),"pct_pagaron_train":float(y.mean()),"presupuesto_usd":PRESUPUESTO,
      "pesos_regla":PESOS_REGLA,
      "criterio_seleccion":"maximizar ganancia esperada total; reglas ponderadas 1:1:1 como desempate marginal",
      "holdout":holdout,"cv_5_folds":cv,
      "test":{"aprobadas":int(aprobado.sum()),"monto_aprobado":gastado,"presupuesto_restante":PRESUPUESTO-gastado, 
              "ganancia_esperada_total":float((ge*aprobado).sum())}}
    
    with open(OUT_DIR/"resultados.json","w",encoding="utf-8") as f: json.dump(resultados,f,indent=2,ensure_ascii=False)
    assert gastado <= PRESUPUESTO + 1e-6
    print(f"Holdout ganancia neta: ${holdout['ganancia_neta']:,.2f}; ROI {holdout['roi_sobre_capital_prestado']:.2%}")
    print(f"CV ganancia neta media: ${cv['ganancia_neta_media']:,.2f}; ROI medio {cv['roi_medio']:.2%}")
    print(f"Test: {aprobado.sum()} aprobadas, capital ${gastado:,.2f}, ganancia esperada ${resultados['test']['ganancia_esperada_total']:,.2f}")

if __name__ == '__main__': main()
