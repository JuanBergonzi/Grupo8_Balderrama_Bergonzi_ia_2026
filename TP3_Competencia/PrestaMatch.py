"""
PrestaMatch - Árbol de Decisión y optimización de cartera.

Entradas (en DATA_DIR, por defecto ./data):
    train_clean.csv  -> datos de entrenamiento, debe incluir Estado_Final
    test_clean.csv   -> solicitudes nuevas, sin Estado_Final

Salida (en OUT_DIR, por defecto ./salida):
    solicitudes_test.csv -> archivo final con exactamente:
                            ID_Solicitud, Decision_Aprobacion
    resultados.json       -> métricas de evaluación y resumen de decisiones

Ejecución:
    pip install pandas numpy scikit-learn scipy
    python solucion.py

Si los CSV están en otra carpeta:
    DATA_DIR=. OUT_DIR=salida python solucion.py
"""
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier

SEED = 42
PRESUPUESTO = 2_000_000
# Pesos explícitos de la consigna: las tres reglas tienen igual importancia.
PESOS_REGLA = {"ingreso": 1, "interes": 1, "atraso": 1}
DATA_DIR = Path(os.environ.get("DATA_DIR", "data"))
OUT_DIR = Path(os.environ.get("OUT_DIR", "salida"))


def cargar_datos():
    train_path = DATA_DIR / "train_clean.csv"
    test_path = DATA_DIR / "test_clean.csv"
    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError(
            f"No encuentro {train_path} y/o {test_path}. "
            "Poné train_clean.csv y test_clean.csv en la carpeta indicada por DATA_DIR."
        )

    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)

    if "Estado_Final" not in train.columns:
        raise ValueError("train_clean.csv debe contener la columna Estado_Final.")
    if "Estado_Final" in test.columns:
        raise ValueError("test_clean.csv no debe contener Estado_Final.")
    for nombre, df in [("train_clean.csv", train), ("test_clean.csv", test)]:
        if "ID_Solicitud" not in df.columns:
            raise ValueError(f"{nombre} debe contener ID_Solicitud.")
        if df["ID_Solicitud"].duplicated().any():
            raise ValueError(f"{nombre} contiene ID_Solicitud duplicados.")

    return train, test


def crear_modelo(X):
    categoricas = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    numericas = [c for c in X.columns if c not in categoricas]

    preprocesador = ColumnTransformer(
        transformers=[
            ("num", SimpleImputer(strategy="median"), numericas),
            ("cat", Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore")),
            ]), categoricas),
        ],
        remainder="drop",
    )
    return Pipeline(steps=[
        ("preprocesamiento", preprocesador),
        ("arbol", DecisionTreeClassifier(
            max_depth=6,
            min_samples_leaf=20,
            class_weight="balanced",
            random_state=SEED,
        )),
    ])


def preparar_xy(train, test):
    columnas_no_predictoras = ["ID_Solicitud", "ID_Usuario", "Estado_Final"]
    X_train = train.drop(columns=columnas_no_predictoras, errors="ignore")
    y = train["Estado_Final"].astype(int)
    X_test = test.drop(
        columns=["ID_Solicitud", "ID_Usuario", "Estado_Final"], errors="ignore"
    )

    # Mantener exactamente las columnas y el orden que aprendió el entrenamiento.
    faltantes = [c for c in X_train.columns if c not in X_test.columns]
    extras = [c for c in X_test.columns if c not in X_train.columns]
    if faltantes or extras:
        raise ValueError(
            f"Las variables de train y test no coinciden. "
            f"Faltan en test: {faltantes}; sobran en test: {extras}."
        )
    X_test = X_test[X_train.columns]
    return X_train, y, X_test


def ganancia_esperada(prob_pago, monto, tasa):
    # Si paga: se gana Monto*Tasa; si cae en default: se pierde Monto.
    return monto * (prob_pago * (1.0 + tasa) - 1.0)


def calcular_puntaje_reglas(df):
    """Puntaje 0..1 por percentiles: más ingreso/interés y menos atraso es mejor."""
    requeridas = {
        "ingreso": "Ingreso_Mensual",
        "interes": "interes_potencial",
        "atraso": "atraso_medio",
    }
    faltan = [col for col in requeridas.values() if col not in df.columns]
    if faltan:
        raise ValueError(f"Faltan columnas para aplicar PESOS_REGLA: {faltan}")

    ingreso = pd.to_numeric(df[requeridas["ingreso"]], errors="coerce")
    interes = pd.to_numeric(df[requeridas["interes"]], errors="coerce")
    atraso = pd.to_numeric(df[requeridas["atraso"]], errors="coerce")

    # Percentiles en [0,1]. Para atraso se invierte: menos días = mejor puntaje.
    percentiles = {
        "ingreso": ingreso.rank(pct=True).fillna(0.0).to_numpy(),
        "interes": interes.rank(pct=True).fillna(0.0).to_numpy(),
        "atraso": (1.0 - atraso.rank(pct=True)).fillna(0.0).to_numpy(),
    }
    suma_pesos = sum(PESOS_REGLA.values())
    if suma_pesos <= 0:
        raise ValueError("La suma de PESOS_REGLA debe ser mayor que cero.")
    puntaje = sum(PESOS_REGLA[k] * percentiles[k] for k in PESOS_REGLA) / suma_pesos
    return np.asarray(puntaje, dtype=float)


def seleccionar_cartera(ganancia, monto, presupuesto, puntaje_reglas):
    """Selecciona préstamos rentables por puntaje ponderado sin exceder el presupuesto.

    Primero descarta ganancias esperadas no positivas. Después prioriza el puntaje
    explícito de las reglas (pesos ingreso/interés/atraso) por dólar solicitado.
    La ganancia realizada/esperada se sigue reportando como métrica, pero esta
    selección no garantiza maximizarla; prioriza cumplir las reglas ponderadas.
    """
    ganancia = np.asarray(ganancia, dtype=float)
    monto = np.asarray(monto, dtype=float)
    puntaje_reglas = np.asarray(puntaje_reglas, dtype=float)
    aprobado = np.zeros(len(monto), dtype=int)
    elegibles = np.where(
        (monto > 0) & np.isfinite(ganancia) & (ganancia > 0)
        & np.isfinite(puntaje_reglas)
    )[0]
    if len(elegibles) == 0 or presupuesto <= 0:
        return aprobado

    # Mochila 0/1: maximiza el puntaje ponderado de reglas con el presupuesto.
    try:
        from scipy.optimize import Bounds, LinearConstraint, milp
        resultado = milp(
            c=-puntaje_reglas[elegibles],
            integrality=np.ones(len(elegibles)),
            bounds=Bounds(np.zeros(len(elegibles)), np.ones(len(elegibles))),
            constraints=LinearConstraint(
                monto[elegibles].reshape(1, -1), -np.inf, presupuesto
            ),
            options={"time_limit": 60},
        )
        if resultado.x is not None and resultado.success:
            aprobado[elegibles] = (resultado.x >= 0.5).astype(int)
            return aprobado
    except Exception:
        pass

    # Alternativa si el optimizador no está disponible: puntaje por dólar.
    orden = elegibles[np.argsort(-(puntaje_reglas[elegibles] / monto[elegibles]))]
    gastado = 0.0
    for i in orden:
        if gastado + monto[i] <= presupuesto:
            aprobado[i] = 1
            gastado += monto[i]
    return aprobado


def ganancia_real(y_real, monto, tasa, aprobado):
    ganancia_por_solicitud = np.where(
        np.asarray(y_real) == 1,
        np.asarray(monto) * np.asarray(tasa),
        -np.asarray(monto),
    )
    return float((ganancia_por_solicitud * np.asarray(aprobado)).sum())


def evaluar(modelo, X, y, df, presupuesto):
    clases = list(modelo.named_steps["arbol"].classes_)
    prob = modelo.predict_proba(X)[:, clases.index(1)]
    monto = df["Monto_Solicitado"].to_numpy(dtype=float)
    tasa = df["Tasa_Interes"].to_numpy(dtype=float)
    puntaje_reglas = calcular_puntaje_reglas(df)
    aprobado = seleccionar_cartera(
        ganancia_esperada(prob, monto, tasa), monto, presupuesto, puntaje_reglas
    )
    gastado = float((monto * aprobado).sum())
    return {
        "ganancia_neta": ganancia_real(y, monto, tasa, aprobado),
        "roi_sobre_capital_prestado": (
            ganancia_real(y, monto, tasa, aprobado) / gastado if gastado > 0 else 0.0
        ),
        "aprobadas": int(aprobado.sum()),
        "monto_aprobado": gastado,
    }


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    train, test = cargar_datos()
    X, y, X_test = preparar_xy(train, test)

    if not set(y.unique()).issubset({0, 1}) or len(y.unique()) != 2:
        raise ValueError("Estado_Final debe contener las dos clases 0 (default) y 1 (pagado).")
    for col in ["Monto_Solicitado", "Tasa_Interes"]:
        if col not in train.columns or col not in test.columns:
            raise ValueError(f"Falta la columna requerida para calcular ganancia: {col}.")

    # Hold-out estratificado para evaluar en filas no usadas para entrenar.
    idx_fit, idx_val = train_test_split(
        np.arange(len(train)), test_size=0.30, stratify=y, random_state=SEED
    )
    modelo_val = crear_modelo(X)
    modelo_val.fit(X.iloc[idx_fit], y.iloc[idx_fit])
    presupuesto_val = min(
        PRESUPUESTO, float(train.iloc[idx_val]["Monto_Solicitado"].sum())
    )
    holdout = evaluar(
        modelo_val, X.iloc[idx_val], y.iloc[idx_val],
        train.iloc[idx_val], presupuesto_val
    )

    # Validación cruzada estratificada de 5 folds.
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    folds = []
    for idx_fit, idx_eval in skf.split(X, y):
        modelo_fold = crear_modelo(X)
        modelo_fold.fit(X.iloc[idx_fit], y.iloc[idx_fit])
        presupuesto_fold = min(
            PRESUPUESTO, float(train.iloc[idx_eval]["Monto_Solicitado"].sum())
        )
        folds.append(evaluar(
            modelo_fold, X.iloc[idx_eval], y.iloc[idx_eval],
            train.iloc[idx_eval], presupuesto_fold
        ))

    cv = {
        "ganancia_neta_media": float(np.mean([r["ganancia_neta"] for r in folds])),
        "roi_medio": float(np.mean([r["roi_sobre_capital_prestado"] for r in folds])),
        "detalle_folds": folds,
    }

    # Modelo final entrenado con todo train_clean.csv.
    modelo_final = crear_modelo(X)
    modelo_final.fit(X, y)
    clases = list(modelo_final.named_steps["arbol"].classes_)
    prob_test = modelo_final.predict_proba(X_test)[:, clases.index(1)]

    monto_test = test["Monto_Solicitado"].to_numpy(dtype=float)
    tasa_test = test["Tasa_Interes"].to_numpy(dtype=float)
    ganancia_test = ganancia_esperada(prob_test, monto_test, tasa_test)
    puntaje_reglas_test = calcular_puntaje_reglas(test)
    aprobado = seleccionar_cartera(
        ganancia_test, monto_test, PRESUPUESTO, puntaje_reglas_test
    )

    # Archivo final con los nombres y el orden de columnas solicitados.
    salida = pd.DataFrame({
        "ID_Solicitud": test["ID_Solicitud"].to_numpy(),
        "Decision_Aprobacion": aprobado.astype(int),
    })
    salida.to_csv(OUT_DIR / "solicitudes_test.csv", index=False)

    monto_aprobado = float((monto_test * aprobado).sum())
    resultados = {
        "modelo": "DecisionTreeClassifier",
        "n_train": int(len(train)),
        "n_test": int(len(test)),
        "pct_pagaron_train": float(y.mean()),
        "presupuesto_usd": PRESUPUESTO,
        "pesos_regla": PESOS_REGLA,
        "criterio_seleccion": "maximizar puntaje ponderado de reglas entre préstamos con ganancia esperada positiva, sujeto al presupuesto",
        "holdout": holdout,
        "cv_5_folds": cv,
        "test": {
            "aprobadas": int(aprobado.sum()),
            "monto_aprobado": monto_aprobado,
            "presupuesto_restante": float(PRESUPUESTO - monto_aprobado),
            "ganancia_esperada_total": float((ganancia_test * aprobado).sum()),
        },
    }
    with open(OUT_DIR / "resultados.json", "w", encoding="utf-8") as f:
        json.dump(resultados, f, indent=2, ensure_ascii=False)

    assert monto_aprobado <= PRESUPUESTO + 1e-6, "Se superó el presupuesto."
    assert salida.columns.tolist() == ["ID_Solicitud", "Decision_Aprobacion"]
    assert salida["ID_Solicitud"].is_unique

    print(f"Árbol de Decisión | train: {len(train)} | test: {len(test)}")
    print(f"Hold-out: ganancia neta ${holdout['ganancia_neta']:,.2f} | "
          f"ROI: {holdout['roi_sobre_capital_prestado']:.2%}")
    print(f"CV: ganancia neta media ${cv['ganancia_neta_media']:,.2f} | "
          f"ROI medio: {cv['roi_medio']:.2%}")
    print(f"Test: {aprobado.sum()} aprobadas | monto ${monto_aprobado:,.2f} "
          f"de ${PRESUPUESTO:,.2f}")
    print(f"Archivo final: {OUT_DIR / 'solicitudes_test.csv'}")


if __name__ == "__main__":
    main()
