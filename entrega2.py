# =====================================================
# CAMPAMENTO MARCIANO - CSP
# =====================================================
#
# Resolución mediante:
# - SimpleAI
# - Backtracking CSP
#
# =====================================================

from simpleai.search import (
    CspProblem,
    backtrack,
    MOST_CONSTRAINED_VARIABLE,
    LEAST_CONSTRAINING_VALUE
)

import itertools


# =====================================================
# FUNCIÓN PRINCIPAL
# =====================================================

def build_camp(
    camp_size,
    habs,
    generators,
    labs,
    deposits,
    airlocks,
    craters
):

    # =================================================
    # TAMAÑO DEL MAPA
    # =================================================

    filas, columnas = camp_size

    crateres = set(craters)

    # =================================================
    # VARIABLES
    # =================================================
    #
    # Cada módulo individual es una variable.
    #
    # Ejemplo:
    # hab_0
    # hab_1
    # gen_0
    #
    # =================================================

    variables = []

    for i in range(habs):
        variables.append(f"hab_{i}")

    for i in range(generators):
        variables.append(f"gen_{i}")

    for i in range(labs):
        variables.append(f"lab_{i}")

    for i in range(deposits):
        variables.append(f"dep_{i}")

    for i in range(airlocks):
        variables.append(f"air_{i}")

    # =================================================
    # DOMINIOS
    # =================================================
    #
    # Cada variable puede ubicarse
    # en cualquier celda NO cráter.
    #
    # =================================================

    posiciones_validas = []

    for f in range(filas):

        for c in range(columnas):

            if (f, c) not in crateres:
                posiciones_validas.append((f, c))

    dominios = {}

    for var in variables:
        dominios[var] = posiciones_validas

    # =================================================
    # FUNCIONES AUXILIARES
    # =================================================

    def adyacentes(pos1, pos2):
        # =============================================
        # VERIFICA ADYACENCIA ORTOGONAL
        # =============================================
        #
        # Dos posiciones son adyacentes si:
        # - están separadas exactamente
        #   por 1 casilla Manhattan
        #
        # Se consideran únicamente:
        # - arriba
        # - abajo
        # - izquierda
        # - derecha
        #
        # NO diagonales.
        #
        # Ejemplos válidos:
        # (1,1) y (1,2)
        # (3,4) y (2,4)
        #
        # Ejemplo inválido:
        # (1,1) y (2,2)
        #
        # =============================================

        f1, c1 = pos1
        f2, c2 = pos2

        distancia = abs(f1 - f2) + abs(c1 - c2)

        return distancia == 1

    def en_borde(pos):
        # =============================================
        # VERIFICA SI UNA POSICIÓN
        # ESTÁ EN EL BORDE DEL MAPA
        # =============================================
        #
        # Una celda pertenece al borde si:
        #
        # - está en la primera fila
        # - está en la última fila
        # - está en la primera columna
        # - está en la última columna
        #
        # Esto se utiliza para:
        # - esclusas (deben estar)
        # - habitacionales (no deben estar)
        #
        # =============================================

        f, c = pos

        return (
            f == 0
            or f == filas - 1
            or c == 0
            or c == columnas - 1
        )

    # =================================================
    # RESTRICCIONES
    # =================================================

    restricciones = []

    # =================================================
    # 1) SIN SUPERPOSICIÓN
    # =================================================
    #
    # Dos módulos no pueden ocupar
    # la misma celda.
    #
    # =================================================

    def sin_superposicion(variables, valores):

        return valores[0] != valores[1]

    for var1, var2 in itertools.combinations(variables, 2):

        restricciones.append(
            (
                (var1, var2),
                sin_superposicion
            )
        )

    # =================================================
    # 2) ESCLUSAS EN EL BORDE
    # =================================================

    def airlock_en_borde(variables, valores):

        posicion = valores[0]

        return en_borde(posicion)

    for var in variables:

        if var.startswith("air"):

            restricciones.append(
                (
                    (var,),
                    airlock_en_borde
                )
            )

    # =================================================
    # 3) HABITACIONALES AL INTERIOR
    # =================================================

    def hab_interior(variables, valores):

        posicion = valores[0]

        return not en_borde(posicion)

    for var in variables:

        if var.startswith("hab"):

            restricciones.append(
                (
                    (var,),
                    hab_interior
                )
            )

    # =================================================
    # 4) GENERADOR NO ADYACENTE A HAB
    # =================================================

    def gen_lejos_hab(variables, valores):

        hab_pos = valores[0]
        gen_pos = valores[1]

        return not adyacentes(hab_pos, gen_pos)

    hab_vars = [
        v for v in variables
        if v.startswith("hab")
    ]

    gen_vars = [
        v for v in variables
        if v.startswith("gen")
    ]

    for hab_var in hab_vars:

        for gen_var in gen_vars:

            restricciones.append(
                (
                    (hab_var, gen_var),
                    gen_lejos_hab
                )
            )

    # =================================================
    # 5) GENERADORES NO ADYACENTES
    # =================================================

    def gens_separados(variables, valores):

        gen1 = valores[0]
        gen2 = valores[1]

        return not adyacentes(gen1, gen2)

    for g1, g2 in itertools.combinations(gen_vars, 2):

        restricciones.append(
            (
                (g1, g2),
                gens_separados
            )
        )

    # =================================================
    # 6) LABORATORIO CON DEPÓSITO
    # =================================================
    #
    # Cada laboratorio debe tener
    # al menos un depósito adyacente.
    #
    # =================================================

    dep_vars = [
        v for v in variables
        if v.startswith("dep")
    ]

    lab_vars = [
        v for v in variables
        if v.startswith("lab")
    ]

    def lab_con_deposito(variables, valores):

        laboratorio = valores[0]

        depositos = valores[1:]

        for deposito in depositos:

            if adyacentes(laboratorio, deposito):
                return True

        return False

    for lab_var in lab_vars:

        vars_relacionadas = tuple(
            [lab_var] + dep_vars
        )

        restricciones.append(
            (
                vars_relacionadas,
                lab_con_deposito
            )
        )

    # =================================================
    # 7) RUTA DE EVACUACIÓN
    # =================================================
    #
    # Cada habitacional debe tener
    # una celda libre ortogonal vecina.
    #
    # Libre significa:
    # - sin módulo
    # - sin cráter
    #
    # =================================================

    def crear_restriccion_evacuacion(hab_var):

        def ruta_evacuacion(variables, valores):

            asignaciones = dict(
                zip(variables, valores)
            )

            posicion_hab = asignaciones[hab_var]

            f, c = posicion_hab

            vecinos = [
                (f - 1, c),
                (f + 1, c),
                (f, c - 1),
                (f, c + 1)
            ]

            posiciones_ocupadas = set(
                asignaciones.values()
            )

            for vf, vc in vecinos:

                # Dentro del mapa
                if not (
                    0 <= vf < filas
                    and
                    0 <= vc < columnas
                ):
                    continue

                pos = (vf, vc)

                # No puede ser cráter
                if pos in crateres:
                    continue

                # No puede estar ocupada
                if pos in posiciones_ocupadas:
                    continue

                # Hay salida libre
                return True

            return False

        return ruta_evacuacion

    for hab_var in hab_vars:

        restricciones.append(
            (
                tuple(variables),
                crear_restriccion_evacuacion(hab_var)
            )
        )

    # =================================================
    # CREAR PROBLEMA CSP
    # =================================================

    problema = CspProblem(
        variables,
        dominios,
        restricciones
    )

    # =================================================
    # RESOLVER CSP
    # =================================================

    solucion = backtrack(
        problema,
        variable_heuristic=MOST_CONSTRAINED_VARIABLE,
        value_heuristic=LEAST_CONSTRAINING_VALUE,
        inference=True
    )

    # =================================================
    # SI NO HAY SOLUCIÓN
    # =================================================

    if solucion is None:
        return None

    # =================================================
    # FORMATEAR RESULTADO
    # =================================================
    #
    # Formato requerido:
    #
    # [
    #   ("hab", fila, columna),
    #   ("gen", fila, columna),
    # ]
    #
    # =================================================

    resultado = []

    for variable, posicion in solucion.items():

        tipo = variable.split("_")[0]

        fila, columna = posicion

        resultado.append(
            (
                tipo,
                fila,
                columna
            )
        )

    return resultadoh