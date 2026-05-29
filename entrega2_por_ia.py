from simpleai.search import (
    CspProblem,
    backtrack,
    MOST_CONSTRAINED_VARIABLE,
    LEAST_CONSTRAINING_VALUE
)

def build_camp(camp_size, habs, generators, labs, deposits, airlocks, craters):
    filas, columnas = camp_size
    set_craters = set(craters)

    # 1. GENERACIÓN DE VARIABLES IDENTIFICABLES
    variables = []
    list_habs = [f"hab_{i}" for i in range(habs)]
    list_gens = [f"gen_{i}" for i in range(generators)]
    list_labs = [f"lab_{i}" for i in range(labs)]
    list_deps = [f"dep_{i}" for i in range(deposits)]
    list_airs = [f"air_{i}" for i in range(airlocks)]
    
    variables.extend(list_habs)
    variables.extend(list_gens)
    variables.extend(list_labs)
    variables.extend(list_deps)
    variables.extend(list_airs)

    # Si no hay módulos que ubicar, la grilla está vacía pero válida
    if not variables:
        return []

    # 2. DEFINICIÓN DE DOMINIOS FILTRADOS
    celdas_validas = [(f, c) for f in range(filas) for c in range(columnas) if (f, c) not in set_craters]
    
    domains = {}
    for var in variables:
        if var.startswith("air_"):
            # Esclusas estrictamente en el perímetro externo
            domains[var] = [
                (f, c) for (f, c) in celdas_validas 
                if f == 0 or f == filas - 1 or c == 0 or c == columnas - 1
            ]
        elif var.startswith("hab_"):
            # Módulos habitacionales protegidos al interior (no bordes)
            domains[var] = [
                (f, c) for (f, c) in celdas_validas 
                if 0 < f < filas - 1 and 0 < c < columnas - 1
            ]
        else:
            domains[var] = list(celdas_validas)

    # 3. FUNCIONES DE RESTRICCIÓN (CONSTRAINTS)

    def son_adyacentes(pos1, pos2):
        f1, c1 = pos1
        f2, c2 = pos2
        return (abs(f1 - f2) == 1 and c1 == c2) or (abs(c1 - c2) == 1 and f1 == f2)

    # Restricción: No solapamiento en la misma celda
    def restriccion_distintos(variables_all, values):
        return len(set(values)) == len(values)

    # Restricción: Seguridad (Generador lejano a Habitacional)
    def restriccion_gen_hab(variables_all, values):
        return not son_adyacentes(values[0], values[1])

    # Restricción: Aislamiento energético (Generadores separados)
    def restriccion_gen_gen(variables_all, values):
        return not son_adyacentes(values[0], values[1])

    # Restricción: Suministro científico (Lab junto a mínimo un depósito)
    def restriccion_lab_dep(variables_all, values):
        pos_lab = values[0]
        pos_depositos = values[1:]
        return any(son_adyacentes(pos_lab, pos_dep) for pos_dep in pos_depositos)

    # Restricción: Escape de emergencia (Habitacional con entorno transitable)
    def restriccion_evacuacion_hab(variables_all, values):
        pos_hab = values[0]
        pos_otros_modulos = set(values[1:])
        
        f, c = pos_hab
        vecinos = [(f - 1, c), (f + 1, c), (f, c - 1), (f, c + 1)]
        
        for vf, vc in vecinos:
            if 0 <= vf < filas and 0 <= vc < columnas:
                if (vf, vc) not in set_craters and (vf, vc) not in pos_otros_modulos:
                    return True
        return False

    # 4. ENLACE DE RESTRICCIONES AL PROBLEMA
    constraints = []

    # Unicidad de celdas ocupadas
    constraints.append((variables, restriccion_distintos))

    # Cruzar Generadores con Habitacionales
    for g in list_gens:
        for h in list_habs:
            constraints.append(([g, h], restriccion_gen_hab))

    # Cruzar Generadores entre sí
    for i in range(len(list_gens)):
        for j in range(i + 1, len(list_gens)):
            constraints.append(([list_gens[i], list_gens[j]], restriccion_gen_gen))

    # Laboratorios requieren depósitos cargados
    if list_labs:
        if not list_deps:
            return None  # Imposible satisfacer la adyacencia
        for l in list_labs:
            constraints.append(([l] + list_deps, restriccion_lab_dep))

    # Rutas de evacuación para cada Habitacional contra el resto de la base
    for h in list_habs:
        otros = [v for v in variables if v != h]
        constraints.append(([h] + otros, restriccion_evacuacion_hab))

    # 5. INSTANCIACIÓN Y RESOLUCIÓN CON HEURÍSTICAS SOLICITADAS
    problema_csp = CspProblem(variables, domains, constraints)
    
    solucion = backtrack(
        problema_csp,
        variable_heuristic=MOST_CONSTRAINED_VARIABLE,
        value_heuristic=LEAST_CONSTRAINING_VALUE
    )

    if not solucion:
        return None

    # 6. FORMATEO DE SALIDA TRADUCIDA
    resultado_formateado = []
    for var, pos in solucion.items():
        tipo_modulo = var.split("_")[0]
        resultado_formateado.append((tipo_modulo, pos[0], pos[1]))

    return resultado_formateado