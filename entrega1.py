from simpleai.search import SearchProblem, astar


class RoverProblem(SearchProblem):

    def __init__(
        self,
        rover_inicio,
        bateria_inicial,
        zonas_sombra,
        muestras_igneas,
        muestras_sedimentarias
    ):

        # =================================================
        # GUARDAMOS LAS ZONAS DE SOMBRA
        # =================================================
        # Se usan para impedir acciones de recarga
        # en esas coordenadas.
        # Las convertimos a set para búsquedas rápidas.
        self.zonas_sombra = set(zonas_sombra)

        # =================================================
        # DICCIONARIO DE MUESTRAS
        # =================================================
        # Asociamos cada coordenada con su tipo:
        # "ignea" o "sedimentaria"
        self.muestras = {}

        for m in muestras_igneas:
            self.muestras[m] = "ignea"

        for m in muestras_sedimentarias:
            self.muestras[m] = "sedimentaria"

        # =================================================
        # LIMITES DEL MAPA
        # =================================================
        # Calculamos un rectángulo de búsqueda para
        # evitar que el rover explore infinitamente.
        todos = (
            [rover_inicio]
            + list(zonas_sombra)
            + list(muestras_igneas)
            + list(muestras_sedimentarias)
        )

        filas = [f for f, c in todos]
        columnas = [c for f, c in todos]

        # Margen extra para permitir maniobras
        # y rodeos necesarios.
        margen = 3

        self.min_f = min(filas) - margen
        self.max_f = max(filas) + margen

        self.min_c = min(columnas) - margen
        self.max_c = max(columnas) + margen

        # =================================================
        # ESTADO INICIAL
        # =================================================
        #
        # estado = (
        #   posicion,
        #   bateria,
        #   taladro_actual,
        #   carga_actual,
        #   muestras_restantes
        # )
        #
        estado_inicial = (
            rover_inicio,
            bateria_inicial,
            "ninguno",
            0,
            frozenset(self.muestras.keys())
        )

        super().__init__(estado_inicial)

    # =================================================
    # ACCIONES POSIBLES
    # =================================================

    def actions(self, state):

        acciones = []

        (
            posicion,
            bateria,
            taladro,
            carga,
            restantes
        ) = state

        fila, col = posicion

        # Movimientos posibles:
        # arriba, abajo, izquierda y derecha
        direcciones = [
            (-1, 0),
            (1, 0),
            (0, -1),
            (0, 1)
        ]

        # =================================================
        # RECARGAR
        # =================================================
        #
        # El rover recarga cuando la batería
        # está cerca del mínimo necesario.
        #
        # Esto evita:
        # - quedarse sin batería
        # - recargar demasiado seguido
        #

        energia_minima = 999

        if restantes:

            distancias = []

            for r, c in restantes:

                d = abs(fila - r) + abs(col - c)
                distancias.append(d)

            cercana = min(distancias)

            # Dejamos un margen extra de energía
            energia_minima = cercana + 4

        if (
            posicion not in self.zonas_sombra
            and bateria <= energia_minima
        ):

            acciones.append(
                ("recargar", None)
            )

        # =================================================
        # RECOLECTAR MUESTRAS
        # =================================================
        #
        # Solo se puede recolectar si:
        # - estamos sobre una muestra
        # - hay espacio en la carga
        # - hay batería suficiente
        # - el taladro correcto está equipado
        #

        if (
            posicion in restantes
            and carga < 2
            and bateria > 3
        ):

            tipo = self.muestras[posicion]

            # Muestra ígnea → taladro térmico
            if (
                tipo == "ignea"
                and taladro == "termico"
            ):

                acciones.append(
                    ("recolectar", "ignea")
                )

            # Muestra sedimentaria → taladro percusión
            elif (
                tipo == "sedimentaria"
                and taladro == "percusion"
            ):

                acciones.append(
                    ("recolectar", "sedimentaria")
                )

        # =================================================
        # EQUIPAR TALADRO
        # =================================================
        #
        # Solo equipamos el taladro necesario
        # si estamos sobre una muestra y aún
        # no tenemos el correcto.
        #
        # Esto evita cambios innecesarios.
        #

        if posicion in restantes and bateria > 1:

            tipo = self.muestras[posicion]

            if (
                tipo == "ignea"
                and taladro != "termico"
            ):

                acciones.append(
                    ("equipar", "termico")
                )

            elif (
                tipo == "sedimentaria"
                and taladro != "percusion"
            ):

                acciones.append(
                    ("equipar", "percusion")
                )

        # =================================================
        # DEPOSITAR CARGA
        # =================================================
        #
        # Se puede depositar:
        # - cuando la carga está llena (2)
        # - o cuando ya no quedan muestras
        #

        if carga > 0 and bateria > 1:

            # Última muestra restante
            if len(restantes) == 0:

                acciones.append(
                    ("depositar", None)
                )

            # Cápsula completa
            elif carga == 2:

                acciones.append(
                    ("depositar", None)
                )

        # =================================================
        # MOVIMIENTO NORMAL
        # =================================================
        #
        # Movimiento de 1 casilla.
        #
        # Se aplica una poda suave:
        # el rover puede alejarse un poco,
        # pero no demasiado.
        #
        # Esto mejora muchísimo el tiempo
        # de búsqueda sin romper casos
        # complejos.
        #

        if restantes:

            distancia_actual = min(
                abs(fila - rf) + abs(col - rc)
                for rf, rc in restantes
            )

            for df, dc in direcciones:

                nf = fila + df
                nc = col + dc

                # Verificamos que el movimiento
                # permanezca dentro del mapa.
                dentro_limites = (
                    self.min_f <= nf <= self.max_f
                    and
                    self.min_c <= nc <= self.max_c
                )

                if not dentro_limites:
                    continue

                # Nunca dejamos batería en 0
                if bateria <= 1:
                    continue

                nueva_distancia = min(
                    abs(nf - rf) + abs(nc - rc)
                    for rf, rc in restantes
                )

                # =================================================
                # PODA SUAVE
                # =================================================
                #
                # Permitimos alejarse un máximo
                # de 2 casillas respecto a la
                # mejor distancia actual.
                #
                # Esto:
                # - mantiene rutas válidas
                # - evita explosión de estados
                # - acelera muchísimo A*
                #

                if nueva_distancia <= distancia_actual + 2:

                    acciones.append(
                        ("moverse", (nf, nc))
                    )

        # =================================================
        # SOBREMARCHA
        # =================================================
        #
        # Movimiento de 2 casillas rectas.
        #
        # Consume mucha batería pero cuesta
        # solamente 1 minuto.
        #

        if bateria > 4:

            for df, dc in direcciones:

                nf = fila + 2 * df
                nc = col + 2 * dc

                dentro_limites = (
                    self.min_f <= nf <= self.max_f
                    and
                    self.min_c <= nc <= self.max_c
                )

                if not dentro_limites:
                    continue

                acciones.append(
                    ("sobremarcha", (nf, nc))
                )

        return acciones

    # =================================================
    # RESULTADO DE UNA ACCION
    # =================================================

    def result(self, state, action):

        (
            posicion,
            bateria,
            taladro,
            carga,
            restantes
        ) = state

        tipo, parametro = action

        restantes = set(restantes)

        # Movimiento normal
        if tipo == "moverse":

            posicion = parametro
            bateria -= 1

        # Sobremarcha
        elif tipo == "sobremarcha":

            posicion = parametro
            bateria -= 4

        # Cambio de taladro
        elif tipo == "equipar":

            taladro = parametro
            bateria -= 1

        # Recolección de muestra
        elif tipo == "recolectar":

            restantes.remove(posicion)

            carga += 1

            bateria -= 3

        # Depositar cápsula
        elif tipo == "depositar":

            carga = 0

            bateria -= 1

        # Recargar batería
        elif tipo == "recargar":

            bateria = min(20, bateria + 10)

        return (
            posicion,
            bateria,
            taladro,
            carga,
            frozenset(restantes)
        )

    # =================================================
    # COSTO DE CADA ACCION
    # =================================================

    def cost(self, state, action, state2):

        tipo, _ = action

        if tipo == "moverse":
            return 1

        if tipo == "sobremarcha":
            return 1

        if tipo == "equipar":
            return 3

        if tipo == "recolectar":
            return 2

        # Depositar cuesta según cantidad
        # de muestras cargadas
        if tipo == "depositar":

            carga = state[3]

            return carga

        if tipo == "recargar":
            return 4

        return 1

    # =================================================
    # ESTADO META
    # =================================================
    #
    # La misión termina cuando:
    # - no quedan muestras
    # - la carga está vacía
    #

    def is_goal(self, state):

        restantes = state[4]

        carga = state[3]

        return (
            len(restantes) == 0
            and carga == 0
        )

    # =================================================
    # HEURISTICA
    # =================================================
    #
    # Estimación optimista:
    # distancia Manhattan mínima
    # dividida por 3.
    #
    # Ayuda a A* a priorizar estados
    # cercanos a muestras.
    #

    def heuristic(self, state):

        posicion, bateria, taladro, carga, restantes = state

        # Si no quedan muestras,
        # solo falta depositar.
        if not restantes:
            return carga

        distancias = []

        for r in restantes:

            d = abs(posicion[0] - r[0]) + abs(posicion[1] - r[1])
            distancias.append(d)

        minima = min(distancias)

        return minima // 3


# =====================================================
# FUNCION PRINCIPAL
# =====================================================

def planear_rover(
    rover_inicio,
    bateria_inicial,
    zonas_sombra,
    muestras_igneas,
    muestras_sedimentarias
):

    # Creamos el problema de búsqueda
    problema = RoverProblem(
        rover_inicio,
        bateria_inicial,
        zonas_sombra,
        muestras_igneas,
        muestras_sedimentarias
    )

    # Ejecutamos A*
    resultado = astar(
        problema,
        graph_search=True
    )

    # Si no hay solución
    if resultado is None:
        return []

    acciones = []

    # Reconstruimos el camino
    # de acciones encontrado.
    for accion, estado in resultado.path()[1:]:

        acciones.append(accion)

    return acciones