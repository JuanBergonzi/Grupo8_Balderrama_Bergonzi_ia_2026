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

        # ---------------------------------------------
        # ZONAS DE SOMBRA
        # ---------------------------------------------

        self.zonas_sombra = set(zonas_sombra)

        # ---------------------------------------------
        # MUESTRAS
        # ---------------------------------------------

        self.muestras = {}

        for m in muestras_igneas:
            self.muestras[m] = "ignea"

        for m in muestras_sedimentarias:
            self.muestras[m] = "sedimentaria"

        # ---------------------------------------------
        # LÍMITES DEL MAPA
        # ---------------------------------------------

        todos = (
            [rover_inicio]
            + list(zonas_sombra)
            + list(muestras_igneas)
            + list(muestras_sedimentarias)
        )

        filas = [f for f, c in todos]
        columnas = [c for f, c in todos]

        margen = 5

        self.min_f = min(filas) - margen
        self.max_f = max(filas) + margen

        self.min_c = min(columnas) - margen
        self.max_c = max(columnas) + margen

        # ---------------------------------------------
        # ESTADO INICIAL
        # ---------------------------------------------
        #
        # (
        #   posicion,
        #   bateria,
        #   taladro,
        #   carga,
        #   muestras_restantes
        # )

        estado_inicial = (
            rover_inicio,
            bateria_inicial,
            None,
            0,
            frozenset(self.muestras.keys())
        )

        super().__init__(estado_inicial)

    # =================================================
    # ACCIONES
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

        direcciones = [
            (-1, 0),
            (1, 0),
            (0, -1),
            (0, 1)
        ]

        # ---------------------------------------------
        # MOVIMIENTO NORMAL
        # ---------------------------------------------

        for df, dc in direcciones:

            nf = fila + df
            nc = col + dc

            dentro_limites = (
                self.min_f <= nf <= self.max_f
                and
                self.min_c <= nc <= self.max_c
            )

            if bateria > 1 and dentro_limites:

                acciones.append(
                    ("moverse", (nf, nc))
                )

        # ---------------------------------------------
        # SOBREMARCHA
        # ---------------------------------------------

        for df, dc in direcciones:

            nf = fila + 2 * df
            nc = col + 2 * dc

            dentro_limites = (
                self.min_f <= nf <= self.max_f
                and
                self.min_c <= nc <= self.max_c
            )

            if bateria > 4 and dentro_limites:

                acciones.append(
                    ("sobremarcha", (nf, nc))
                )

        # ---------------------------------------------
        # EQUIPAR TALADRO
        # ---------------------------------------------

        if taladro != "termico" and bateria > 1:

            acciones.append(
                ("equipar", "termico")
            )

        if taladro != "percusion" and bateria > 1:

            acciones.append(
                ("equipar", "percusion")
            )

        # ---------------------------------------------
        # RECOLECTAR
        # ---------------------------------------------

        if posicion in restantes and carga < 2 and bateria > 3:

            tipo = self.muestras[posicion]

            if (
                (tipo == "ignea" and taladro == "termico")
                or
                (tipo == "sedimentaria" and taladro == "percusion")
            ):

                acciones.append(
                    ("recolectar", tipo)
                )

        # ---------------------------------------------
        # DEPOSITAR
        # ---------------------------------------------

        if carga > 0 and bateria > 1:

            total_restantes = len(restantes)

            if carga == 2 or total_restantes == 0:

                acciones.append(
                    ("depositar", None)
                )

        # ---------------------------------------------
        # RECARGAR
        # ---------------------------------------------

        if posicion not in self.zonas_sombra and bateria < 20:

            acciones.append(
                ("recargar", None)
            )

        return acciones

    # =================================================
    # RESULTADO
    # =================================================

    def result(self, state, action):

        (
            posicion,
            bateria,
            taladro,
            carga,
            restantes
        ) = state

        restantes = set(restantes)

        tipo, parametro = action

        # ---------------------------------------------
        # MOVERSE
        # ---------------------------------------------

        if tipo == "moverse":

            posicion = parametro
            bateria -= 1

        # ---------------------------------------------
        # SOBREMARCHA
        # ---------------------------------------------

        elif tipo == "sobremarcha":

            posicion = parametro
            bateria -= 4

        # ---------------------------------------------
        # EQUIPAR
        # ---------------------------------------------

        elif tipo == "equipar":

            taladro = parametro
            bateria -= 1

        # ---------------------------------------------
        # RECOLECTAR
        # ---------------------------------------------

        elif tipo == "recolectar":

            restantes.remove(posicion)

            carga += 1

            bateria -= 3

        # ---------------------------------------------
        # DEPOSITAR
        # ---------------------------------------------

        elif tipo == "depositar":

            carga = 0

            bateria -= 1

        # ---------------------------------------------
        # RECARGAR
        # ---------------------------------------------

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
    # COSTO
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

        if tipo == "depositar":

            carga = state[3]

            return carga

        if tipo == "recargar":
            return 4

        return 1

    # =================================================
    # META
    # =================================================

    def is_goal(self, state):

        restantes = state[4]

        carga = state[3]

        return len(restantes) == 0 and carga == 0

    # =================================================
    # HEURÍSTICA
    # =================================================

    def heuristic(self, state):

        posicion = state[0]

        restantes = state[4]

        carga = state[3]

        if not restantes:

            return carga

        fila, col = posicion

        distancias = []

        for rf, rc in restantes:

            dist = abs(fila - rf) + abs(col - rc)

            distancias.append(dist)

        return min(distancias) + (2 * len(restantes)) + carga


# =====================================================
# FUNCIÓN PRINCIPAL
# =====================================================

def planear_rover(
    rover_inicio,
    bateria_inicial,
    zonas_sombra,
    muestras_igneas,
    muestras_sedimentarias
):

    problema = RoverProblem(
        rover_inicio,
        bateria_inicial,
        zonas_sombra,
        muestras_igneas,
        muestras_sedimentarias
    )

    resultado = astar(problema)

    acciones = []

    for accion, estado in resultado.path()[1:]:

        acciones.append(accion)

    return acciones