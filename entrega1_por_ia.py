# entrega1.py

from math import ceil
from simpleai.search import SearchProblem, astar


MAX_BATERIA = 20


def distancia(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


class RoverProblem(SearchProblem):

    def __init__(
        self,
        rover_inicio,
        bateria_inicial,
        zonas_sombra,
        muestras_igneas,
        muestras_sedimentarias,
    ):

        self.zonas_sombra = set(zonas_sombra)

        self.muestras = {}

        for pos in muestras_igneas:
            self.muestras[pos] = "ignea"

        for pos in muestras_sedimentarias:
            self.muestras[pos] = "sedimentaria"

        posiciones = (
            [rover_inicio]
            + list(zonas_sombra)
            + list(muestras_igneas)
            + list(muestras_sedimentarias)
        )

        filas = [p[0] for p in posiciones]
        columnas = [p[1] for p in posiciones]

        self.min_fila = min(filas) - 2
        self.max_fila = max(filas) + 2

        self.min_col = min(columnas) - 2
        self.max_col = max(columnas) + 2

        estado_inicial = (
            rover_inicio,
            bateria_inicial,
            None,
            tuple(),
            frozenset(
                (pos, tipo)
                for pos, tipo in self.muestras.items()
            ),
        )

        super().__init__(estado_inicial)

    def dentro_limites(self, pos):
        f, c = pos

        return (
            self.min_fila <= f <= self.max_fila
            and
            self.min_col <= c <= self.max_col
        )

    def actions(self, state):

        (
            posicion,
            bateria,
            taladro,
            carga,
            restantes,
        ) = state

        acciones = []

        fila, col = posicion

        # movimientos normales
        direcciones = [
            (-1, 0),
            (1, 0),
            (0, -1),
            (0, 1),
        ]

        for df, dc in direcciones:

            nueva = (fila + df, col + dc)

            if (
                self.dentro_limites(nueva)
                and bateria - 1 > 0
            ):
                acciones.append(
                    ("moverse", nueva)
                )

        # sobremarcha
        for df, dc in direcciones:

            nueva = (
                fila + 2 * df,
                col + 2 * dc,
            )

            if (
                self.dentro_limites(nueva)
                and bateria - 4 > 0
            ):
                acciones.append(
                    ("sobremarcha", nueva)
                )

        # equipar
        if bateria - 1 > 0:

            if taladro != "termico":
                acciones.append(
                    ("equipar", "termico")
                )

            if taladro != "percusion":
                acciones.append(
                    ("equipar", "percusion")
                )

        # recolectar
        restantes_dict = dict(restantes)

        if (
            posicion in restantes_dict
            and len(carga) < 2
            and bateria - 3 > 0
        ):

            tipo = restantes_dict[posicion]

            taladro_correcto = (
                (tipo == "ignea" and taladro == "termico")
                or
                (tipo == "sedimentaria" and taladro == "percusion")
            )

            if taladro_correcto:
                acciones.append(
                    ("recolectar", tipo)
                )

        # depositar
        if (
            len(carga) > 0
            and bateria - 1 > 0
        ):

            muestras_restantes = len(restantes)

            ultima = (
                muestras_restantes == 0
            )

            if len(carga) == 2 or ultima:
                acciones.append(
                    ("depositar", None)
                )

        # recargar
        if (
            posicion not in self.zonas_sombra
            and bateria < MAX_BATERIA
        ):
            acciones.append(
                ("recargar", None)
            )

        return acciones

    def result(self, state, action):

        (
            posicion,
            bateria,
            taladro,
            carga,
            restantes,
        ) = state

        accion, parametro = action

        restantes = set(restantes)
        carga = list(carga)

        if accion == "moverse":
            posicion = parametro
            bateria -= 1

        elif accion == "sobremarcha":
            posicion = parametro
            bateria -= 4

        elif accion == "equipar":
            taladro = parametro
            bateria -= 1

        elif accion == "recolectar":

            tipo = parametro

            restantes.remove((posicion, tipo))

            carga.append(tipo)

            bateria -= 3

        elif accion == "depositar":

            bateria -= 1

            carga = []

        elif accion == "recargar":

            bateria = min(
                MAX_BATERIA,
                bateria + 10,
            )

        return (
            posicion,
            bateria,
            taladro,
            tuple(carga),
            frozenset(restantes),
        )

    def cost(self, state, action, state2):

        accion, parametro = action

        if accion == "moverse":
            return 1

        if accion == "sobremarcha":
            return 1

        if accion == "equipar":
            return 3

        if accion == "recolectar":
            return 2

        if accion == "depositar":
            return len(state[3])

        if accion == "recargar":
            return 4

        return 1

    def is_goal(self, state):

        (
            _,
            _,
            _,
            carga,
            restantes,
        ) = state

        return (
            len(restantes) == 0
            and len(carga) == 0
        )

    def heuristic(self, state):

        (
            posicion,
            _,
            _,
            carga,
            restantes,
        ) = state

        restantes = list(restantes)

        if not restantes:
            return 0 if not carga else len(carga)

        dist_min = min(
            distancia(posicion, pos)
            for pos, _ in restantes
        )

        cantidad = len(restantes)

        return (
            dist_min
            + 2 * cantidad
            + ceil(cantidad / 2)
        )


def planear_rover(
    rover_inicio,
    bateria_inicial,
    zonas_sombra,
    muestras_igneas,
    muestras_sedimentarias,
):

    problema = RoverProblem(
        rover_inicio,
        bateria_inicial,
        zonas_sombra,
        muestras_igneas,
        muestras_sedimentarias,
    )

    resultado = astar(
        problema,
        graph_search=True,
    )

    camino = resultado.path()

    acciones = []

    for accion, _estado in camino[1:]:
        acciones.append(accion)

    return acciones