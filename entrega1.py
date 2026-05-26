from simpleai.search import SearchProblem, astar


class RoverProblem(SearchProblem):

    # -------------------------------------------------
    # CONSTRUCTOR
    # -------------------------------------------------

    def __init__(
        self,
        rover_inicio,
        bateria_inicial,
        zonas_sombra,
        muestras_igneas,
        muestras_sedimentarias
    ):

        # Guardamos las zonas de sombra en un set
        # para que las búsquedas sean más rápidas
        self.zonas_sombra = set(zonas_sombra)

        # Diccionario donde:
        # clave -> coordenada
        # valor -> tipo de muestra
        self.muestras = {}

        # Agregamos las muestras ígneas
        for m in muestras_igneas:
            self.muestras[m] = "ignea"

        # Agregamos las muestras sedimentarias
        for m in muestras_sedimentarias:
            self.muestras[m] = "sedimentaria"

        # -------------------------------------------------
        # ESTRUCTURA DEL ESTADO
        # -------------------------------------------------
        #
        # (
        #   posicion,            -> (fila, columna)
        #   bateria,             -> batería actual
        #   taladro,             -> taladro equipado
        #   carga,               -> cantidad de muestras cargadas
        #   muestras_restantes,  -> muestras que faltan recolectar
        #   muestras_cargadas    -> muestras actualmente cargadas
        # )
        #

        estado_inicial = (
            rover_inicio,                    # posición inicial
            bateria_inicial,                # batería inicial
            None,                           # comienza sin taladro
            0,                              # sin carga
            frozenset(self.muestras.keys()),# todas las muestras faltan
            tuple()                         # no lleva muestras encima
        )

        # Llamamos al constructor de SearchProblem
        super().__init__(estado_inicial)

    # =================================================
    # ACCIONES POSIBLES
    # =================================================

    def actions(self, state):

        # Lista donde guardaremos las acciones válidas
        acciones = []

        # Desempaquetamos el estado
        (
            posicion,
            bateria,
            taladro,
            carga,
            restantes,
            cargadas
        ) = state

        # Obtenemos fila y columna actuales
        fila, col = posicion

        # ------------------------------------------------
        # MOVIMIENTOS NORMALES
        # ------------------------------------------------
        #
        # El rover puede moverse:
        # arriba, abajo, izquierda o derecha
        #

        direcciones = [
            (-1, 0),  # arriba
            (1, 0),   # abajo
            (0, -1),  # izquierda
            (0, 1)    # derecha
        ]

        # Recorremos cada dirección posible
        for df, dc in direcciones:

            # Nueva posición
            nf = fila + df
            nc = col + dc

            # Solo puede moverse si la batería
            # no quedará en 0
            if bateria > 1:

                acciones.append(
                    ("moverse", (nf, nc))
                )

        # ------------------------------------------------
        # SOBREMARCHA
        # ------------------------------------------------
        #
        # El rover avanza exactamente 2 casillas
        # en línea recta
        #

        for df, dc in direcciones:

            nf = fila + 2 * df
            nc = col + 2 * dc

            # La sobremarcha consume 4 baterías
            # así que necesitamos más de 4
            if bateria > 4:

                acciones.append(
                    ("sobremarcha", (nf, nc))
                )

        # ------------------------------------------------
        # EQUIPAR TALADRO
        # ------------------------------------------------

        # Si no tiene equipado el térmico,
        # puede equiparlo
        if taladro != "termico" and bateria > 1:

            acciones.append(
                ("equipar", "termico")
            )

        # Si no tiene equipado el de percusión,
        # puede equiparlo
        if taladro != "percusion" and bateria > 1:

            acciones.append(
                ("equipar", "percusion")
            )

        # ------------------------------------------------
        # RECOLECTAR MUESTRA
        # ------------------------------------------------

        # Solo puede recolectar si:
        # - está parado sobre una muestra
        # - tiene espacio en la carga
        # - tiene batería suficiente
        if posicion in restantes and carga < 2 and bateria > 3:

            # Obtenemos el tipo de muestra
            tipo = self.muestras[posicion]

            # Verificamos que tenga el taladro correcto
            if (
                (tipo == "ignea" and taladro == "termico")
                or
                (tipo == "sedimentaria" and taladro == "percusion")
            ):

                acciones.append(
                    ("recolectar", tipo)
                )

        # ------------------------------------------------
        # DEPOSITAR MUESTRAS
        # ------------------------------------------------

        # Solo puede depositar si tiene carga
        # y batería suficiente
        if carga > 0 and bateria > 1:

            total_restantes = len(restantes)

            # Puede depositar si:
            # - tiene 2 muestras
            # - o si es la última muestra existente
            if carga == 2 or total_restantes == 0:

                acciones.append(
                    ("depositar", None)
                )

        # ------------------------------------------------
        # RECARGAR BATERÍA
        # ------------------------------------------------

        # Solo puede recargar si:
        # - NO está en una zona de sombra
        # - la batería no está completa
        if posicion not in self.zonas_sombra and bateria < 20:

            acciones.append(
                ("recargar", None)
            )

        # Devolvemos todas las acciones posibles
        return acciones

    # =================================================
    # RESULTADO DE UNA ACCIÓN
    # =================================================

    def result(self, state, action):

        # Desempaquetamos el estado
        (
            posicion,
            bateria,
            taladro,
            carga,
            restantes,
            cargadas
        ) = state

        # Convertimos para poder modificarlos
        restantes = set(restantes)
        cargadas = list(cargadas)

        # Desempaquetamos la acción
        tipo, parametro = action

        # ------------------------------------------------
        # MOVERSE
        # ------------------------------------------------

        if tipo == "moverse":

            posicion = parametro
            bateria -= 1

        # ------------------------------------------------
        # SOBREMARCHA
        # ------------------------------------------------

        elif tipo == "sobremarcha":

            posicion = parametro
            bateria -= 4

        # ------------------------------------------------
        # EQUIPAR TALADRO
        # ------------------------------------------------

        elif tipo == "equipar":

            taladro = parametro
            bateria -= 1

        # ------------------------------------------------
        # RECOLECTAR
        # ------------------------------------------------

        elif tipo == "recolectar":

            # Eliminamos la muestra de las restantes
            restantes.remove(posicion)

            # La agregamos a las cargadas
            cargadas.append(posicion)

            # Aumentamos la carga
            carga += 1

            # Consumimos batería
            bateria -= 3

        # ------------------------------------------------
        # DEPOSITAR
        # ------------------------------------------------

        elif tipo == "depositar":

            # Vacía completamente la carga
            carga = 0

            # Ya no lleva muestras encima
            cargadas = []

            bateria -= 1

        # ------------------------------------------------
        # RECARGAR
        # ------------------------------------------------

        elif tipo == "recargar":

            # Recupera hasta 10 unidades
            # sin superar el máximo de 20
            bateria = min(20, bateria + 10)

        # Retornamos el nuevo estado
        return (
            posicion,
            bateria,
            taladro,
            carga,
            frozenset(restantes),
            tuple(cargadas)
        )

    # =================================================
    # COSTO DE LAS ACCIONES
    # =================================================

    def cost(self, state, action, state2):

        tipo, _ = action

        # Movimiento normal
        if tipo == "moverse":
            return 1

        # Sobremarcha
        if tipo == "sobremarcha":
            return 1

        # Equipar taladro
        if tipo == "equipar":
            return 3

        # Recolectar muestra
        if tipo == "recolectar":
            return 2

        # Depositar muestras
        if tipo == "depositar":

            # El costo depende de cuántas
            # muestras llevaba cargadas
            carga = state[3]

            return carga

        # Recargar batería
        if tipo == "recargar":
            return 4

        return 1

    # =================================================
    # OBJETIVO
    # =================================================

    def is_goal(self, state):

        # Muestras restantes
        restantes = state[4]

        # Carga actual
        carga = state[3]

        # El objetivo se cumple cuando:
        # - no quedan muestras
        # - y el rover no tiene carga
        return len(restantes) == 0 and carga == 0

    # =================================================
    # HEURÍSTICA
    # =================================================

    def heuristic(self, state):

        posicion = state[0]
        restantes = state[4]

        # Si no quedan muestras,
        # el costo estimado es 0
        if not restantes:
            return 0

        fila, col = posicion

        distancias = []

        # Calculamos distancia Manhattan
        # hasta cada muestra restante
        for rf, rc in restantes:

            dist = abs(fila - rf) + abs(col - rc)

            distancias.append(dist)

        # ------------------------------------------------
        # HEURÍSTICA ADMISIBLE
        # ------------------------------------------------
        #
        # Usamos:
        # distancia mínima
        # + cantidad de muestras restantes
        #
        # Nunca sobreestima el costo real
        #

        return min(distancias) + len(restantes)


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

    # Creamos el problema
    problema = RoverProblem(
        rover_inicio,
        bateria_inicial,
        zonas_sombra,
        muestras_igneas,
        muestras_sedimentarias
    )

    # Resolvemos usando A*
    resultado = astar(problema)

    # Lista final de acciones
    acciones = []

    # path() devuelve:
    # [(accion, estado), ...]
    #
    # Saltamos el primer elemento porque
    # corresponde al estado inicial
    for accion, estado in resultado.path()[1:]:

        acciones.append(accion)

    # Devolvemos la secuencia de acciones
    return acciones


# =====================================================
# EJEMPLO DE USO
# =====================================================

acciones = planear_rover(
    rover_inicio=(0, 0),
    bateria_inicial=20,
    zonas_sombra=[(0, 1), (0, 2)],
    muestras_igneas=[(1, 1), (1, 2)],
    muestras_sedimentarias=[(2, 3)],
)

# Mostramos el plan encontrado
print(acciones)