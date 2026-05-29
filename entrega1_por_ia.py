from simpleai.search import SearchProblem, astar

class RoverProblem(SearchProblem):
    def __init__(self, rover_inicio, bateria_inicial, zonas_sombra, muestras_igneas, muestras_sedimentarias):
        # Estado inicial: (pos_rover, bateria, taladro_activo, muestras_en_bodega, muestras_restantes_igneas, muestras_restantes_sedimentarias)
        self.initial_state = (
            rover_inicio,
            bateria_inicial,
            None,  # Sin taladro al inicio
            frozenset(),  # Bodega vacía
            frozenset(muestras_igneas),
            frozenset(muestras_sedimentarias)
        )
        self.zonas_sombra = set(zonas_sombra)

    def actions(self, state):
        pos, bateria, taladro, bodega, m_igneas, m_sedimentarias = state
        r, c = pos
        list_actions = []

        # --- Acción: Recargar ---
        if pos not in self.zonas_sombra and bateria < 20:
            list_actions.append(("recargar", None))

        # --- Acción: Equipar taladro ---
        if taladro != "termico" and bateria > 1:
            list_actions.append(("equipar", "termico"))
        if taladro != "percusión" and bateria > 1:
            list_actions.append(("equipar", "percusión"))

        # --- Acción: Depositar ---
        total_restantes = len(m_igneas) + len(m_sedimentarias)
        if len(bodega) > 0 and bateria > 1:
            # Se puede depositar si la bodega está llena (2) o si ya no quedan más muestras en el mapa
            if total_restantes == 0 or len(bodega) == 2:
                list_actions.append(("depositar", None))

        # --- Acción: Perforar y Recolectar ---
        if len(bodega) < 2 and bateria > 3:
            if pos in m_igneas and taladro == "termico":
                list_actions.append(("recolectar", "ignea"))
            if pos in m_sedimentarias and taladro == "percusión":
                list_actions.append(("recolectar", "sedimentaria"))

        # --- Acciones de Movimiento ---
        direcciones = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        for dr, dc in direcciones:
            # Moverse (1 celda)
            if bateria > 1:
                list_actions.append(("moverse", (r + dr, c + dc)))
            # Sobremarcha (2 celdas en línea recta)
            if bateria > 4:
                list_actions.append(("sobremarcha", (r + 2 * dr, c + 2 * dc)))

        return list_actions

    def result(self, state, action):
        pos, bateria, taladro, bodega, m_igneas, m_sedimentarias = state
        tipo, param = action

        if tipo == "recargar":
            nueva_bateria = min(20, bateria + 10)
            return (pos, nueva_bateria, taladro, bodega, m_igneas, m_sedimentarias)

        elif tipo == "equipar":
            return (pos, bateria - 1, param, bodega, m_igneas, m_sedimentarias)

        elif tipo == "recolectar":
            nueva_bodega = set(bodega)
            if param == "ignea":
                nuevas_igneas = set(m_igneas)
                nuevas_igneas.remove(pos)
                nueva_bodega.add(("ignea", pos))
                return (pos, bateria - 3, taladro, frozenset(nueva_bodega), frozenset(nuevas_igneas), m_sedimentarias)
            else:
                nuevas_sedimentarias = set(m_sedimentarias)
                nuevas_sedimentarias.remove(pos)
                nueva_bodega.add(("sedimentaria", pos))
                return (pos, bateria - 3, taladro, frozenset(nueva_bodega), m_igneas, frozenset(nuevas_sedimentarias))

        elif tipo == "depositar":
            return (pos, bateria - 1, taladro, frozenset(), m_igneas, m_sedimentarias)

        elif tipo == "moverse":
            return (param, bateria - 1, taladro, bodega, m_igneas, m_sedimentarias)

        elif tipo == "sobremarcha":
            return (param, bateria - 4, taladro, bodega, m_igneas, m_sedimentarias)

        return state

    def is_goal(self, state):
        _, _, _, bodega, m_igneas, m_sedimentarias = state
        return len(m_igneas) == 0 and len(m_sedimentarias) == 0 and len(bodega) == 0

    def cost(self, state, action, state2):
        tipo, _ = action
        if tipo in ("moverse", "sobremarcha"):
            return 1
        elif tipo == "equipar":
            return 3
        elif tipo == "recolectar":
            return 2
        elif tipo == "depositar":
            return len(state[3])  # 1 minuto por cada muestra que había en la bodega antes de vaciar
        elif tipo == "recargar":
            return 4
        return 0

    def heuristic(self, state):
        _, _, _, bodega, m_igneas, m_sedimentarias = state
        cant_restantes = len(m_igneas) + len(m_sedimentarias)
        cant_bodega = len(bodega)
        total_por_procesar = cant_restantes + cant_bodega
        
        if total_por_procesar == 0:
            return 0
            
        # Estimación admisible en minutos independientes del movimiento:
        # Cada muestra restante cuesta al menos 2 min de perforación
        costo_minimo = cant_restantes * 2
        # Cada muestra (en mapa o bodega) cuesta 1 min al ser depositada
        costo_minimo += total_por_procesar
        # Al menos una acción de depositar será requerida
        costo_minimo += 1 
            
        return costo_minimo

def planear_rover(rover_inicio, bateria_inicial, zonas_sombra, muestras_igneas, muestras_sedimentarias):
    problema = RoverProblem(
        rover_inicio,
        bateria_inicial,
        zonas_sombra,
        muestras_igneas,
        muestras_sedimentarias
    )
    
    resultado = astar(problema, graph_search=True)
    
    if resultado:
        # resultado.path() devuelve una lista de tuplas (accion, estado).
        # Usamos una comprensión de lista para extraer solo las acciones válidas.
        return [accion for accion, estado in resultado.path() if accion is not None]
        
    return []
