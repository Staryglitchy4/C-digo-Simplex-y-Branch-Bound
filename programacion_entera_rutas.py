"""
=============================================================
  OPTIMIZACIÓN DE RUTAS — PROGRAMACIÓN ENTERA (ILP)
=============================================================
Problema: Minimizar el costo de combustible (diesel) de un
autobús Mercedes-Benz MBO asignando una ruta a cada horario.

Variables BINARIAS:
  x[h][r] ∈ {0, 1}
  h ∈ {0..5} → 07:25, 14:10, 15:00, 15:15, 17:10, 19:25
  r ∈ {0..2} → Blvd. Campestre, Pº Insurgentes, J.A. de Torres

Función objetivo:
  Min Z = Σ_h Σ_r C[h][r] * x[h][r]

Restricciones:
  (1) Σ_r x[h][r] = 1   ∀h   (una ruta por horario)
  (2) Σ_r T[h][r]*x[h][r] ≤ 20  ∀h  (tiempo máximo 20 min)
  (3) x[h][r] ∈ {0, 1}
=============================================================
"""

# ── Dependencias ──────────────────────────────────────────────
try:
    from scipy.optimize import milp, LinearConstraint, Bounds
    USAR_MILP = True
except ImportError:
    USAR_MILP = False

import numpy as np

# ─────────────────────────────────────────
# 1. DATOS DEL PROBLEMA
# ─────────────────────────────────────────

HORARIOS = ["07:25 (Ida)", "14:10 (Reg)", "15:00 (Reg)",
            "15:15 (Ida)", "17:10 (Reg)", "19:25 (Reg)"]

RUTAS = ["Blvd. Campestre", "Pº Insurgentes", "J.A. de Torres"]

# Costos C[h][r] en pesos
COSTOS = [
    [42.26, 46.22, 44.71],
    [43.61, 48.43, 42.92],
    [41.09, 45.40, 42.92],
    [44.78, 48.24, 47.23],
    [40.09, 47.42, 40.40],
    [41.60, 49.44, 42.92],
]

# Tiempos T[h][r] en minutos
TIEMPOS = [
    [13.5, 16.0, 13.5],
    [19.0, 19.0, 15.5],
    [16.5, 16.0, 15.5],
    [16.0, 18.0, 16.0],
    [15.5, 18.0, 13.0],
    [17.0, 20.0, 15.5],
]

TIEMPO_MAX = 20
N_HORARIOS = 6
N_RUTAS    = 3
N_VARS     = N_HORARIOS * N_RUTAS  # 18 variables binarias

def idx(h, r):
    return h * N_RUTAS + r

c = np.array([COSTOS[h][r] for h in range(N_HORARIOS) for r in range(N_RUTAS)],
             dtype=float)

def resolver_con_milp():
    from scipy.optimize import milp, LinearConstraint, Bounds
    from scipy.sparse import lil_matrix

    print("=" * 62)
    print("  PROGRAMACIÓN ENTERA — scipy.optimize.milp (HiGHS B&B)")
    print("=" * 62)

    # Matriz de restricciones A (12 filas × 18 cols)
    A = lil_matrix((2 * N_HORARIOS, N_VARS))

    # Restricciones de asignación: Σ_r x[h][r] = 1
    for h in range(N_HORARIOS):
        for r in range(N_RUTAS):
            A[h, idx(h, r)] = 1.0

    # Restricciones de tiempo: Σ_r T[h][r]*x[h][r] ≤ 20
    for h in range(N_HORARIOS):
        for r in range(N_RUTAS):
            A[N_HORARIOS + h, idx(h, r)] = TIEMPOS[h][r]

    A = A.tocsc()

    # Límites inferior y superior de cada restricción
    lb = np.concatenate([np.ones(N_HORARIOS),          # = 1
                         np.full(N_HORARIOS, -np.inf)]) # ≤ 20
    ub = np.concatenate([np.ones(N_HORARIOS),           # = 1
                         np.full(N_HORARIOS, float(TIEMPO_MAX))])

    constraints = LinearConstraint(A, lb, ub)

    # Variables enteras: todas las 18 (binarias → bounds 0/1)
    integrality = np.ones(N_VARS)   # 1 = variable entera
    bounds      = Bounds(lb=0.0, ub=1.0)

    res = milp(c, constraints=constraints,
               integrality=integrality, bounds=bounds)

    return res.x.reshape(N_HORARIOS, N_RUTAS), res.fun, res.message, res.status


# ─────────────────────────────────────────────────────────────
# RANCH & BOUND MANUAL (puro Python/NumPy)
# ─────────────────────────────────────────────────────────────

def resolver_relajacion(fijadas: dict):
    #Resuelve la relajación lineal con algunas variables fijadas.
    from scipy.optimize import linprog

    bounds = []
    for i in range(N_VARS):
        if i in fijadas:
            v = float(fijadas[i])
            bounds.append((v, v))
        else:
            bounds.append((0.0, 1.0))

    # Restricciones de igualdad
    A_eq = np.zeros((N_HORARIOS, N_VARS))
    b_eq = np.ones(N_HORARIOS)
    for h in range(N_HORARIOS):
        for r in range(N_RUTAS):
            A_eq[h, idx(h, r)] = 1.0

    # Restricciones de tiempo
    A_ub = np.zeros((N_HORARIOS, N_VARS))
    b_ub = np.full(N_HORARIOS, float(TIEMPO_MAX))
    for h in range(N_HORARIOS):
        for r in range(N_RUTAS):
            A_ub[h, idx(h, r)] = TIEMPOS[h][r]

    res = linprog(c, A_ub=A_ub, b_ub=b_ub,
                  A_eq=A_eq, b_eq=b_eq,
                  bounds=bounds, method="highs",
                  options={"disp": False})

    if res.status == 0:
        return res.fun, res.x
    return None, None


def branch_and_bound():
    mejor_val = float("inf")
    mejor_sol = None
    log        = []

    # Cola de nodos: (cota_inferior, fijadas_dict)
    # Usamos una lista ordenada manualmente (pequeño problema → ok)
    cola = []

    val0, x0 = resolver_relajacion({})
    if val0 is None:
        return None, None, []

    cola.append((val0, {}))
    nodo_id  = 0
    podados  = 0

    while cola:
        # Seleccionar nodo con menor cota (Best-First)
        cola.sort(key=lambda t: t[0])
        cota, fijadas = cola.pop(0)
        nodo_id += 1

        # Poda por cota
        if cota >= mejor_val - 1e-6:
            podados += 1
            log.append({"nodo": nodo_id, "accion": "PODADO",
                        "cota": cota, "mejor": mejor_val})
            continue

        # Resolver relajación en este nodo
        val, x = resolver_relajacion(fijadas)
        if val is None:
            log.append({"nodo": nodo_id, "accion": "INFACTIBLE",
                        "cota": cota, "mejor": mejor_val})
            continue

        # Buscar variable fraccionaria para hacer branch
        var_branch = None
        for i in range(N_VARS):
            if i not in fijadas and 0.01 < x[i] < 0.99:
                var_branch = i
                break

        if var_branch is None:
            # Solución entera encontrada
            if val < mejor_val:
                mejor_val = val
                mejor_sol = x.copy()
                log.append({"nodo": nodo_id, "accion": "NUEVA_MEJOR",
                            "cota": val, "mejor": mejor_val})
            else:
                log.append({"nodo": nodo_id, "accion": "ENTERA_SUBOPTIMA",
                            "cota": val, "mejor": mejor_val})
            continue

        # Ramificar sobre var_branch
        log.append({"nodo": nodo_id, "accion": f"BRANCH x{var_branch}",
                    "cota": val, "mejor": mejor_val})

        for valor_fijo in [0, 1]:
            nuevas_fijadas = dict(fijadas)
            nuevas_fijadas[var_branch] = valor_fijo

            val_hijo, x_hijo = resolver_relajacion(nuevas_fijadas)
            if val_hijo is not None and val_hijo < mejor_val:
                cola.append((val_hijo, nuevas_fijadas))

    return mejor_val, mejor_sol, log, podados

def mostrar_resultados(x_mat, costo_total, titulo=""):
    print(f"\n{titulo}")
    print(f"  Costo total óptimo: ${costo_total:.4f}")
    print()
    print(f"  {'Horario':<18} {'Ruta Elegida':<22} {'Costo ($)':>10} {'Tiempo (min)':>14}  {'Estado':>10}")
    print("  " + "-" * 80)

    for h in range(N_HORARIOS):
        r_opt = int(np.argmax(x_mat[h]))
        c_h   = COSTOS[h][r_opt]
        t_h   = TIEMPOS[h][r_opt]
        ok    = "✓ OK" if t_h <= TIEMPO_MAX else "✗ VIOLA"
        print(f"  {HORARIOS[h]:<18} {RUTAS[r_opt]:<22} {c_h:>10.2f} {t_h:>14.1f}  {ok:>10}")

    print("  " + "-" * 80)
    print(f"  {'TOTAL':<18} {'':<22} {costo_total:>10.2f}")

    # Comparar con greedy (siempre la ruta más barata sin restricción de tiempo)
    costo_greedy = sum(min(COSTOS[h]) for h in range(N_HORARIOS))
    ahorro = costo_greedy - costo_total
    print(f"\n  Costo si siempre se elige la ruta más barata (sin filtrar): ${costo_greedy:.2f}")
    print(f"  Coincide con el óptimo: {'Sí ✓' if abs(ahorro) < 0.01 else 'No — diferencia: $' + f'{abs(ahorro):.2f}'}")


# ── Ejecutar solver principal ──────────────────────────────────
print("=" * 62)
print("  PROGRAMACIÓN ENTERA BINARIA — Optimización de Rutas")
print("=" * 62)

if USAR_MILP:
    x_mat, costo_opt, msg, status = resolver_con_milp()
    mostrar_resultados(
        x_mat, costo_opt,
        "► Resultados — HiGHS Mixed-Integer Linear Programming"
    )
    print(f"\n  Solver status: {msg}")
else:
    print("  scipy.optimize.milp no disponible. Usando B&B manual...")

# ── Branch & Bound manual (siempre se ejecuta como validación) ──
print("\n" + "=" * 62)
print("  BRANCH & BOUND MANUAL — Implementación didáctica")
print("=" * 62)

bb_val, bb_sol, bb_log, podados = branch_and_bound()

if bb_sol is not None:
    x_mat_bb = bb_sol.reshape(N_HORARIOS, N_RUTAS)
    mostrar_resultados(
        x_mat_bb, bb_val,
        "► Resultados — Branch & Bound manual"
    )

    print(f"\n{'─'*62}")
    print("  Registro del árbol Branch & Bound")
    print(f"{'─'*62}")
    print(f"  {'Nodo':>5} {'Acción':<22} {'Cota LB':>10} {'Mejor UB':>10}")
    print("  " + "-" * 54)
    for entry in bb_log:
        print(f"  {entry['nodo']:>5} {entry['accion']:<22} "
              f"{entry['cota']:>10.4f} {entry['mejor']:>10.4f}")
    print(f"\n  Total nodos explorados : {len(bb_log)}")
    print(f"  Nodos podados          : {podados}")
    print(f"  Valor óptimo           : ${bb_val:.4f}")

print("\n" + "=" * 62)
print("  COMPARACIÓN DE COSTOS POR HORARIO Y RUTA")
print("=" * 62)
print(f"\n  {'Horario':<18} {'Campestre':>12} {'Insurgentes':>12} {'J.A. Torres':>12}  Óptimo")
print("  " + "-" * 70)
for h in range(N_HORARIOS):
    r_opt = int(np.argmax(bb_sol.reshape(N_HORARIOS, N_RUTAS)[h]))
    costos_str = " ".join(
        f"{'▶' if r == r_opt else ' '}{COSTOS[h][r]:>9.2f}{'◀' if r == r_opt else ' '}"
        for r in range(N_RUTAS)
    )
    print(f"  {HORARIOS[h]:<18} {costos_str}  R{r_opt+1}: {RUTAS[r_opt]}")

print()