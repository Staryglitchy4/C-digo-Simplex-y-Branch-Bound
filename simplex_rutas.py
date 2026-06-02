"""
=============================================================
  OPTIMIZACIÓN DE RUTAS - MÉTODO SIMPLEX (Relajación Lineal)
=============================================================
Problema: Minimizar el costo de combustible (diesel) de un
autobús Mercedes-Benz MBO asignando una ruta óptima a cada
uno de los 6 horarios.

Variables de decisión:
  x[h][r] = 1 si se elige la ruta r en el horario h, 0 si no.
  h ∈ {0..5} → horarios: 07:25, 14:10, 15:00, 15:15, 17:10, 19:25
  r ∈ {0..2} → rutas: Blvd. Campestre, Pº Insurgentes, J.A. de Torres
=============================================================
"""

import numpy as np
from scipy.optimize import linprog

# ─────────────────────────────────────────
# 1. DATOS DEL PROBLEMA
# ─────────────────────────────────────────

HORARIOS = ["07:25 (Ida)", "14:10 (Reg)", "15:00 (Reg)",
            "15:15 (Ida)", "17:10 (Reg)", "19:25 (Reg)"]

RUTAS = ["Blvd. Campestre", "Pº Insurgentes", "J.A. de Torres"]

# Costos C[h][r] en pesos (incluyen combustible + ralentí)
# Fuente: tabla del documento
COSTOS = [
    [42.26, 46.22, 44.71],  # 07:25 Ida   (mañana, 50 pax, 2.6 km/L)
    [43.61, 48.43, 42.92],  # 14:10 Reg   (tarde,  32 pax, 2.8 km/L)
    [41.09, 45.40, 42.92],  # 15:00 Reg
    [44.78, 48.24, 47.23],  # 15:15 Ida
    [40.09, 47.42, 40.40],  # 17:10 Reg
    [41.60, 49.44, 42.92],  # 19:25 Reg   (noche,  10 pax, 3.1 km/L)
]

# Tiempos T[h][r] en minutos (con tráfico)
TIEMPOS = [
    [13.5, 16.0, 13.5],  # 07:25
    [19.0, 19.0, 15.5],  # 14:10
    [16.5, 16.0, 15.5],  # 15:00
    [16.0, 18.0, 16.0],  # 15:15
    [15.5, 18.0, 13.0],  # 17:10
    [17.0, 20.0, 15.5],  # 19:25
]

TIEMPO_MAX = 20  # minutos — límite por viaje
N_HORARIOS = 6
N_RUTAS    = 3
N_VARS     = N_HORARIOS * N_RUTAS  # 18 variables en total

def idx(h, r):
    """Índice lineal de la variable x[h][r]."""
    return h * N_RUTAS + r

# Vector de costos (función objetivo)
c = np.array([COSTOS[h][r] for h in range(N_HORARIOS) for r in range(N_RUTAS)],
             dtype=float)

# ── Restricciones de igualdad: exactamente una ruta por horario ──
# Σ_r x[h][r] = 1  para cada h
A_eq = np.zeros((N_HORARIOS, N_VARS))
b_eq = np.ones(N_HORARIOS)

for h in range(N_HORARIOS):
    for r in range(N_RUTAS):
        A_eq[h, idx(h, r)] = 1.0

# ── Restricciones de desigualdad: tiempo ≤ 20 min ──
# Σ_r T[h][r] * x[h][r] ≤ 20  para cada h
A_ub = np.zeros((N_HORARIOS, N_VARS))
b_ub = np.full(N_HORARIOS, float(TIEMPO_MAX))

for h in range(N_HORARIOS):
    for r in range(N_RUTAS):
        A_ub[h, idx(h, r)] = TIEMPOS[h][r]

# ── Bounds: relajación lineal → 0 ≤ x ≤ 1 ──
bounds = [(0.0, 1.0)] * N_VARS

resultado = linprog(
    c,
    A_ub=A_ub, b_ub=b_ub,
    A_eq=A_eq, b_eq=b_eq,
    bounds=bounds,
    method="highs",          # HiGHS usa internamente Simplex / Dual
    options={"disp": False}
)

if resultado.status != 0:
    print(f"\n⚠  El solver no encontró solución óptima.")
    print(f"   Mensaje: {resultado.message}")
    exit()

x_sol = resultado.x.reshape(N_HORARIOS, N_RUTAS)

print(f"\n{'─'*60}")
print(f"  Valor óptimo de la función objetivo: ${resultado.fun:.4f}")
print(f"{'─'*60}\n")

print(f"{'Horario':<18} {'Ruta Elegida':<22} {'Costo ($)':>10} {'Tiempo (min)':>14}")
print("-" * 68)

costo_total  = 0.0
tiempo_total = 0.0

for h in range(N_HORARIOS):
    fila = x_sol[h]

    # Detectar si la solución es fraccionaria (no entera)
    es_fraccionaria = any(0.01 < v < 0.99 for v in fila)

    if es_fraccionaria:
        # Mostramos la mezcla (situación rara en este problema bien estructurado)
        desc = "FRACCIONARIA:"
        for r in range(N_RUTAS):
            if fila[r] > 0.001:
                desc += f" {fila[r]:.2f}×R{r+1}"
        costo_h  = sum(fila[r] * COSTOS[h][r]  for r in range(N_RUTAS))
        tiempo_h = sum(fila[r] * TIEMPOS[h][r] for r in range(N_RUTAS))
        print(f"{HORARIOS[h]:<18} {desc:<22} {costo_h:>10.2f} {tiempo_h:>14.1f}")
    else:
        # Solución entera — comportamiento esperado
        r_opt    = int(np.argmax(fila))
        costo_h  = COSTOS[h][r_opt]
        tiempo_h = TIEMPOS[h][r_opt]
        print(f"{HORARIOS[h]:<18} {RUTAS[r_opt]:<22} {costo_h:>10.2f} {tiempo_h:>14.1f}")

    costo_total  += sum(fila[r] * COSTOS[h][r]  for r in range(N_RUTAS))
    tiempo_total += sum(fila[r] * TIEMPOS[h][r] for r in range(N_RUTAS))

print("-" * 68)
print(f"{'TOTAL':<18} {'':<22} {costo_total:>10.2f} {tiempo_total:>14.1f}")

# ─── Verificación de restricciones ───
print(f"\n{'─'*60}")
print("  Verificación de restricciones")
print(f"{'─'*60}")
print(f"  Restricciones de asignación (suma = 1 por horario):")
for h in range(N_HORARIOS):
    s = sum(x_sol[h])
    ok = "✓" if abs(s - 1.0) < 1e-6 else "✗"
    print(f"    {HORARIOS[h]}: suma = {s:.6f}  {ok}")

print(f"\n  Restricciones de tiempo (≤ {TIEMPO_MAX} min):")
for h in range(N_HORARIOS):
    t = sum(x_sol[h][r] * TIEMPOS[h][r] for r in range(N_RUTAS))
    ok = "✓" if t <= TIEMPO_MAX + 1e-6 else "✗"
    print(f"    {HORARIOS[h]}: {t:.2f} min  {ok}")


# Para las restricciones de igualdad las convertimos a ≤ y ≥
# Mostramos sólo las 6 filas de asignación para legibilidad
header = [f"x{h+1}{r+1}" for h in range(N_HORARIOS) for r in range(N_RUTAS)]
header += [f"s{h+1}" for h in range(N_HORARIOS)]
header += ["RHS"]

print("  Variables: " + " | ".join(header))
print("  " + "-" * (len(" | ".join(header)) + 2))

for h in range(N_HORARIOS):
    fila = [0.0] * N_VARS
    for r in range(N_RUTAS):
        fila[idx(h, r)] = 1.0
    # Variables de holgura (identidad para restricciones de asignación)
    slack = [1.0 if i == h else 0.0 for i in range(N_HORARIOS)]
    rhs   = 1.0
    row_str = " | ".join(f"{v:4.0f}" for v in fila + slack + [rhs])
    print(f"  h={h+1}: {row_str}")