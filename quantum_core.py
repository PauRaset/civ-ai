import numpy as np
from numpy.fft import fft, ifft, fftfreq

# ==========================
# Núcleo cuántico 1D serio (split-step Fourier)
# ==========================

def _clamp(value, vmin, vmax):
    return max(vmin, min(vmax, value))


def build_spatial_grid(L, N):
    """
    Construye la rejilla 1D en el intervalo [-L/2, L/2).
    Devuelve:
        x: array de posiciones (N,)
        dx: espaciado
    """
    N = int(N)
    if N <= 0:
        raise ValueError("N debe ser positivo")
    L = float(L)
    x = np.linspace(-L / 2.0, L / 2.0, N, endpoint=False)
    dx = x[1] - x[0]
    return x, dx


def build_k_grid(N, dx):
    """
    Construye la rejilla de números de onda k correspondiente a la FFT.
    """
    return 2.0 * np.pi * fftfreq(N, d=dx)


def build_potential_1d(x, pot_cfg):
    """
    Construye un potencial 1D en función de la configuración.
    pot_cfg debe tener al menos la clave "tipo".
    Tipos soportados:
        - "libre": V(x) = 0
        - "pozo": V = 0 en [x_min, x_max], V = V_out fuera
        - "barrera": V = V0 en [x_min, x_max], 0 fuera
        - "armonic": V = 0.5 * k * (x - x0)^2
        - "doble_pozo": V = a * x^4 - b * x^2   (doble pozo simétrico)
    """
    V = np.zeros_like(x, dtype=float)
    if pot_cfg is None:
        return V

    tipo = str(pot_cfg.get("tipo", "libre")).lower()

    if tipo == "libre":
        V[:] = 0.0

    elif tipo == "pozo":
        x_min = float(pot_cfg.get("x_min", -0.5))
        x_max = float(pot_cfg.get("x_max", 0.5))
        V_out = float(pot_cfg.get("V_out", 10.0))
        V[:] = V_out
        mask = (x >= x_min) & (x <= x_max)
        V[mask] = 0.0

    elif tipo == "barrera":
        x_min = float(pot_cfg.get("x_min", -0.5))
        x_max = float(pot_cfg.get("x_max", 0.5))
        V0 = float(pot_cfg.get("V0", 5.0))
        V[:] = 0.0
        mask = (x >= x_min) & (x <= x_max)
        V[mask] = V0

    elif tipo == "armonic" or tipo == "armonico":
        k = float(pot_cfg.get("k", 1.0))
        x0 = float(pot_cfg.get("x0", 0.0))
        V = 0.5 * k * (x - x0) ** 2

    elif tipo == "doble_pozo":
        # Potencial tipo a x^4 - b x^2
        a = float(pot_cfg.get("a", 1.0))
        b = float(pot_cfg.get("b", 5.0))
        V = a * x**4 - b * x**2

    else:
        # Tipo desconocido -> libre
        V[:] = 0.0

    return V


def build_initial_state_1d(x, init_cfg):
    """
    Construye un estado inicial psi(x) y lo normaliza.
    init_cfg:
        tipo:
            - "gauss": gaussiana centrada en x0, sin momento
            - "gauss_momentum": gaussiana con fase oscilante (momento k0)
            - "superposicion": suma de dos gaussianas
    """
    N = x.size
    psi = np.zeros(N, dtype=np.complex128)
    if init_cfg is None:
        # Por defecto, gaussiana centrada en 0
        x0 = 0.0
        sigma = 1.0
        psi = np.exp(-0.5 * ((x - x0) / sigma) ** 2)
    else:
        tipo = str(init_cfg.get("tipo", "gauss")).lower()
        if tipo == "gauss":
            x0 = float(init_cfg.get("x0", 0.0))
            sigma = float(init_cfg.get("sigma", 1.0))
            psi = np.exp(-0.5 * ((x - x0) / sigma) ** 2)

        elif tipo == "gauss_momentum":
            x0 = float(init_cfg.get("x0", -2.0))
            sigma = float(init_cfg.get("sigma", 1.0))
            k0 = float(init_cfg.get("k0", 2.0))
            psi = np.exp(-0.5 * ((x - x0) / sigma) ** 2) * np.exp(1j * k0 * x)

        elif tipo == "superposicion":
            x1 = float(init_cfg.get("x1", -2.0))
            x2 = float(init_cfg.get("x2", 2.0))
            sigma = float(init_cfg.get("sigma", 0.7))
            psi = np.exp(-0.5 * ((x - x1) / sigma) ** 2) + np.exp(
                -0.5 * ((x - x2) / sigma) ** 2
            )

        else:
            # Por defecto, gaussiana centrada en 0
            x0 = 0.0
            sigma = 1.0
            psi = np.exp(-0.5 * ((x - x0) / sigma) ** 2)

    # Normalizar
    dx = x[1] - x[0]
    norm = np.sqrt(np.sum(np.abs(psi) ** 2) * dx)
    if norm > 0:
        psi /= norm
    return psi


def measure_probability_region_1d(psi, x, metrica_cfg):
    """
    Calcula la probabilidad en una región definida por x_min, x_max.
    metrica_cfg:
        {
          "tipo": "prob_region",
          "x_min": float,
          "x_max": float
        }
    Si no se especifica nada, devuelve la probabilidad total (≈1).
    """
    dx = x[1] - x[0]
    prob_density = np.abs(psi) ** 2
    prob_total = float(np.sum(prob_density) * dx)

    if metrica_cfg is None:
        return prob_total, prob_total

    tipo = str(metrica_cfg.get("tipo", "prob_region")).lower()
    if tipo == "prob_region":
        x_min = float(metrica_cfg.get("x_min", x.min()))
        x_max = float(metrica_cfg.get("x_max", x.max()))
        mask = (x >= x_min) & (x <= x_max)
        if not np.any(mask):
            prob_region = 0.0
        else:
            prob_region = float(np.sum(prob_density[mask] * dx))
    else:
        # Métrica desconocida -> prob total
        prob_region = prob_total

    return prob_region, prob_total


def run_schrodinger_1d(config):
    """
    Ejecuta un experimento de Schrödinger 1D usando split-step Fourier.

    config esperado (claves principales):
    {
      "modelo": "schrodinger_1d",
      "L": float,
      "N": int,
      "T": float,
      "dt": float,
      "potencial": {...},
      "estado_inicial": {...},
      "metrica": {...}
    }
    """
    if config is None:
        raise ValueError("config no puede ser None")

    modelo = str(config.get("modelo", "schrodinger_1d")).lower()
    if modelo != "schrodinger_1d":
        raise ValueError(f"Modelo no soportado: {modelo}")

    # Parámetros con límites de seguridad
    L = float(config.get("L", 20.0))
    N = int(config.get("N", 512))
    T = float(config.get("T", 5.0))
    dt = float(config.get("dt", 0.01))

    # Limitar para evitar simulaciones absurdas
    N = _clamp(N, 64, 2048)
    dt = max(1e-4, min(0.05, dt))
    T = max(0.1, min(20.0, T))

    steps = int(round(T / dt))
    MAX_STEPS = 5000
    if steps > MAX_STEPS:
        steps = MAX_STEPS
        T = steps * dt

    # Construir rejilla y potencial
    x, dx = build_spatial_grid(L, N)
    V = build_potential_1d(x, config.get("potencial"))

    # Estado inicial
    psi = build_initial_state_1d(x, config.get("estado_inicial"))

    # Preparar split-step
    k = build_k_grid(N, dx)
    # En unidades adimensionales: H = -1/2 d^2/dx^2 + V
    # Término cinético en espacio de Fourier: exp(-i * k^2 * dt / 2)
    kinetic_phase = np.exp(-0.5j * (k ** 2) * dt)
    potential_phase = np.exp(-1j * V * dt / 2.0)

    # Evolución temporal
    for _ in range(steps):
        # V/2
        psi = potential_phase * psi
        # K
        psi_k = fft(psi)
        psi_k *= kinetic_phase
        psi = ifft(psi_k)
        # V/2
        psi = potential_phase * psi

    # Renormalizar por seguridad
    prob_density = np.abs(psi) ** 2
    norm = np.sqrt(np.sum(prob_density) * dx)
    if norm > 0:
        psi /= norm
        prob_density = np.abs(psi) ** 2

    prob_region, prob_total = measure_probability_region_1d(
        psi, x, config.get("metrica")
    )

    resultados = {
        "modelo": "schrodinger_1d",
        "L": L,
        "N": N,
        "T": T,
        "dt": dt,
        "steps": steps,
        "prob_region": prob_region,
        "prob_total": prob_total,
        "METRICA_CONTROL": prob_region,
    }

    return resultados, x, psi
