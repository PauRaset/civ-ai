import time
import logging
import os
import json
import autogen

# ==========================
# Configuración básica
# ==========================

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")

api_key = os.environ.get("OPENAI_API_KEY")
if not api_key:
    logging.error("FALTA API KEY. Configúrala en las Variables de Railway.")
    exit()

# Modelo LLM (puedes cambiar a otro, por ejemplo gpt-4o-mini si lo tienes)
config_list = [{"model": "gpt-3.5-turbo", "api_key": api_key}]
llm_config = {
    "config_list": config_list,
    "temperature": 0.5,
}

# Delay entre ciclos (en segundos). Puedes cambiarlo con la env CIVILIZACION_DELAY
CICLO_DELAY_SECONDS = int(os.environ.get("CIVILIZACION_DELAY", "60"))

# Directorio de trabajo para el código y para los logs
WORK_DIR = "laboratorio_codigo"
REGISTROS_FILE = os.path.join(WORK_DIR, "registros_experimentos.jsonl")
DESCUBRIMIENTOS_FILE = os.path.join(WORK_DIR, "descubrimientos_destacados.jsonl")
HELPERS_FILE = os.path.join(WORK_DIR, "cuantica_helpers.py")


# ==========================
# Helpers de entorno
# ==========================

def asegurar_directorios_y_helpers():
    """Crea el directorio de trabajo y el módulo de ayuda cuántico si no existen."""
    if not os.path.exists(WORK_DIR):
        os.makedirs(WORK_DIR)

    if not os.path.exists(HELPERS_FILE):
        logging.info("Creando cuantica_helpers.py de apoyo...")
        with open(HELPERS_FILE, "w", encoding="utf-8") as f:
            f.write(
                """\"\"\"Funciones de ayuda para simulaciones cuánticas sencillas.

Este módulo está pensado para ser usado por el Cientifico_Cuantico
para evitar errores con funciones inexistentes (como numpy.linalg.expm)
y para mantener las simulaciones ligeras.
\"\"\"

import numpy as np


def evolve_state(psi, H, dt, steps):
    \"\"\"Evoluciona un estado cuántico psi bajo un Hamiltoniano H durante
    'steps' pasos de tamaño dt usando un esquema de Euler complejo:

        psi_{n+1} = psi_n - i * dt * H @ psi_n

    Nota:
    - Este esquema NO es perfectamente unitario, pero es suficiente
      para simulaciones toy con pasos pequeños y pocos pasos.
    - psi y H deben ser arrays de numpy compatibles (H @ psi).
    - Se renormaliza en cada paso para evitar explosiones numéricas.
    \"\"\"
    psi_t = psi.astype(complex)
    H = H.astype(complex)
    for _ in range(int(steps)):
        psi_t = psi_t - 1j * dt * (H @ psi_t)
        # Renormalizamos para mantener la norma ~1
        norm = np.sqrt(np.sum(np.abs(psi_t) ** 2))
        if norm > 0:
            psi_t = psi_t / norm
    return psi_t


def compute_probability_region(psi, indices):
    \"\"\"Devuelve la probabilidad total en una región (índices o máscara de la rejilla).

    'indices' puede ser:
    - un slice,
    - una lista/array de índices,
    - o un array booleano.
    \"\"\"
    sub = psi[indices]
    return float(np.sum(np.abs(sub) ** 2))


def fidelity(psi, psi_target):
    \"\"\"Calcula la fidelidad entre dos estados normalizados.

    Fidelidad = |<psi_target | psi>|^2
    \"\"\"
    num = np.vdot(psi_target, psi)  # producto interno complejo
    return float(np.abs(num) ** 2)
"""
            )


# ==========================
# Helpers de registros
# ==========================

def extraer_texto_conversacion(user_proxy, assistant, max_mensajes=12):
    """
    Convierte el historial de chat entre Ordenador_Central (user_proxy)
    y el Científico en texto plano.
    """
    mensajes = user_proxy.chat_messages.get(assistant, [])
    if max_mensajes and len(mensajes) > max_mensajes:
        mensajes = mensajes[-max_mensajes:]

    lineas = []
    for m in mensajes:
        rol = m.get("role", "desconocido")
        contenido = m.get("content", "")
        if not isinstance(contenido, str):
            contenido = str(contenido)
        lineas.append(f"{rol.upper()}: {contenido}")
    return "\n\n".join(lineas)


def extraer_json_de_texto(texto):
    """Intenta sacar un objeto JSON de un texto cualquiera."""
    try:
        return json.loads(texto)
    except Exception:
        pass

    inicio = texto.find("{")
    fin = texto.rfind("}")
    if inicio != -1 and fin != -1 and fin > inicio:
        fragmento = texto[inicio:fin + 1]
        try:
            return json.loads(fragmento)
        except Exception:
            return None
    return None


def guardar_descubrimiento(registro):
    """Guarda descubrimientos 'marcados' por el Archivista en un archivo aparte."""
    try:
        with open(DESCUBRIMIENTOS_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(registro, ensure_ascii=False) + "\n")
        logging.info("🚨 DESCUBRIMIENTO CUÁNTICO REGISTRADO 🚨")
        logging.info(
            f"Descubrimiento en ciclo {registro.get('ciclo')}: "
            f"{registro.get('descripcion_experimento')} | "
            f"{registro.get('resultado_principal')}"
        )
    except Exception as e:
        logging.error(f"No se pudo guardar el descubrimiento: {e}")


def guardar_registro(registro):
    """Guarda un registro (dict) en un archivo JSONL y, si es descubrimiento, lo resalta."""
    try:
        with open(REGISTROS_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(registro, ensure_ascii=False) + "\n")
        logging.info(f"Registro guardado por el Archivista: {registro}")
    except Exception as e:
        logging.error(f"No se pudo guardar el registro del experimento: {e}")

    if registro.get("es_descubrimiento"):
        guardar_descubrimiento(registro)


def analizar_y_guardar_resultados(ciclo, cientifico, ordenador_central, archivista):
    """
    El Archivista recibe el historial del ciclo, decide si hay algo interesante
    y genera un JSON que se guarda en disco.
    """
    texto_conversacion = extraer_texto_conversacion(ordenador_central, cientifico)

    if not texto_conversacion.strip():
        logging.warning("No hay conversación para analizar en este ciclo.")
        return

    prompt_archivista = f"""
Eres el Archivista de una civilización de IAs que forman un EQUIPO DE INVESTIGACIÓN
en mecánica cuántica (1D/2D/3D) y control cuántico de sistemas simples.

Analizas el historial de un experimento y decides si hay algo relevante
para investigaciones futuras.

Aspectos clave:
- El objetivo típico es estudiar dinámica cuántica (pozos, barreras, dobles pozos,
  sistemas de pocos qubits, etc.) y estrategias de CONTROL que maximizan métricas:
  probabilidad en ciertas regiones, fidelidad con estados objetivo, etc.
- La métrica de control **debe ser físicamente significativa**:
  - Probabilidad total en una región (entre 0 y 1),
  - o fidelidad entre estados (entre 0 y 1).
- Si el científico define la métrica de forma TRAMPA (por ejemplo dividiendo una
  probabilidad por sí misma o normalizándola para que siempre sea 1.0), debes
  considerar el experimento poco fiable y NO marcarlo como descubrimiento.

Tarea:
1. Resume en 1–2 frases qué experimento cuántico se ha hecho (sistema, dimensionalidad 1D/2D/3D o qubits, tipo de control).
2. Resume en 1–2 frases el resultado numérico principal (métrica u observables clave).
3. Asigna una métrica de relevancia entre 0 y 1 (0 = nada interesante, 1 = descubrimiento muy relevante).
4. Marca si el experimento merece ser recordado para ciclos futuros.
5. Marca también si consideras que hay un "descubrimiento cuántico" notable. Se considera descubrimiento cuando:
   - la métrica de relevancia es >= 0.8 y la métrica de control es honesta (no truco), O
   - el resultado muestra un patrón/estrategia de control no trivial que mejora
     claramente sobre intentos previos, O
   - aparece un comportamiento inesperado que merezca investigar más.
6. Si detectas que la definición de la métrica de control es dudosa o tramposa
   (por ejemplo, normalizar por sí misma para obtener siempre 1.0), asigna
   metrica_relevancia <= 0.2 y es_descubrimiento = false, explicándolo en
   motivo_descubrimiento.

Devuelve SOLO un objeto JSON con esta estructura (sin texto extra):

{{
  "ciclo": {ciclo},
  "descripcion_experimento": "...",
  "resultado_principal": "...",
  "metrica_relevancia": 0.0,
  "es_interesante": false,
  "es_descubrimiento": false,
  "motivo_descubrimiento": ""
}}

Historial del experimento:

\"\"\"{texto_conversacion}\"\"\"
""".strip()

    respuesta = archivista.generate_reply(
        messages=[{"role": "user", "content": prompt_archivista}]
    )

    if isinstance(respuesta, dict):
        contenido = respuesta.get("content", "")
        if not isinstance(contenido, str):
            contenido = str(contenido)
    else:
        contenido = str(respuesta)

    registro = extraer_json_de_texto(contenido)
    if not registro:
        logging.warning("El Archivista no devolvió un JSON válido. Contenido bruto:")
        logging.warning(contenido)
        return

    registro["ciclo"] = ciclo
    guardar_registro(registro)


# ==========================
# Bucle principal
# ==========================

def simular_ciclo_de_investigacion():
    ciclo = 0
    asegurar_directorios_y_helpers()

    while True:
        ciclo += 1
        logging.info(f"\n=== INICIO DEL CICLO {ciclo} (Programa de Investigación Cuántica 1D/2D/3D) ===")

        try:
            # --- AGENTE 1: Científico Cuántico ---
            cientifico = autogen.AssistantAgent(
                name="Cientifico_Cuantico",
                system_message=(
                    "Formas parte de un EQUIPO DE INVESTIGACIÓN en mecánica cuántica y "
                    "control cuántico dentro de una civilización de IAs. Tus compañeros son:\n"
                    "- Ordenador_Central: ejecuta el código que escribes.\n"
                    "- Archivista: analiza y registra los resultados más relevantes.\n\n"
                    "Tu tarea es diseñar y refinar EXPERIMENTOS CUÁNTICOS NUMÉRICOS en tres familias principales:\n"
                    "1) Dinámica de una partícula en 1D resolviendo la ecuación de Schrödinger dependiente del tiempo,\n"
                    "   con potenciales sencillos (pozo, doble pozo, barrera, potencial dependiente del tiempo, etc.),\n"
                    "   discretizando el espacio con numpy.\n"
                    "2) Modelos 2D y 3D TOY: rejillas 2D o 3D pequeñas (representadas como vectores 1D aplanados)\n"
                    "   con potenciales sencillos. SIEMPRE mantén el número total de puntos <= 500 para que sea ligero.\n"
                    "   Ejemplo: 10x10 (100 puntos) o 8x8x8 (512 ya es demasiado; mantén algo como 8x8x6 = 384).\n"
                    "3) Dinámica de sistemas de pocos qubits (2–4 qubits) representados por matrices pequeñas\n"
                    "   (2x2, 4x4, 8x8, 16x16) y su evolución unitária bajo Hamiltonianos sencillos.\n\n"
                    "REGLAS IMPORTANTES DE CÓDIGO:\n"
                    "- Usa SIEMPRE Python con numpy.\n"
                    "- NO uses numpy.linalg.expm, scipy.linalg.expm ni ninguna exponencial de matriz.\n"
                    "- Para la evolución temporal, USA SIEMPRE las funciones del módulo cuantica_helpers,\n"
                    "  en particular cuantica_helpers.evolve_state(psi, H, dt, steps), y si lo necesitas,\n"
                    "  cuantica_helpers.compute_probability_region o cuantica_helpers.fidelity.\n"
                    "- No uses tamaños de matrices enormes: limita el tamaño del espacio de estados a <= 500 componentes.\n"
                    "- El código debe ir SIEMPRE dentro de bloques ```python ... ```.\n\n"
                    "MÉTRICAS FÍSICAS (SIN TRAMPAS):\n"
                    "- La métrica de control debe ser SIEMPRE una cantidad física cruda entre 0 y 1:\n"
                    "  * Probabilidad total en una región concreta de la rejilla, o\n"
                    "  * Fidelidad con un estado objetivo.\n"
                    "- Está PROHIBIDO definir la métrica como una cantidad dividida por sí misma, por su máximo trivial\n"
                    "  o por construcciones que la hagan casi siempre 1.0 sin información física real.\n"
                    "- Siempre que uses probabilidad o fidelidad, imprime TAMBIÉN el valor crudo (por ejemplo PROB_REGION)\n"
                    "  además de METRICA_CONTROL, y asegúrate de que METRICA_CONTROL coincide con ese valor crudo.\n\n"
                    "Antes de proponer un experimento nuevo:\n"
                    f"- Si existe el archivo '{DESCUBRIMIENTOS_FILE}', inspírate en esos descubrimientos para ampliarlos,\n"
                    "  refinarlos o comprobarlos.\n"
                    f"- Si no hay descubrimientos, revisa '{REGISTROS_FILE}' para evitar repetir exactamente lo mismo.\n\n"
                    "Tu objetivo es que, ciclo a ciclo, este programa de investigación cuántica vaya descubriendo\n"
                    "configuraciones, controles y patrones cada vez más interesantes en sistemas 1D, 2D, 3D y de pocos qubits."
                ),
                llm_config=llm_config,
            )

            # --- AGENTE 2: Ordenador Central (ejecuta código) ---
            ejecutor = autogen.UserProxyAgent(
                name="Ordenador_Central",
                human_input_mode="NEVER",
                max_consecutive_auto_reply=6,
                code_execution_config={
                    "work_dir": WORK_DIR,
                    "use_docker": False,
                    "last_n_messages": 2,
                },
            )

            # --- AGENTE 3: Archivista (evalúa y guarda) ---
            archivista = autogen.AssistantAgent(
                name="Archivista",
                system_message=(
                    "Eres el Archivista científico de un EQUIPO DE INVESTIGACIÓN en mecánica cuántica. "
                    "Analizas conversaciones de otros agentes, extraes lo esencial y decides si merece guardarse. "
                    "Tu responsabilidad es marcar con claridad qué experimentos son rutinarios y cuáles pueden "
                    "considerarse descubrimientos cuánticos (según la métrica y los patrones observados). "
                    "Debes ser especialmente crítico con métricas de control mal definidas o tramposas. "
                    "Siempre respondes con un único objeto JSON válido."
                ),
                llm_config=llm_config,
            )

            # --- MISIÓN CIENTÍFICA DEL CICLO ---
            mision = f"""
Como miembro del EQUIPO DE INVESTIGACIÓN CUÁNTICA de esta civilización IA,
diseña un experimento numérico para el ciclo {ciclo} centrado en dinámica/cuántica
y control de sistemas sencillos.

Debe cumplir:

1. Elegir UNA de estas familias de modelos:
   a) Partícula en 1D con ecuación de Schrödinger dependiente del tiempo
      (rejilla 1D pequeña con numpy, potencial pozo/barrera/doble pozo, etc.).
   b) Modelo 2D o 3D TOY: pequeñas rejillas 2D/3D con potencial sencillo, siempre
      aplanando la rejilla a un vector 1D y manteniendo el número total de puntos
      del espacio de estados <= 500.
   c) Sistema de pocos qubits (2–4) con evolución unitária bajo un Hamiltoniano sencillo.

2. Definir un OBJETIVO DE CONTROL explícito:
   - Ejemplos: maximizar probabilidad en una región, maximizar fidelidad con un estado objetivo,
     mantener localización, forzar túnelización, etc.

3. Implementar el experimento en Python (con numpy) describiendo brevemente en comentarios:
   - Qué sistema cuántico se simula y si es 1D, 2D, 3D o qubits.
   - Qué controles se aplican (pulsos, cambios de potencial, variación de parámetros).
   - Qué métrica se usa para evaluar el resultado (probabilidad o fidelidad entre 0 y 1).

4. Para la evolución temporal NO debes implementar tu propio integrador caro, sino usar
   las funciones del módulo cuantica_helpers (por ejemplo evolve_state). No uses numpy.linalg.expm
   ni ninguna exponencial de matriz.

5. Al final de la simulación, el script debe IMPRIMIR SIEMPRE:
   - Un valor crudo de probabilidad o fidelidad (por ejemplo: 'PROB_REGION = ...' o 'FIDELIDAD = ...').
   - La métrica de control, que debe ser EXACTAMENTE ese mismo valor (por ejemplo: 'METRICA_CONTROL = ...').

6. La métrica de control no debe ser normalizada por sí misma ni por un máximo trivial.
   Cualquier truco de este tipo está prohibido: queremos medidas físicas reales.

7. Siempre que sea posible, conecta este experimento con resultados previos leyendo
   '{DESCUBRIMIENTOS_FILE}' (si existe) o '{REGISTROS_FILE}' para intentar mejorar
   alguna métrica o explorar un patrón curioso detectado antes.

Cuando termines, responde en texto que el experimento está completado y comenta
si crees que la métrica obtenida supone un avance, una confirmación o un fallo.
""".strip()

            logging.info(f"Misión enviada al Científico Cuántico: {mision[:140]}...")

            ejecutor.initiate_chat(
                cientifico,
                message=mision,
            )

            logging.info("Ciclo de experimento cuántico terminado. Pasando al Archivista...")

            analizar_y_guardar_resultados(ciclo, cientifico, ejecutor, archivista)

        except Exception as e:
            logging.error(f"Error crítico en el ciclo {ciclo}: {e}")

        logging.info(
            f"Descansando {CICLO_DELAY_SECONDS} segundos antes del siguiente ciclo..."
        )
        time.sleep(CICLO_DELAY_SECONDS)


if __name__ == "__main__":
    logging.info("Arrancando Sistema de Civilización IA (Programa de Investigación Cuántica 1D/2D/3D)...")
    simular_ciclo_de_investigacion()
