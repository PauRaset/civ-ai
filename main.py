import time
import logging
import os
import json
import autogen

# ==========================
# Configuración básica
# ==========================

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

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
    'steps' pasos de tamaño dt usando un esquema de Euler complejo simple:

        psi_{n+1} = psi_n - i * dt * H @ psi_n

    Nota:
    - Este esquema NO es perfectamente unitario, pero es suficiente
      para simulaciones toy con pasos pequeños y pocos pasos.
    - psi y H deben ser arrays de numpy compatibles (H @ psi).
    - Es responsabilidad del usuario renormalizar psi si es necesario.
    \"\"\"
    psi_t = psi.astype(complex)
    H = H.astype(complex)
    for _ in range(steps):
        psi_t = psi_t - 1j * dt * (H @ psi_t)
        # Renormalizamos ligeramente para evitar que se dispare numéricamente
        norm = np.sqrt(np.sum(np.abs(psi_t) ** 2))
        if norm > 0:
            psi_t = psi_t / norm
    return psi_t


def compute_probability_region(psi, indices):
    \"\"\"Devuelve la probabilidad total en una región (índices de la rejilla).\"\"\"
    sub = psi[indices]
    return float(np.sum(np.abs(sub) ** 2))


def fidelity(psi, psi_target):
    \"\"\"Calcula una fidelidad sencilla entre dos estados normalizados.\"\"\"
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
en mecánica cuántica y control cuántico de sistemas simples.

Analizas el historial de un experimento y decides si hay algo relevante
para investigaciones futuras.

Aspectos a tener en cuenta:
- El objetivo típico de los experimentos es estudiar dinámica cuántica en 1D
  (ecuación de Schrödinger con potenciales sencillos) o en sistemas de pocos qubits
  (2–4 qubits) y buscar estrategias de CONTROL que maximicen alguna métrica:
  probabilidad de encontrar la partícula en cierta región, fidelidad de un estado
  objetivo, coherencia, etc.
- Un experimento es más relevante cuanto más claramente mejora alguna métrica de
  control o revela un patrón/cuasi-regla interesante (p.ej. pauta en parámetros,
  interferencias inesperadas, comportamiento no trivial).

Tarea:
1. Resume en 1–2 frases qué experimento cuántico se ha hecho (sistema, potencial/qubits, tipo de control).
2. Resume en 1–2 frases el resultado numérico principal (métrica u observables clave).
3. Asigna una métrica de relevancia entre 0 y 1 (0 = nada interesante, 1 = descubrimiento muy relevante).
4. Marca si el experimento merece ser recordado para ciclos futuros.
5. Marca también si consideras que hay un "descubrimiento cuántico" notable. Se considera descubrimiento cuando:
   - la métrica de relevancia es >= 0.8, O
   - el resultado muestra un patrón/estrategia de control no trivial que mejora
     claramente sobre intentos previos, O
   - aparece un comportamiento inesperado que merezca investigar más.
6. Si es un descubrimiento, explica brevemente el motivo.

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
        logging.info(f"\n=== INICIO DEL CICLO {ciclo} (Programa de Investigación Cuántica) ===")

        try:
            # --- AGENTE 1: Científico Cuántico ---
            cientifico = autogen.AssistantAgent(
                name="Cientifico_Cuantico",
                system_message=(
                    "Formas parte de un EQUIPO DE INVESTIGACIÓN en mecánica cuántica y "
                    "control cuántico dentro de una civilización de IAs. Tus compañeros son:\n"
                    "- Ordenador_Central: ejecuta el código que escribes.\n"
                    "- Archivista: analiza y registra los resultados más relevantes.\n\n"
                    "Tu tarea es diseñar y refinar EXPERIMENTOS CUÁNTICOS NUMÉRICOS en dos familias principales:\n"
                    "1) Dinámica de una partícula en 1D resolviendo la ecuación de Schrödinger dependiente del tiempo\n"
                    "   para potenciales sencillos (pozo, doble pozo, barrera, potencial dependiente del tiempo, etc.),\n"
                    "   discretizando el espacio con numpy.\n"
                    "2) Dinámica de sistemas de pocos qubits (2–4 qubits) representados por matrices pequeñas\n"
                    "   (2x2, 4x4, 8x8, 16x16) y su evolución unitária bajo Hamiltonianos sencillos.\n\n"
                    "REGLAS IMPORTANTES DE CÓDIGO:\n"
                    "- Usa SIEMPRE Python con numpy.\n"
                    "- NO uses numpy.linalg.expm, scipy.linalg.expm ni ninguna función expm de matriz.\n"
                    f"- Para la evolución temporal, USA SIEMPRE las funciones del módulo '{os.path.basename(HELPERS_FILE)}',\n"
                    "  en particular cuantica_helpers.evolve_state(psi, H, dt, steps), y si lo necesitas\n"
                    "  cuantica_helpers.compute_probability_region o cuantica_helpers.fidelity.\n"
                    "- No uses tamaños de matrices enormes: limita N (número de puntos en 1D) a algo como 100–300,\n"
                    "  y el número de pasos de tiempo a algo como 100–500 para que las simulaciones sean ligeras.\n"
                    "- El código debe ir SIEMPRE dentro de bloques ```python ... ```.\n\n"
                    "Antes de proponer un experimento nuevo:\n"
                    f"- Si existe el archivo '{DESCUBRIMIENTOS_FILE}', inspírate en esos descubrimientos para ampliarlos,\n"
                    "  refinarlos o comprobarlos.\n"
                    f"- Si no hay descubrimientos, revisa '{REGISTROS_FILE}' para evitar repetir exactamente lo mismo.\n\n"
                    "Define SIEMPRE una métrica numérica entre 0 y 1 que mida el éxito del control\n"
                    "(por ejemplo, probabilidad en una región o fidelidad con un estado objetivo) y haz que el script\n"
                    "la imprima con claridad (por ejemplo: 'METRICA_CONTROL = 0.87').\n"
                    "Tu objetivo es que, ciclo a ciclo, este programa de investigación cuántica vaya descubriendo\n"
                    "configuraciones, controles y patrones cada vez más interesantes."
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

1. Elegir UNA de estas dos familias de modelos:
   a) Partícula en 1D con ecuación de Schrödinger dependiente del tiempo
      (discretizando el espacio en una rejilla 1D con numpy, usando un potencial sencillo
      como pozo, doble pozo, barrera, potencial escalón, etc.), o
   b) Sistema de pocos qubits (2–4) con evolución unitária bajo un Hamiltoniano sencillo.

2. Definir un OBJETIVO DE CONTROL explícito:
   - Ejemplos: maximizar probabilidad en una región, maximizar fidelidad con un estado objetivo,
     mantener localización, forzar túnelización, etc.

3. Implementar el experimento en Python (con numpy) describiendo brevemente en comentarios:
   - Qué sistema cuántico se simula.
   - Qué controles se aplican (pulsos, cambios de potencial, variación de parámetros).
   - Qué métrica se usa para evaluar el resultado (entre 0 y 1).

4. Para la evolución temporal NO debes implementar tu propio integrador caro, sino usar
   las funciones del módulo cuantica_helpers (por ejemplo evolve_state). No uses numpy.linalg.expm
   ni ninguna exponencial de matriz.

5. Al final de la simulación, el script debe IMPRIMIR:
   - La métrica de control (por ejemplo: 'METRICA_CONTROL = 0.87').
   - Un breve resumen de lo que significa ese valor (en texto).

6. Siempre que sea posible, conecta este experimento con resultados previos leyendo
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
    logging.info("Arrancando Sistema de Civilización IA (Programa de Investigación Cuántica)...")
    simular_ciclo_de_investigacion()
