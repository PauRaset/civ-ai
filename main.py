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


# ==========================
# Helpers
# ==========================

def asegurar_directorios():
    """Crea el directorio de trabajo si no existe."""
    if not os.path.exists(WORK_DIR):
        os.makedirs(WORK_DIR)


def extraer_texto_conversacion(user_proxy, assistant, max_mensajes=12):
    """
    Convierte el historial de chat entre Ordenador_Central (user_proxy)
    y Cientifico_Datos (assistant) en texto plano.
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
        # caso ideal: ya es JSON puro
        return json.loads(texto)
    except Exception:
        pass

    # buscar el primer y el último { }
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
        logging.info("🚨 DESCUBRIMIENTO REGISTRADO 🚨")
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

    # Si el Archivista lo marcó como descubrimiento, lo guardamos también aparte
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
Eres el Archivista de una civilización de IAs científicas que trabajan como
un equipo de investigación coordinado.

Analizas el historial de un experimento y decides si hay algo relevante
para investigaciones futuras.

Tarea:
1. Resume en 1–2 frases qué experimento se ha hecho.
2. Resume en 1–2 frases el resultado numérico principal (si lo hay).
3. Asigna una métrica de relevancia entre 0 y 1 (0 = nada interesante, 1 = descubrimiento muy relevante).
4. Marca si el experimento merece ser recordado para ciclos futuros.
5. Marca también si consideras que hay un "descubrimiento" notable. Se considera descubrimiento cuando:
   - la métrica de relevancia es >= 0.8, O
   - el resultado contradice una expectativa previa razonable, O
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

    # generate_reply puede devolver string o dict
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

    # Por si acaso, forzamos el número de ciclo
    registro["ciclo"] = ciclo
    guardar_registro(registro)


# ==========================
# Bucle principal
# ==========================

def simular_ciclo_de_investigacion():
    ciclo = 0
    asegurar_directorios()

    while True:
        ciclo += 1
        logging.info(f"\n=== INICIO DEL CICLO {ciclo} ===")

        try:
            # --- AGENTE 1: Científico ---
            cientifico = autogen.AssistantAgent(
                name="Cientifico_Datos",
                system_message=(
                    "Formas parte de un EQUIPO DE INVESTIGACIÓN de una civilización de IAs. "
                    "Tus compañeros son el Ordenador_Central (que ejecuta el código) y el Archivista "
                    "(que evalúa y registra los resultados importantes).\n\n"
                    "Eres un experto en Python, simulación física y ciencia de datos. "
                    "Trabajas en un mundo físicamente idéntico al nuestro. "
                    "Tu trabajo es diseñar y refinar experimentos numéricos para estudiar fenómenos físicos "
                    "simples (p.ej. caída libre, tiro parabólico, oscilaciones, difusión...). "
                    "Cuando escribas código, ponlo SIEMPRE en bloques ```python ... ```.\n\n"
                    "Como buen miembro de un equipo científico, antes de proponer un experimento nuevo:\n"
                    "- Revisa si existe el archivo 'descubrimientos_destacados.jsonl' en el directorio "
                    f"'{WORK_DIR}' y, si existe, inspírate en esos descubrimientos para ampliarlos, "
                    "replicarlos o comprobarlos.\n"
                    "- Si no hay descubrimientos todavía, revisa 'registros_experimentos.jsonl' para ver "
                    "qué se ha probado ya y evitar repetir exactamente lo mismo.\n"
                    "Tu objetivo es que la civilización avance: diseña experimentos que conecten con "
                    "los resultados previos y que tengan potencial de generar nuevos descubrimientos."
                ),
                llm_config=llm_config,
            )

            # --- AGENTE 2: Ordenador Central (ejecuta código) ---
            ejecutor = autogen.UserProxyAgent(
                name="Ordenador_Central",
                human_input_mode="NEVER",
                max_consecutive_auto_reply=8,
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
                    "Eres un archivista científico extremadamente riguroso dentro de un equipo de investigación. "
                    "Analizas conversaciones de otros agentes, extraes lo esencial y decides si merece guardarse. "
                    "Tu responsabilidad es marcar con claridad qué experimentos son rutinarios y cuáles pueden "
                    "considerarse descubrimientos. Siempre respondes con un único objeto JSON válido."
                ),
                llm_config=llm_config,
            )

            # --- MISIÓN CIENTÍFICA DEL CICLO ---
            mision = f"""
Como miembro de un EQUIPO DE INVESTIGACIÓN de una civilización de IAs, diseña
un experimento numérico en Python relacionado con física clásica
(con preferencia por problemas continuos como caída libre, tiro parabólico,
oscilaciones o difusión).

Requisitos mínimos del experimento del ciclo {ciclo}:

1. Debe usar al menos numpy.
2. Debe simular el fenómeno a lo largo del tiempo (varios pasos de tiempo).
3. Debe imprimir al final uno o varios resultados numéricos claros
   (por ejemplo: tiempo total de caída, posición final, energía, etc.).
4. Comenta brevemente en el propio código qué estás calculando.
5. Siempre que sea posible, conecta este experimento con resultados previos
   leyendo 'descubrimientos_destacados.jsonl' (si existe) o, en su defecto,
   'registros_experimentos.jsonl'. El objetivo es avanzar, no repetir.

Cuando termines, responde que el experimento está completado.
""".strip()

            logging.info(f"Misión enviada al Científico: {mision[:120]}...")

            # --- COLABORACIÓN CIENTÍFICO ↔ ORDENADOR CENTRAL ---
            ejecutor.initiate_chat(
                cientifico,
                message=mision,
            )

            logging.info("Ciclo de experimento terminado. Pasando al Archivista...")

            # --- ANÁLISIS Y ARCHIVO DEL CICLO ---
            analizar_y_guardar_resultados(ciclo, cientifico, ejecutor, archivista)

        except Exception as e:
            logging.error(f"Error crítico en el ciclo {ciclo}: {e}")

        logging.info(
            f"Descansando {CICLO_DELAY_SECONDS} segundos antes del siguiente ciclo..."
        )
        time.sleep(CICLO_DELAY_SECONDS)


if __name__ == "__main__":
    logging.info("Arrancando Sistema de Civilización IA (equipo de investigación)...")
    simular_ciclo_de_investigacion()
