import time
import logging
import os
import json

import autogen
import quantum_core

# ==========================
# Configuración básica
# ==========================

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")

api_key = os.environ.get("OPENAI_API_KEY")
if not api_key:
    logging.error("FALTA API KEY. Configúrala en las Variables de Railway.")
    raise SystemExit(1)

config_list = [{"model": "gpt-3.5-turbo", "api_key": api_key}]
llm_config = {
    "config_list": config_list,
    "temperature": 0.4,  # un poco más conservador para configuraciones
}

CICLO_DELAY_SECONDS = int(os.environ.get("CIVILIZACION_DELAY", "60"))

WORK_DIR = "laboratorio_codigo"
REGISTROS_FILE = os.path.join(WORK_DIR, "registros_experimentos.jsonl")
DESCUBRIMIENTOS_FILE = os.path.join(WORK_DIR, "descubrimientos_destacados.jsonl")


# ==========================
# Helpers de entorno y ficheros
# ==========================

def asegurar_directorios():
    if not os.path.exists(WORK_DIR):
        os.makedirs(WORK_DIR)


def leer_ultimos_registros(max_lineas=5):
    """
    Devuelve las últimas 'max_lineas' entradas del archivo de registros
    como una cadena de texto para dar contexto al Científico.
    """
    if not os.path.exists(REGISTROS_FILE):
        return ""

    try:
        with open(REGISTROS_FILE, "r", encoding="utf-8") as f:
            lineas = f.readlines()
    except Exception:
        return ""

    if not lineas:
        return ""

    ultimas = lineas[-max_lineas:]
    # Evitamos que sea gigante
    texto = "".join(ultimas)
    if len(texto) > 4000:
        texto = texto[-4000:]
    return texto


def extraer_json_de_texto(texto):
    """
    Intenta extraer un JSON de una respuesta de texto.
    """
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


def guardar_registro_completo(registro):
    """
    Guarda un registro completo (config + resultados + evaluación) en JSONL.
    """
    try:
        with open(REGISTROS_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(registro, ensure_ascii=False) + "\n")
        logging.info("Registro de experimento guardado.")
    except Exception as e:
        logging.error(f"No se pudo guardar el registro del experimento: {e}")


def guardar_descubrimiento(registro):
    """
    Guarda un registro marcado como descubrimiento en un archivo aparte.
    """
    try:
        with open(DESCUBRIMIENTOS_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(registro, ensure_ascii=False) + "\n")
        logging.info("🚨 DESCUBRIMIENTO CUÁNTICO REGISTRADO 🚨")
    except Exception as e:
        logging.error(f"No se pudo guardar el descubrimiento: {e}")


# ==========================
# Bucle principal de la civilización
# ==========================

def simular_ciclo_de_investigacion():
    ciclo = 0
    asegurar_directorios()

    while True:
        ciclo += 1
        logging.info(
            f"\n=== INICIO DEL CICLO {ciclo} (Laboratorio Cuántico 1D serio) ==="
        )

        try:
            # ---------- 1. CIENTÍFICO: PROPONE CONFIGURACIÓN ----------
            cientifico = autogen.AssistantAgent(
                name="Cientifico_Cuantico",
                system_message=(
                    "Eres el científico principal de un LABORATORIO CUÁNTICO 1D serio. "
                    "Tu trabajo NO es escribir código, sino diseñar EXPERIMENTOS BIEN DEFINIDOS "
                    "en mecánica cuántica 1D que luego serán simulados por un núcleo numérico fiable.\n\n"
                    "SIEMPRE debes responder con UN ÚNICO OBJETO JSON VÁLIDO, sin texto extra, "
                    "con el siguiente esquema (ejemplo):\n\n"
                    "{\n"
                    '  "modelo": "schrodinger_1d",\n'
                    '  "L": 20.0,\n'
                    '  "N": 512,\n'
                    '  "T": 5.0,\n'
                    '  "dt": 0.01,\n'
                    '  "potencial": {\n'
                    '    "tipo": "doble_pozo",\n'
                    '    "a": 1.0,\n'
                    '    "b": 5.0\n'
                    "  },\n"
                    '  "estado_inicial": {\n'
                    '    "tipo": "gauss_momentum",\n'
                    '    "x0": -4.0,\n'
                    '    "sigma": 0.7,\n'
                    '    "k0": 2.0\n'
                    "  },\n"
                    '  "metrica": {\n'
                    '    "tipo": "prob_region",\n'
                    '    "x_min": 0.0,\n'
                    '    "x_max": 5.0\n'
                    "  }\n"
                    "}\n\n"
                    "CONDICIONES:\n"
                    "- Usa SIEMPRE \"modelo\": \"schrodinger_1d\" (de momento solo está implementado ese).\n"
                    "- Elige L en el rango [10, 40].\n"
                    "- Elige N en el rango [128, 1024].\n"
                    "- Elige T en el rango [0.5, 10.0].\n"
                    "- Elige dt en el rango [0.001, 0.05].\n\n"
                    "POTENCIALES SOPORTADOS (potencial.tipo):\n"
                    "- \"libre\": V(x) = 0.\n"
                    "- \"pozo\": V = 0 en [x_min, x_max], V = V_out fuera. Claves: x_min, x_max, V_out.\n"
                    "- \"barrera\": V = V0 en [x_min, x_max], 0 fuera. Claves: x_min, x_max, V0.\n"
                    "- \"armonic\": V = 0.5 * k * (x - x0)^2. Claves: k, x0.\n"
                    "- \"doble_pozo\": V = a * x^4 - b * x^2. Claves: a, b.\n\n"
                    "ESTADOS INICIALES SOPORTADOS (estado_inicial.tipo):\n"
                    "- \"gauss\": gaussiana sin momento. Claves: x0, sigma.\n"
                    "- \"gauss_momentum\": gaussiana con momento inicial. Claves: x0, sigma, k0.\n"
                    "- \"superposicion\": suma de dos gaussianas. Claves: x1, x2, sigma.\n\n"
                    "MÉTRICA (metrica.tipo):\n"
                    "- Usa siempre \"prob_region\" con x_min y x_max. La métrica de control será literalmente\n"
                    "  la probabilidad total en esa región (entre 0 y 1). No inventes otros tipos ahora.\n\n"
                    "Tu objetivo como científico no es trivializar la métrica (no la coloques siempre donde\n"
                    "ya sabes que la partícula estará), sino proponer configuraciones interesantes que exploren\n"
                    "túnel, localización, interferencias, captura en pozos, etc., IDEALMENTE mejorando o\n"
                    "contrastando experimentos anteriores.\n"
                ),
                llm_config=llm_config,
            )

            ultimos = leer_ultimos_registros(max_lineas=5)
            contexto_previos = (
                ultimos if ultimos.strip() else "No hay experimentos previos registrados."
            )

            mensaje_cientifico = f"""
Vas a diseñar el experimento del ciclo {ciclo}.

Resúmenes recientes de experimentos (config + resultados + evaluación archivista),
en formato JSONL (cada línea un JSON):
{contexto_previos}

Debes devolver SOLO un JSON (sin texto adicional) con la configuración del nuevo experimento
siguiendo el esquema indicado en tu mensaje del sistema.
""".strip()

            respuesta_cientifico = cientifico.generate_reply(
                messages=[{"role": "user", "content": mensaje_cientifico}]
            )

            if isinstance(respuesta_cientifico, dict):
                contenido_cientifico = respuesta_cientifico.get("content", "")
                if not isinstance(contenido_cientifico, str):
                    contenido_cientifico = str(contenido_cientifico)
            else:
                contenido_cientifico = str(respuesta_cientifico)

            config = extraer_json_de_texto(contenido_cientifico)
            if not isinstance(config, dict):
                logging.error("No se pudo extraer un JSON de configuración válido del Científico.")
                logging.error(f"Respuesta bruta: {contenido_cientifico}")
                raise ValueError("Configuración inválida")

            logging.info(f"Config experimento ciclo {ciclo}: {config}")

            # ---------- 2. NÚCLEO FÍSICO: EJECUTA EL EXPERIMENTO ----------
            try:
                resultados, x, psi = quantum_core.run_schrodinger_1d(config)
            except Exception as e:
                logging.error(f"Error al ejecutar el núcleo cuántico: {e}")
                raise

            logging.info(
                f"Resultados experimento ciclo {ciclo}: "
                f"prob_region={resultados.get('prob_region'):.6f}, "
                f"prob_total={resultados.get('prob_total'):.6f}"
            )

            # ---------- 3. ARCHIVISTA: EVALÚA Y MARCA DESCUBRIMIENTOS ----------
            archivista = autogen.AssistantAgent(
                name="Archivista",
                system_message=(
                    "Eres el Archivista científico de un LABORATORIO CUÁNTICO 1D serio.\n"
                    "Recibes la configuración exacta de un experimento (JSON) y los "
                    "resultados numéricos (JSON) y debes:\n"
                    "- Resumir qué se ha hecho y qué se ha observado.\n"
                    "- Valorar la relevancia científica del experimento.\n"
                    "- Decidir si constituye un 'descubrimiento' dentro de este laboratorio.\n\n"
                    "Debes devolver SIEMPRE un ÚNICO OBJETO JSON con esta estructura:\n"
                    "{\n"
                    '  "ciclo": <int>,\n'
                    '  "descripcion_experimento": "...",\n'
                    '  "resultado_principal": "...",\n'
                    '  "metrica_relevancia": 0.0,\n'
                    '  "es_interesante": false,\n'
                    '  "es_descubrimiento": false,\n'
                    '  "motivo_descubrimiento": ""\n'
                    "}\n\n"
                    "Criterios de relevancia:\n"
                    "- metrica_relevancia en [0, 1].\n"
                    "- Considera más relevante si:\n"
                    "  * La probabilidad en la región objetivo es alta pero no trivial (no siempre 1.0 sin motivo).\n"
                    "  * El experimento explora un régimen diferente a los anteriores (por potencial, estado inicial, etc.).\n"
                    "  * Aparecen patrones o comportamientos no obvios (túnel parcial, oscilaciones, etc.).\n"
                    "- Marca es_descubrimiento = true solo si:\n"
                    "  * La configuración y la métrica sugieren un comportamiento especialmente interesante\n"
                    "    o mejoran claramente experimentos previos.\n"
                ),
                llm_config=llm_config,
            )

            resumen_prompt = f"""
Config del experimento (JSON):
{json.dumps(config, ensure_ascii=False, indent=2)}

Resultados numéricos del experimento (JSON):
{json.dumps(resultados, ensure_ascii=False, indent=2)}

Ciclo: {ciclo}

Genera el JSON de evaluación siguiendo la estructura indicada en tu mensaje del sistema.
""".strip()

            respuesta_archivista = archivista.generate_reply(
                messages=[{"role": "user", "content": resumen_prompt}]
            )

            if isinstance(respuesta_archivista, dict):
                contenido_archivista = respuesta_archivista.get("content", "")
                if not isinstance(contenido_archivista, str):
                    contenido_archivista = str(contenido_archivista)
            else:
                contenido_archivista = str(respuesta_archivista)

            evaluacion = extraer_json_de_texto(contenido_archivista)
            if not isinstance(evaluacion, dict):
                logging.warning(
                    "El Archivista no devolvió un JSON de evaluación válido. "
                    "Contenido bruto:"
                )
                logging.warning(contenido_archivista)
                evaluacion = {
                    "ciclo": ciclo,
                    "descripcion_experimento": "Evaluación no disponible",
                    "resultado_principal": "",
                    "metrica_relevancia": 0.0,
                    "es_interesante": False,
                    "es_descubrimiento": False,
                    "motivo_descubrimiento": "",
                }
            else:
                evaluacion["ciclo"] = ciclo

            registro_completo = {
                "ciclo": ciclo,
                "config": config,
                "resultados": resultados,
                "evaluacion": evaluacion,
            }

            guardar_registro_completo(registro_completo)

            if evaluacion.get("es_descubrimiento"):
                guardar_descubrimiento(registro_completo)
                logging.info(
                    f"Descubrimiento ciclo {ciclo}: {evaluacion.get('descripcion_experimento')}"
                )

        except Exception as ciclo_error:
            logging.error(f"Error crítico en el ciclo {ciclo}: {ciclo_error}")

        logging.info(f"Descansando {CICLO_DELAY_SECONDS} segundos antes del siguiente ciclo...")
        time.sleep(CICLO_DELAY_SECONDS)


if __name__ == "__main__":
    logging.info("Arrancando Laboratorio Cuántico IA (núcleo 1D serio)...")
    simular_ciclo_de_investigacion()
