import time
import logging
import os
import autogen

# 1. Configuración básica de Logs
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

# 2. Configuración de API Key
api_key = os.environ.get("OPENAI_API_KEY")

if not api_key:
    logging.error("FALTA API KEY. Configúrala en las Variables de Railway.")
    exit()

# 3. Configuración del Modelo (Cerebro)
# Nota: Si tienes acceso a GPT-4, úsalo aquí para mejor razonamiento científico.
config_list = [{"model": "gpt-3.5-turbo", "api_key": api_key}]
llm_config = {
    "config_list": config_list,
    "temperature": 0.5, # 0.5 es bueno para equilibrio entre creatividad y precisión
}

def simular_ciclo_de_investigacion():
    ciclo = 0
    
    # Definimos el directorio donde las IAs guardarán sus scripts
    work_dir = "laboratorio_codigo"
    if not os.path.exists(work_dir):
        os.makedirs(work_dir)

    while True:
        ciclo += 1
        logging.info(f"\n--- INICIO DEL CICLO {ciclo} ---")
        
        try:
            # --- AGENTE 1: El Científico (Escribe el código) ---
            cientifico = autogen.AssistantAgent(
                name="Cientifico_Datos",
                system_message="Eres un experto en Python y Ciencia de Datos. Tu trabajo es escribir scripts de Python para resolver problemas matemáticos o simular fenómenos físicos simples. Cuando escribas código, ponlo siempre dentro de bloques ```python ... ```.",
                llm_config=llm_config,
            )

            # --- AGENTE 2: El Ordenador Central (Ejecuta el código) ---
            # use_docker=False es vital en Railway porque ya estamos dentro de un contenedor
            ejecutor = autogen.UserProxyAgent(
                name="Ordenador_Central",
                human_input_mode="NEVER",
                max_consecutive_auto_reply=5,
                code_execution_config={
                    "work_dir": work_dir,
                    "use_docker": False, 
                    "last_n_messages": 2
                }
            )
            
            # --- DEFINICIÓN DE LA MISIÓN ---
            # Pedimos algo que requiera cálculo real, no solo texto.
            mision = "Genera un script en Python que simule una caída libre con resistencia del aire usando numpy. Ejecuta la simulación y dime cuánto tarda el objeto en llegar al suelo desde 100 metros."
            
            logging.info(f"Misión enviada: {mision}")
            
            # --- INICIO DE LA COLABORACIÓN ---
            ejecutor.initiate_chat(
                cientifico,
                message=mision
            )
            
            logging.info("Ciclo terminado exitosamente.")

        except Exception as e:
            logging.error(f"Error crítico en el ciclo: {e}")

        # --- DESCANSO ---
        # 600 segundos = 10 minutos. Ajusta esto según tu presupuesto.
        logging.info("Descansando 600 segundos...")
        time.sleep(600)

if __name__ == "__main__":
    logging.info("Arrancando Sistema de Civilización IA...")
    simular_ciclo_de_investigacion()
