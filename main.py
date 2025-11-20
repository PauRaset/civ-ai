import time
import logging
import os
import autogen

# Configuración básica
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
api_key = os.environ.get("OPENAI_API_KEY")

if not api_key:
    logging.error("FALTA API KEY. Configúrala en Railway.")
    exit()

# Configuración del Cerebro (GPT)
config_list = [{"model": "gpt-3.5-turbo", "api_key": api_key}] # Usa gpt-4 si puedes permitírtelo
llm_config = {"config_list": config_list, "temperature": 0.5}

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
            # 1. El Agente Científico (El que escribe código)
            cientifico = autogen.AssistantAgent(
                name="Cientifico_Datos",
                system_message="Eres un experto en Python y Ciencia de Datos. Tu trabajo es escribir scripts de Python para resolver problemas matemáticos o simular fenómenos físicos simples. Cuando escribas código, ponlo en bloques ```python ... ```.",
                llm_config=llm_config,
            )

            # 2. El Ejecutor (Nosotros/El Sistema que corre el código)
            # AQUÍ ESTÁ LA MAGIA: use_docker=False ejecuta el código en el propio servidor Railway
            ejecutor = autogen.UserProxyAgent(
                name="Ordenador_Central",
                human_input_mode="NEVER",
                max_consecutive_auto_reply=5,  # Dejamos que interactúen hasta 5 veces
                code_execution_config={
                    "work_dir": work_dir,      # Donde se guardan los archivos
                    "use_docker": False,       # Corremos directo en el contenedor de Railway
                    "last_n_messages": 2       # Mira los últimos mensajes para buscar código
                }
            )
            
            # 3. Definimos una misión para este ciclo
            # En el futuro, esto vendría de una base de datos de "problemas pendientes"
            mision = "Genera un script en Python que simule una caída libre con resistencia del aire usando numpy, ejecuta la simulación y dime cuánto tarda el objeto en llegar al suelo desde 100 metros."
            
            logging.info(f"Misión enviada: {mision}")
            
            # 4. Iniciar la colaboración
            # El Ejecutor le da la orden al Científico -> Científico escribe código -> Ejecutor lo corre -> Le da el resultado -> Científico concluye.
            ejecutor.initiate_chat(
                cientifico,
                message=mision
            )
            
            logging.info("Ciclo terminado exitosamente.")

        except Exception as e:
            logging.error(f"Error en el ciclo: {e}")

        # Pausa de seguridad (10 minutos) para controlar gastos
        logging.info("Descansando 600 segundos...")
        time.sleep(600)

if __name__ == "__main__":
    simular_ciclo_de_investigacion()
