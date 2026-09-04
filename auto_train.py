import schedule
import time
import subprocess
import logging
from datetime import datetime

# Logs para saberes se treinou mesmo enquanto dormia
logging.basicConfig(
    filename='logs/auto_train.log',
    level=logging.INFO,
    format='%(asctime)s - %(message)s'
)

def executar_fluxo_completo():
    try:
        logging.info("Iniciando ciclo diário de atualização e treino...")
        #banco de dados 
        subprocess.run("python updater.py --once --intervalo 1h", shell=True, check=True)
        #treinamento da ia
        subprocess.run("python treinar_todos.py", shell=True, check=True)
        
        logging.info("concluído")
        print(f"[{datetime.now()}] Tudo pronto! Modelos atualizados.")
        
    except Exception as e:
        logging.error(f"Erro: {e}")
        print(f"Erro: {e}")

schedule.every().day.at("02:00").do(executar_fluxo_completo)

while True:
    schedule.run_pending()
    time.sleep(60) 