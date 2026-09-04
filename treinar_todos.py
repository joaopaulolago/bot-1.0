import subprocess
import time

ativos = ["bitcoin", "ethereum", "solana", "nvidia", "tesla", "ouro", "sp500", "petroleo"]

print(f"🚀 Iniciando treinamento em massa de {len(ativos)} ativos...")
inicio_total = time.time()

for ativo in ativos:
    print(f"\n" + "="*40)
    print(f"Treinando agora: {ativo.upper()}")
    print("="*40)
    comando = f"python train.py --ativo {ativo} --intervalo 1h --epochs 50 --window 60"
    
    try:
        subprocess.run(comando, shell=True, check=True)
        print(f" {ativo.upper()} concluído com sucesso!")
    except subprocess.CalledProcessError as e:
        print(f" Erro ao treinar {ativo}: {e}")

fim_total = time.time()
tempo_gasto = (fim_total - inicio_total) / 60

print("\n" + "!"*40)
print(f" TODOS OS TREINOS CONCLUÍDOS!")
print(f" Tempo total: {tempo_gasto:.2f} minutos")
print("!"*40)