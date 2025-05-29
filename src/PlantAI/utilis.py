# utilitarios para o pacote PlantAI
import argparse

# parse_args:
# Função para parsear os argumentos da linha de comando# 

def parse_args():
    parser = argparse.ArgumentParser(description="Meu pacote PlantAI")
        
    parser.add_argument(
        "problema",
        help="Problema a ser resolvido",
        )
    
    return parser.parse_args()