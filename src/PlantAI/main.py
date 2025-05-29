import sys
from .utilis import parse_args

def main():    
    print("PlantAI começa aqui!")
    args = parse_args()
    try:
        problema = args.problema
        print(f"Problema: {problema}")
        return 0
    except Exception as e:
        print(f"Erro: {e}")
        return 1
     
if __name__ == "__main__":
    sys.exit(main())