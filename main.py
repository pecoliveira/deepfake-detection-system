
from pathlib import Path


def main():
    
    print("Deepfake Detection Tool")
    print("iniciado com sucesso!")
    
    # Verificar se os diretórios necessários existem
    base_dir = Path(__file__).parent
    required_dirs = [
        base_dir / "data" / "input",
        base_dir / "data" / "processed",
        base_dir / "data" / "output",
        base_dir / "models",
    ]
    
    for dir_path in required_dirs:
        if not dir_path.exists():
            print(f"Aviso: Dir não encontrado: {dir_path}")
        else:
            print(f"✓ Dir encontrado: {dir_path}")


if __name__ == "__main__":
    main()

