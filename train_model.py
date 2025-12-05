
import logging
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

from src.core.face_extractor import FaceExtractor
from src.core.video_processor import VideoLoader
from src.detectors.ai_model import DeepFakeClassifier

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def extract_faces_from_videos(video_paths: List[Path], label: int, face_extractor: FaceExtractor, 
                               frames_per_video: int = 10) -> Tuple[List[np.ndarray], List[int]]:

    all_faces = []
    all_labels = []
    
    for video_path in video_paths:
        logger.info(f"Processando vídeo: {video_path.name}")
        faces_count = 0
        
        try:
            # Processar frames do vídeo
            frame_count = 0
            for frame, frame_number in VideoLoader.load_video(video_path):
                # Processar apenas alguns frames por vídeo
                if frame_count % 30 == 0:  # A cada 30 frames
                    faces = face_extractor.detect_faces(frame, min_face_size=50)
                    
                    if faces:
                        # Pegar a maior face
                        faces_sorted = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
                        face_coords = faces_sorted[0]
                        
                        try:
                            face_img = face_extractor.extract_face(frame, face_coords)
                            all_faces.append(face_img)
                            all_labels.append(label)
                            faces_count += 1
                            
                            if faces_count >= frames_per_video:
                                break
                        except Exception as e:
                            logger.warning(f"Erro ao extrair face do frame {frame_number}: {e}")
                            continue
                    
                    frame_count += 1
                    
                    if faces_count >= frames_per_video:
                        break
                        
            logger.info(f"  ✓ Extraídas {faces_count} faces de {video_path.name}")
            
        except Exception as e:
            logger.error(f"Erro ao processar vídeo {video_path.name}: {e}")
            continue
    
    return all_faces, all_labels


def prepare_training_data(fake_dir: Path, real_dir: Path, face_extractor: FaceExtractor,
                          frames_per_video: int = 10) -> Tuple[np.ndarray, np.ndarray]:
    """
    Prepara dados de treinamento a partir dos vídeos nas pastas.
    
    Args:
        fake_dir: Diretório com vídeos fake.
        real_dir: Diretório com vídeos real.
        face_extractor: Extrator de faces.
        frames_per_video: Número de frames para processar de cada vídeo.
        
    Returns:
        Tupla contendo (arrays de faces, arrays de labels).
    """
    logger.info("=" * 60)
    logger.info("PREPARANDO DADOS DE TREINAMENTO")
    logger.info("=" * 60)
    
    # Buscar vídeos fake
    fake_videos = []
    for ext in ['*.mp4', '*.avi', '*.mov', '*.MOV', '*.MP4', '*.AVI']:
        fake_videos.extend(list(fake_dir.glob(ext)))
    
    logger.info(f"\nVídeos FAKE encontrados: {len(fake_videos)}")
    for video in fake_videos:
        logger.info(f"  - {video.name}")
    
    # Buscar vídeos real
    real_videos = []
    for ext in ['*.mp4', '*.avi', '*.mov', '*.MOV', '*.MP4', '*.AVI']:
        real_videos.extend(list(real_dir.glob(ext)))
    
    logger.info(f"\nVídeos REAL encontrados: {len(real_videos)}")
    for video in real_videos:
        logger.info(f"  - {video.name}")
    
    if len(fake_videos) == 0 and len(real_videos) == 0:
        raise ValueError("Nenhum vídeo encontrado nas pastas fake ou real!")
    
    # Extrair faces dos vídeos fake
    logger.info("\n" + "=" * 60)
    logger.info("EXTRAINDO FACES DE VÍDEOS FAKE")
    logger.info("=" * 60)
    fake_faces, fake_labels = extract_faces_from_videos(
        fake_videos, label=1, face_extractor=face_extractor, frames_per_video=frames_per_video
    )
    
    # Extrair faces dos vídeos real
    logger.info("\n" + "=" * 60)
    logger.info("EXTRAINDO FACES DE VÍDEOS REAL")
    logger.info("=" * 60)
    real_faces, real_labels = extract_faces_from_videos(
        real_videos, label=0, face_extractor=face_extractor, frames_per_video=frames_per_video
    )
    
    # Combinar dados
    all_faces = fake_faces + real_faces
    all_labels = fake_labels + real_labels
    
    logger.info("\n" + "=" * 60)
    logger.info("RESUMO DOS DADOS")
    logger.info("=" * 60)
    logger.info(f"Total de faces extraídas: {len(all_faces)}")
    logger.info(f"  - FAKE: {len(fake_faces)}")
    logger.info(f"  - REAL: {len(real_faces)}")
    
    if len(all_faces) == 0:
        raise ValueError("Nenhuma face foi extraída dos vídeos! Verifique se os vídeos contêm faces.")
    
    # Converter para arrays numpy
    X = np.array(all_faces, dtype=np.float32)
    y = np.array(all_labels, dtype=np.float32)
    
    # Embaralhar dados
    X, y = shuffle(X, y, random_state=42)
    
    return X, y


def train_model(epochs: int = 20, batch_size: int = 16, frames_per_video: int = 10):
    """
    Treina o modelo de detecção de deepfakes.
    
    Args:
        epochs: Número de épocas de treinamento.
        batch_size: Tamanho do batch.
        frames_per_video: Número de frames para processar de cada vídeo.
    """
    logger.info("=" * 60)
    logger.info("INICIANDO TREINAMENTO DO MODELO")
    logger.info("=" * 60)
    
    # Definir caminhos
    base_dir = Path(__file__).parent
    fake_dir = base_dir / "data" / "input" / "fake"
    real_dir = base_dir / "data" / "input" / "real"
    models_dir = base_dir / "models"
    models_dir.mkdir(exist_ok=True)
    
    # Verificar se as pastas existem
    if not fake_dir.exists():
        raise FileNotFoundError(f"Pasta de vídeos fake não encontrada: {fake_dir}")
    if not real_dir.exists():
        raise FileNotFoundError(f"Pasta de vídeos real não encontrada: {real_dir}")
    
    # Inicializar extrator de faces
    logger.info("\nInicializando extrator de faces...")
    face_extractor = FaceExtractor()
    logger.info("✓ Extrator de faces inicializado")
    
    # Preparar dados
    X, y = prepare_training_data(fake_dir, real_dir, face_extractor, frames_per_video)
    
    # Dividir em treino e validação
    logger.info("\n" + "=" * 60)
    logger.info("DIVIDINDO DADOS EM TREINO E VALIDAÇÃO")
    logger.info("=" * 60)
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    logger.info(f"Treino: {len(X_train)} amostras")
    logger.info(f"Validação: {len(X_val)} amostras")
    
    # Converter BGR para RGB e pré-processar
    logger.info("\nPré-processando imagens...")
    X_train_rgb = np.array([cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2RGB) for img in X_train])
    X_val_rgb = np.array([cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2RGB) for img in X_val])
    
    # Aplicar pré-processamento MobileNetV2
    from tensorflow.keras.applications import mobilenet_v2
    X_train_preprocessed = mobilenet_v2.preprocess_input(X_train_rgb.astype(np.float32))
    X_val_preprocessed = mobilenet_v2.preprocess_input(X_val_rgb.astype(np.float32))
    
    logger.info("✓ Pré-processamento concluído")
    
    # Criar e compilar modelo
    logger.info("\n" + "=" * 60)
    logger.info("CRIANDO E COMPILANDO MODELO")
    logger.info("=" * 60)
    classifier = DeepFakeClassifier()
    classifier.compile_model()
    logger.info("✓ Modelo criado e compilado")
    
    # Callbacks para treinamento
    weights_path = models_dir / "deepfake_classifier.h5"
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True,
            verbose=1
        ),
        ModelCheckpoint(
            filepath=str(weights_path),
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=0.00001,
            verbose=1
        )
    ]
    
    # Treinar modelo
    logger.info("\n" + "=" * 60)
    logger.info("TREINANDO MODELO")
    logger.info("=" * 60)
    logger.info(f"Épocas: {epochs}")
    logger.info(f"Batch size: {batch_size}")
    
    history = classifier.model.fit(
        X_train_preprocessed, y_train,
        validation_data=(X_val_preprocessed, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1
    )
    
    # Salvar pesos finais
    logger.info("\nSalvando pesos finais...")
    classifier.save_weights(weights_path)
    logger.info(f"✓ Pesos salvos em: {weights_path}")
    
    # Mostrar resultados finais
    logger.info("\n" + "=" * 60)
    logger.info("RESULTADOS DO TREINAMENTO")
    logger.info("=" * 60)
    final_train_acc = history.history['accuracy'][-1]
    final_val_acc = history.history['val_accuracy'][-1]
    logger.info(f"Acurácia de treino: {final_train_acc:.4f}")
    logger.info(f"Acurácia de validação: {final_val_acc:.4f}")
    logger.info("\n✅ Treinamento concluído com sucesso!")


if __name__ == "__main__":
    import sys
    
    try:
        # Configurar parâmetros de treinamento
        EPOCHS = 20
        BATCH_SIZE = 16
        FRAMES_PER_VIDEO = 10
        
        # Executar treinamento
        train_model(
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            frames_per_video=FRAMES_PER_VIDEO
        )
        
    except KeyboardInterrupt:
        logger.info("\n⚠️ Treinamento interrompido.")
        sys.exit(1)
    except Exception as e:
        logger.error(f"\n❌ Erro: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

