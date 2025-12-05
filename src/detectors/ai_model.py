"""
Módulo de detecção de deepfakes baseado em IA.

Implementa redes neurais convolucionais para detecção de deepfakes usando
transfer learning com MobileNetV2 como arquitetura base.
"""

import logging
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from tensorflow import keras
from tensorflow.keras import applications
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DeepFakeClassifier:
    """
    Classificador de deepfakes usando rede neural convolucional.
    
    Utiliza MobileNetV2 como arquitetura base através de transfer learning,
    adicionando camadas personalizadas para classificação binária (Real/Fake).
    
    Attributes:
        model: Modelo Keras compilado e pronto para predição ou treinamento.
        input_size: Tamanho esperado da imagem de entrada (224, 224).
        is_trained: Indica se o modelo possui pesos treinados carregados.
    
    Examples:
        >>> classifier = DeepFakeClassifier()
        >>> face_img = cv2.imread("face.jpg")  # 224x224
        >>> probability = classifier.predict(face_img)
        >>> print(f"Probabilidade de ser fake: {probability:.4f}")
    """
    
    INPUT_SIZE = (224, 224, 3)  # Tamanho padrão para MobileNetV2
    
    def __init__(self, weights_path: Optional[str | Path] = None) -> None:
        """
        Inicializa o classificador e constrói a arquitetura do modelo.
        
        Constrói a arquitetura baseada em MobileNetV2 com camadas personalizadas
        no topo. Tenta carregar pesos treinados se um caminho for fornecido.
        
        Args:
            weights_path: Caminho opcional para arquivo de pesos (.h5 ou .keras).
                         Se None, o modelo inicia com pesos ImageNet (base) e
                         pesos aleatórios (topo).
        
        Examples:
            >>> classifier = DeepFakeClassifier()
            >>> # ou com pesos pré-treinados
            >>> classifier = DeepFakeClassifier(weights_path="models/weights.h5")
        """
        self.model = self._build_model()
        self.is_trained = False
        
        if weights_path is not None:
            self.load_weights(weights_path)
    
    def _build_model(self) -> Model:
        """
        Constrói a arquitetura do modelo usando MobileNetV2 como base.
        
        Cria o modelo com:
        - Base: MobileNetV2 pré-treinado na ImageNet (sem topo)
        - Topo personalizado: GlobalAveragePooling2D, Dense(128), Dropout, Dense(1)
        
        Returns:
            Modelo Keras compilado e pronto para uso.
        """
        # Base: MobileNetV2 pré-treinado na ImageNet
        base_model = applications.MobileNetV2(
            input_shape=self.INPUT_SIZE,
            include_top=False,
            weights='imagenet'
        )
        
        # Congelar os pesos da base para transfer learning
        base_model.trainable = False
        
        # Construir o modelo completo
        inputs = keras.Input(shape=self.INPUT_SIZE)
        
        # Passar pela base (já inclui pré-processamento necessário)
        x = base_model(inputs, training=False)
        
        # Camadas personalizadas no topo
        x = GlobalAveragePooling2D()(x)
        x = Dense(128, activation='relu')(x)
        x = Dropout(0.5)(x)
        outputs = Dense(1, activation='sigmoid')(x)
        
        # Criar o modelo
        model = Model(inputs, outputs, name='DeepFakeClassifier')
        
        return model
    
    def compile_model(self) -> None:
        """
        Compila o modelo para treinamento futuro.
        
        Configura o otimizador Adam e a função de perda BinaryCrossentropy
        adequados para classificação binária.
        
        Examples:
            >>> classifier = DeepFakeClassifier()
            >>> classifier.compile_model()
            >>> # Agora pode treinar com model.fit()
        """
        self.model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss=BinaryCrossentropy(),
            metrics=['accuracy']
        )
        logger.info("Modelo compilado para treinamento")
    
    def load_weights(self, weights_path: str | Path) -> None:
        """
        Carrega pesos treinados do arquivo especificado.
        
        Tenta carregar pesos de arquivos .h5 ou .keras. Se o arquivo não existir,
        o modelo continua com pesos ImageNet (base) e aleatórios (topo).
        
        Args:
            weights_path: Caminho para o arquivo de pesos (.h5 ou .keras).
            
        Raises:
            FileNotFoundError: Se o arquivo não for encontrado.
            ValueError: Se houver erro ao carregar os pesos.
            
        Examples:
            >>> classifier = DeepFakeClassifier()
            >>> classifier.load_weights("models/weights.h5")
        """
        weights_path = Path(weights_path)
        
        if not weights_path.exists():
            raise FileNotFoundError(
                f"Arquivo de pesos não encontrado: {weights_path}. "
                f"O modelo continuará sem pesos treinados."
            )
        
        try:
            # Garantir que o modelo está compilado antes de carregar pesos
            if not self.model.compiled:
                self.compile_model()
            
            # Carregar pesos
            self.model.load_weights(str(weights_path))
            self.is_trained = True
            
            logger.info(f"Pesos carregados com sucesso de: {weights_path}")
            
        except Exception as e:
            raise ValueError(
                f"Erro ao carregar pesos de {weights_path}: {str(e)}"
            ) from e
    
    def save_weights(self, weights_path: str | Path) -> None:
        """
        Salva os pesos do modelo treinado.
        
        Salva os pesos em formato .h5 ou .keras para uso futuro.
        Cria o diretório se não existir.
        
        Args:
            weights_path: Caminho onde os pesos serão salvos.
            
        Raises:
            ValueError: Se houver erro ao salvar os pesos.
            
        Examples:
            >>> classifier = DeepFakeClassifier()
            >>> # Após treinamento
            >>> classifier.save_weights("models/weights.h5")
        """
        weights_path = Path(weights_path)
        
        # Criar diretório se não existir
        weights_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            self.model.save_weights(str(weights_path))
            logger.info(f"Pesos salvos com sucesso em: {weights_path}")
            
        except Exception as e:
            raise ValueError(
                f"Erro ao salvar pesos em {weights_path}: {str(e)}"
            ) from e
    
    def predict(self, face_img: np.ndarray) -> float:
        """
        Realiza predição em uma imagem de rosto.
        
        Recebe uma imagem de rosto já recortada e processada (224x224),
        aplica o pré-processamento específico da MobileNetV2 e retorna
        a probabilidade de ser fake (0.0 = Real, 1.0 = Fake).
        
        Args:
            face_img: Array numpy com a imagem do rosto (224x224x3) em formato BGR.
            
        Returns:
            Probabilidade de ser fake (float entre 0.0 e 1.0).
            
        Raises:
            ValueError: Se a imagem estiver vazia ou com dimensões incorretas.
            
        Examples:
            >>> classifier = DeepFakeClassifier()
            >>> face_img = cv2.imread("face.jpg")  # 224x224
            >>> probability = classifier.predict(face_img)
            >>> if probability > 0.5:
            ...     print("Provavelmente fake!")
        """
        if face_img is None or face_img.size == 0:
            raise ValueError("Imagem vazia ou inválida não pode ser processada")
        
        # Validar dimensões
        if len(face_img.shape) != 3 or face_img.shape[2] != 3:
            raise ValueError(
                f"Imagem deve ter 3 canais (BGR). Recebido: {face_img.shape}"
            )
        
        # Converter BGR para RGB (OpenCV usa BGR, mas MobileNet espera RGB)
        face_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
        
        # Pré-processamento específico da MobileNetV2
        # MobileNetV2 espera valores no range [-1, 1]
        preprocessed = applications.mobilenet_v2.preprocess_input(
            face_rgb.astype(np.float32)
        )
        
        # Expandir dimensões para criar batch de tamanho 1
        batch = np.expand_dims(preprocessed, axis=0)
        
        # Fazer predição
        prediction = self.model.predict(batch, verbose=0)
        
        # Retornar probabilidade de ser fake (valor único do array)
        probability = float(prediction[0][0])
        
        return probability


if __name__ == "__main__":
    """
    Teste básico para validar a classe DeepFakeClassifier.
    
    Instancia a classe, imprime o resumo do modelo e faz uma predição
    dummy com uma imagem aleatória para validar o fluxo de dados.
    """
    print("Testando DeepFakeClassifier...")
    print("=" * 60)
    
    try:
        # Instanciar o classificador
        print("\n1. Instanciando DeepFakeClassifier...")
        classifier = DeepFakeClassifier()
        print("✓ Classificador instanciado com sucesso")
        
        # Compilar o modelo (necessário para algumas operações)
        print("\n2. Compilando modelo...")
        classifier.compile_model()
        print("✓ Modelo compilado")
        
        # Mostrar resumo da arquitetura
        print("\n3. Resumo da Arquitetura do Modelo:")
        print("=" * 60)
        classifier.model.summary()
        print("=" * 60)
        
        # Criar imagem dummy aleatória (224x224x3) no formato BGR
        print("\n4. Criando imagem dummy para teste...")
        dummy_img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        print(f"✓ Imagem dummy criada: {dummy_img.shape}")
        
        # Fazer predição
        print("\n5. Realizando predição...")
        probability = classifier.predict(dummy_img)
        print(f"✓ Predição realizada com sucesso")
        print(f"\n{'=' * 60}")
        print("RESULTADO DA PREDIÇÃO:")
        print(f"{'=' * 60}")
        print(f"Probabilidade de ser FAKE: {probability:.4f}")
        print(f"Classificação: {'FAKE' if probability > 0.5 else 'REAL'}")
        print(f"Modelo treinado: {classifier.is_trained}")
        print(f"{'=' * 60}")
        
        print("\n✅ Todos os testes passaram!")
        
    except Exception as e:
        print(f"\n❌ Erro durante o teste: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        raise
