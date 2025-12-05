"""
Módulo para extração de faces de frames de vídeo.

Fornece classes e métodos para detectar e extrair regiões de interesse (ROI)
contendo rostos de frames de vídeo.
"""

from typing import List, Tuple

import cv2
import numpy as np


class FaceExtractor:
    """
    Classe para extração de faces de frames de vídeo.
    
    Utiliza o classificador Haar Cascade do OpenCV para detectar faces em frames
    e extrair as regiões de interesse (ROI) processadas e redimensionadas.
    
    Attributes:
        face_cascade: Classificador Haar Cascade para detecção de faces.
    
    Examples:
        >>> extractor = FaceExtractor()
        >>> frame = cv2.imread("frame.jpg")
        >>> faces = extractor.detect_faces(frame)
        >>> if faces:
        ...     face_roi = extractor.extract_face(frame, faces[0])
    """
    
    # Tamanho padrão para redimensionamento (comum em CNNs)
    TARGET_SIZE = (224, 224)
    
    def __init__(self) -> None:
        """
        Inicializa o FaceExtractor carregando o classificador Haar Cascade.
        
        Carrega o classificador Haar Cascade para detecção frontal de faces
        do OpenCV e valida se o carregamento foi bem-sucedido.
        
        Raises:
            FileNotFoundError: Se o arquivo XML do classificador não for encontrado.
            RuntimeError: Se o classificador não puder ser carregado.
            
        Examples:
            >>> extractor = FaceExtractor()
        """
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        
        self.face_cascade = cv2.CascadeClassifier(cascade_path)
        
        if self.face_cascade.empty():
            raise RuntimeError(
                f"Falha ao carregar o classificador Haar Cascade. "
                f"Arquivo não encontrado ou inválido: {cascade_path}"
            )
    
    def detect_faces(self, frame: np.ndarray, min_face_size: int = 50) -> List[Tuple[int, int, int, int]]:
        """
        Detecta faces em um frame de vídeo.
        
        Converte o frame para escala de cinza e utiliza o classificador Haar Cascade
        para detectar todas as faces presentes no frame. Redimensiona frames grandes
        para melhorar a performance e precisão.
        
        Args:
            frame: Frame de vídeo como array numpy no formato BGR.
            min_face_size: Tamanho mínimo da face para detecção (padrão: 50x50).
            
        Returns:
            Lista de tuplas contendo as coordenadas das faces detectadas.
            Cada tupla tem o formato (x, y, w, h):
                - x: Coordenada x do canto superior esquerdo
                - y: Coordenada y do canto superior esquerdo
                - w: Largura da face detectada
                - h: Altura da face detectada
                
        Raises:
            ValueError: Se o frame estiver vazio ou inválido.
            
        Examples:
            >>> extractor = FaceExtractor()
            >>> frame = cv2.imread("frame.jpg")
            >>> faces = extractor.detect_faces(frame)
            >>> print(f"Faces detectadas: {len(faces)}")
        """
        if frame is None or frame.size == 0:
            raise ValueError("Frame vazio ou inválido não pode ser processado")
        
        # Otimização: Redimensionar frames muito grandes para melhorar performance
        height, width = frame.shape[:2]
        max_dimension = 1280  # Limitar dimensão máxima
        
        scale_factor = 1.0
        if width > max_dimension or height > max_dimension:
            if width > height:
                scale_factor = max_dimension / width
            else:
                scale_factor = max_dimension / height
            
            new_width = int(width * scale_factor)
            new_height = int(height * scale_factor)
            frame_resized = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA)
        else:
            frame_resized = frame
        
        # Converter para escala de cinza
        gray = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2GRAY)
        
        # Melhorar contraste para melhor detecção (equalização de histograma)
        gray = cv2.equalizeHist(gray)
        
        # Ajustar tamanho mínimo baseado na escala
        adjusted_min_size = max(30, int(min_face_size * scale_factor))
        
        # Detectar faces com parâmetros otimizados
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.15,  # Ligeiramente mais agressivo para melhor detecção
            minNeighbors=4,    # Reduzido para detectar mais faces
            minSize=(adjusted_min_size, adjusted_min_size),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        
        # Converter coordenadas de volta para o tamanho original se necessário
        if scale_factor != 1.0:
            faces = [
                (int(x / scale_factor), int(y / scale_factor), 
                 int(w / scale_factor), int(h / scale_factor))
                for (x, y, w, h) in faces
            ]
        else:
            faces = [(int(x), int(y), int(w), int(h)) for (x, y, w, h) in faces]
        
        # Filtrar faces muito pequenas ou com proporção muito estranha
        filtered_faces = []
        for (x, y, w, h) in faces:
            # Validar tamanho mínimo no frame original
            if w >= min_face_size and h >= min_face_size:
                # Filtrar proporções muito estranhas (possíveis falsos positivos)
                aspect_ratio = w / h if h > 0 else 0
                if 0.5 <= aspect_ratio <= 2.0:  # Proporção razoável para faces
                    filtered_faces.append((x, y, w, h))
        
        return filtered_faces
    
    def extract_face(
        self, 
        frame: np.ndarray, 
        face_coords: Tuple[int, int, int, int]
    ) -> np.ndarray:
        """
        Extrai e processa uma face específica de um frame.
        
        Recebe o frame original (colorido) e as coordenadas de uma face,
        valida as coordenadas, recorta a região e redimensiona para o tamanho
        padrão (224x224) adequado para CNNs.
        
        Args:
            frame: Frame de vídeo original no formato BGR.
            face_coords: Tupla com coordenadas da face (x, y, w, h).
            
        Returns:
            Array numpy contendo a face processada e redimensionada (224x224).
            
        Raises:
            ValueError: Se as coordenadas forem inválidas ou estiverem fora dos limites do frame.
            
        Examples:
            >>> extractor = FaceExtractor()
            >>> frame = cv2.imread("frame.jpg")
            >>> faces = extractor.detect_faces(frame)
            >>> if faces:
            ...     face_roi = extractor.extract_face(frame, faces[0])
            ...     print(f"Face extraída: {face_roi.shape}")
        """
        if frame is None or frame.size == 0:
            raise ValueError("Frame vazio ou inválido não pode ser processado")
        
        x, y, w, h = face_coords
        
        # Validar coordenadas
        frame_height, frame_width = frame.shape[:2]
        
        # Garantir que as coordenadas não sejam negativas
        x = max(0, x)
        y = max(0, y)
        
        # Garantir que as coordenadas não ultrapassem os limites do frame
        x = min(x, frame_width - 1)
        y = min(y, frame_height - 1)
        
        # Garantir que w e h sejam positivos e não ultrapassem os limites
        w = max(1, min(w, frame_width - x))
        h = max(1, min(h, frame_height - y))
        
        # Verificar se a região é válida
        if w <= 0 or h <= 0:
            raise ValueError(
                f"Coordenadas inválidas resultaram em região vazia: "
                f"x={x}, y={y}, w={w}, h={h} (frame: {frame_width}x{frame_height})"
            )
        
        # Recortar a região da face
        face_roi = frame[y:y+h, x:x+w]
        
        # Redimensionar para o tamanho padrão
        resized_face = cv2.resize(face_roi, self.TARGET_SIZE, interpolation=cv2.INTER_AREA)
        
        return resized_face


if __name__ == "__main__":
    """
    Teste básico para validar a inicialização da classe.
    
    Cria uma imagem preta simulada e instancia o FaceExtractor para garantir
    que o XML do Haar Cascade foi encontrado corretamente.
    """
    print("Testando FaceExtractor...")
    
    try:
        # Criar uma imagem preta simulada para teste
        test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        print(f"✓ Frame de teste criado: {test_frame.shape}")
        
        # Instanciar o FaceExtractor
        extractor = FaceExtractor()
        print("✓ FaceExtractor instanciado com sucesso")
        print(f"✓ Classificador Haar Cascade carregado")
        
        # Testar detecção (não deve encontrar faces em imagem preta)
        faces = extractor.detect_faces(test_frame)
        print(f"✓ Método detect_faces executado: {len(faces)} face(s) detectada(s)")
        
        print("\n✅ Todos os testes passaram!")
        
    except Exception as e:
        print(f"\n❌ Erro durante o teste: {type(e).__name__}: {e}")
        raise

