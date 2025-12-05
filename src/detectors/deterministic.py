"""
Módulo de detecção determinística de deepfakes.

Implementa técnicas clássicas de processamento de imagem para detecção
de deepfakes sem uso de redes neurais. Utiliza análise de bordas e análise
espectral para identificar artefatos comuns em deepfakes de baixa qualidade.
"""

from typing import Dict, Any

import cv2
import numpy as np


class DeterministicDetector:
    """
    Detector determinístico de deepfakes baseado em análise visual.
    
    Utiliza técnicas clássicas de processamento de imagem para identificar
    artefatos comuns em deepfakes:
    - Bordas borradas ou muito nítidas (falha no blending)
    - Textura da pele excessivamente lisa (perda de alta frequência)
    
    Attributes:
        EDGE_VARIANCE_THRESHOLD: Limiar empírico para variância do Laplaciano.
        FFT_MEAN_THRESHOLD_LOW: Limiar inferior para média do espectro FFT.
        FFT_MEAN_THRESHOLD_HIGH: Limiar superior para média do espectro FFT.
    
    Examples:
        >>> detector = DeterministicDetector()
        >>> face_img = cv2.imread("face.jpg")
        >>> result = detector.detect(face_img)
        >>> print(result['is_suspect'], result['confidence'])
    """
    
    # Thresholds empíricos para detecção
    EDGE_VARIANCE_THRESHOLD = 100.0  # Variância do Laplaciano
    FFT_MEAN_THRESHOLD_LOW = 10.0    # Limiar inferior para média do espectro
    FFT_MEAN_THRESHOLD_HIGH = 50.0   # Limiar superior para média do espectro
    
    def _analyze_edges(self, face_img: np.ndarray) -> float:
        """
        Analisa as bordas da imagem usando o operador Laplaciano.
        
        Converte a imagem para escala de cinza e aplica o filtro Laplaciano
        para destacar as bordas. Calcula a variância do resultado para detectar
        se a imagem está muito borrada (variância baixa) ou com bordas artificiais
        muito nítidas (variância muito alta).
        
        Args:
            face_img: Imagem da face como array numpy (BGR ou RGB).
            
        Returns:
            Variância do Laplaciano (score de nitidez/borrão).
            
        Raises:
            ValueError: Se a imagem estiver vazia ou inválida.
            
        Examples:
            >>> detector = DeterministicDetector()
            >>> face_img = cv2.imread("face.jpg")
            >>> variance = detector._analyze_edges(face_img)
            >>> print(f"Variância: {variance}")
        """
        if face_img is None or face_img.size == 0:
            raise ValueError("Imagem vazia ou inválida não pode ser processada")
        
        # Converter para escala de cinza
        gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY) if len(face_img.shape) == 3 else face_img
        
        # Aplicar o filtro Laplaciano para destacar bordas
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        
        # Calcular a variância do Laplaciano
        # Valores baixos indicam imagem borrada (comum em deepfakes)
        # Valores muito altos podem indicar recortes artificiais
        variance = laplacian.var()
        
        return float(variance)
    
    def _analyze_fft(self, face_img: np.ndarray) -> float:
        """
        Analisa o espectro de frequências da imagem usando Transformada de Fourier.
        
        Converte a imagem para escala de cinza, aplica a Transformada de Fourier 2D,
        move o componente de frequência zero para o centro e calcula a magnitude do
        espectro. Deepfakes tendem a apresentar frequências altas anômalas ou ausentes
        devido à perda de detalhes durante o processamento.
        
        Args:
            face_img: Imagem da face como array numpy (BGR ou RGB).
            
        Returns:
            Média da magnitude do espectro de frequências.
            
        Raises:
            ValueError: Se a imagem estiver vazia ou inválida.
            
        Examples:
            >>> detector = DeterministicDetector()
            >>> face_img = cv2.imread("face.jpg")
            >>> fft_mean = detector._analyze_fft(face_img)
            >>> print(f"Média do espectro: {fft_mean}")
        """
        if face_img is None or face_img.size == 0:
            raise ValueError("Imagem vazia ou inválida não pode ser processada")
        
        # Converter para escala de cinza
        gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY) if len(face_img.shape) == 3 else face_img
        
        # Converter para float para melhor precisão na FFT
        gray_float = gray.astype(np.float32)
        
        # Aplicar a Transformada de Fourier 2D
        f_transform = np.fft.fft2(gray_float)
        
        # Mover o componente de frequência zero para o centro
        f_shift = np.fft.fftshift(f_transform)
        
        # Calcular a magnitude do espectro
        # Usar log para melhor visualização (evitar divisão por zero)
        magnitude_spectrum = 20 * np.log(np.abs(f_shift) + 1)  # +1 para evitar log(0)
        
        # Calcular a média da magnitude
        fft_mean = float(np.mean(magnitude_spectrum))
        
        return fft_mean
    
    def detect(self, face_img: np.ndarray) -> Dict[str, Any]:
        """
        Detecta indícios de deepfake na imagem da face.
        
        Orquestra as análises de bordas e espectro de frequências para identificar
        artefatos comuns em deepfakes. Combina os resultados para gerar uma predição
        final com nível de confiança.
        
        Args:
            face_img: Imagem da face como array numpy (BGR ou RGB).
            
        Returns:
            Dicionário contendo:
                - 'is_suspect': True se a imagem falhar nos testes (possível deepfake)
                - 'confidence': Valor de 0.0 a 1.0 indicando o nível de confiança
                - 'details': String descritiva com os resultados das análises
                
        Raises:
            ValueError: Se a imagem estiver vazia ou inválida.
            
        Examples:
            >>> detector = DeterministicDetector()
            >>> face_img = cv2.imread("face.jpg")
            >>> result = detector.detect(face_img)
            >>> if result['is_suspect']:
            ...     print(f"Possível deepfake: {result['details']}")
        """
        if face_img is None or face_img.size == 0:
            raise ValueError("Imagem vazia ou inválida não pode ser processada")
        
        try:
            # Realizar análises
            edge_variance = self._analyze_edges(face_img)
            fft_mean = self._analyze_fft(face_img)
            
            # Coletar indicadores de suspeita
            suspect_flags = []
            confidence_scores = []
            
            # Análise de bordas
            if edge_variance < self.EDGE_VARIANCE_THRESHOLD:
                suspect_flags.append("Baixa variância de borda (Blurry)")
                # Calcular confiança baseada na distância do threshold
                confidence = min(1.0, (self.EDGE_VARIANCE_THRESHOLD - edge_variance) / self.EDGE_VARIANCE_THRESHOLD)
                confidence_scores.append(confidence)
            elif edge_variance > self.EDGE_VARIANCE_THRESHOLD * 3:
                suspect_flags.append("Bordas excessivamente nítidas (Possível recorte artificial)")
                confidence = min(1.0, (edge_variance - self.EDGE_VARIANCE_THRESHOLD * 3) / (self.EDGE_VARIANCE_THRESHOLD * 3))
                confidence_scores.append(confidence)
            
            # Análise espectral
            if fft_mean < self.FFT_MEAN_THRESHOLD_LOW:
                suspect_flags.append("Espectro de frequências anômalo (Baixa frequência)")
                confidence = min(1.0, (self.FFT_MEAN_THRESHOLD_LOW - fft_mean) / self.FFT_MEAN_THRESHOLD_LOW)
                confidence_scores.append(confidence)
            elif fft_mean > self.FFT_MEAN_THRESHOLD_HIGH:
                suspect_flags.append("Espectro de frequências anômalo (Alta frequência)")
                confidence = min(1.0, (fft_mean - self.FFT_MEAN_THRESHOLD_HIGH) / self.FFT_MEAN_THRESHOLD_HIGH)
                confidence_scores.append(confidence)
            
            # Determinar resultado final
            is_suspect = len(suspect_flags) > 0
            
            # Calcular confiança geral (média das confianças individuais)
            if confidence_scores:
                overall_confidence = np.mean(confidence_scores)
            else:
                overall_confidence = 0.0
            
            # Construir mensagem de detalhes
            if is_suspect:
                details = f"Possível Deepfake: {', '.join(suspect_flags)}"
            else:
                details = "Nenhum artefato suspeito detectado nos testes determinísticos"
            
            return {
                'is_suspect': is_suspect,
                'confidence': float(overall_confidence),
                'details': details,
                'edge_variance': edge_variance,
                'fft_mean': fft_mean
            }
            
        except Exception as e:
            # Em caso de erro durante o processamento
            return {
                'is_suspect': False,
                'confidence': 0.0,
                'details': f"Erro durante análise: {str(e)}",
                'edge_variance': None,
                'fft_mean': None
            }


if __name__ == "__main__":
    """
    Teste básico para validar a classe DeterministicDetector.
    
    Cria uma imagem dummy com ruído aleatório e testa os métodos de detecção.
    """
    print("Testando DeterministicDetector...")
    print("-" * 60)
    
    try:
        # Criar detector
        detector = DeterministicDetector()
        print("✓ DeterministicDetector instanciado com sucesso")
        
        # Criar imagem dummy com ruído aleatório
        # Usar tamanho similar ao padrão de faces (224x224)
        dummy_img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        print(f"✓ Imagem dummy criada: {dummy_img.shape}")
        
        # Testar análise de bordas
        edge_variance = detector._analyze_edges(dummy_img)
        print(f"✓ Análise de bordas executada")
        print(f"  Variância do Laplaciano: {edge_variance:.2f}")
        print(f"  Threshold: {detector.EDGE_VARIANCE_THRESHOLD}")
        
        # Testar análise FFT
        fft_mean = detector._analyze_fft(dummy_img)
        print(f"✓ Análise FFT executada")
        print(f"  Média do espectro: {fft_mean:.2f}")
        print(f"  Threshold (low/high): {detector.FFT_MEAN_THRESHOLD_LOW}/{detector.FFT_MEAN_THRESHOLD_HIGH}")
        
        # Testar detecção completa
        result = detector.detect(dummy_img)
        print(f"\n✓ Detecção completa executada")
        print(f"\n{'=' * 60}")
        print("RESULTADO DA DETECÇÃO:")
        print(f"{'=' * 60}")
        print(f"Suspeito: {result['is_suspect']}")
        print(f"Confiança: {result['confidence']:.4f}")
        print(f"Detalhes: {result['details']}")
        print(f"Variância de borda: {result.get('edge_variance', 'N/A')}")
        print(f"Média FFT: {result.get('fft_mean', 'N/A')}")
        print(f"{'=' * 60}")
        
        print("\n✅ Todos os testes passaram!")
        
    except Exception as e:
        print(f"\n❌ Erro durante o teste: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        raise
