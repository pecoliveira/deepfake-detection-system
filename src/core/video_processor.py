"""
Módulo para processamento de vídeo.

Fornece classes e métodos para carregar e processar arquivos de vídeo.
"""

from pathlib import Path
from typing import Any, Iterator, Tuple

import cv2
import numpy as np


class VideoLoader:
    """
    Classe para carregar e processar arquivos de vídeo.
    
    Esta classe fornece métodos estáticos para carregar vídeos e iterar
    sobre seus frames usando OpenCV. Segue princípios de Clean Code e SOLID,
    mantendo a responsabilidade única de processamento de vídeo.
    
    Attributes:
        None (classe com métodos estáticos)
    
    Examples:
        >>> video_path = "data/input/video.mp4"
        >>> for frame, frame_number in VideoLoader.load_video(video_path):
        ...     # Processar frame
        ...     pass
    """
    
    @staticmethod
    def load_video(video_path: str | Path) -> Iterator[Tuple[np.ndarray, int]]:
        """
        Carrega um vídeo e retorna um gerador de frames.
        
        Abre um arquivo de vídeo e itera sobre seus frames, retornando cada
        frame como um array numpy junto com seu número de índice.
        
        Args:
            video_path: Caminho para o arquivo de vídeo. Pode ser string ou Path.
            
        Yields:
            Tupla contendo:
                - frame: Array numpy (BGR) representando o frame do vídeo
                - frame_number: Número do frame (índice baseado em zero)
                
        Raises:
            FileNotFoundError: Se o arquivo de vídeo não for encontrado.
            ValueError: Se o arquivo não for um vídeo válido ou não puder ser aberto.
            
        Examples:
            >>> for frame, frame_num in VideoLoader.load_video("video.mp4"):
            ...     cv2.imshow("Frame", frame)
            ...     if cv2.waitKey(1) & 0xFF == ord('q'):
            ...         break
        """
        video_path = Path(video_path)
        
        if not video_path.exists():
            raise FileNotFoundError(f"Arquivo de vídeo não encontrado: {video_path}")
        
        if not video_path.is_file():
            raise ValueError(f"O caminho fornecido não é um arquivo: {video_path}")
        
        cap = cv2.VideoCapture(str(video_path))
        
        if not cap.isOpened():
            raise ValueError(f"Não foi possível abrir o vídeo: {video_path}")
        
        try:
            frame_number = 0
            while True:
                ret, frame = cap.read()
                
                if not ret:
                    break
                
                yield frame, frame_number
                frame_number += 1
                
        finally:
            cap.release()
    
    @staticmethod
    def get_video_properties(video_path: str | Path) -> dict[str, Any]:
        """
        Obtém propriedades do vídeo sem carregar todos os frames.
        
        Extrai informações como largura, altura, FPS e número total de frames
        do arquivo de vídeo.
        
        Args:
            video_path: Caminho para o arquivo de vídeo. Pode ser string ou Path.
            
        Returns:
            Dicionário contendo:
                - width: Largura do vídeo em pixels
                - height: Altura do vídeo em pixels
                - fps: Taxa de frames por segundo
                - frame_count: Número total de frames
                - duration: Duração do vídeo em segundos
                
        Raises:
            FileNotFoundError: Se o arquivo de vídeo não for encontrado.
            ValueError: Se o arquivo não for um vídeo válido ou não puder ser aberto.
            
        Examples:
            >>> props = VideoLoader.get_video_properties("video.mp4")
            >>> print(f"Resolução: {props['width']}x{props['height']}")
            >>> print(f"FPS: {props['fps']}")
        """
        video_path = Path(video_path)
        
        if not video_path.exists():
            raise FileNotFoundError(f"Arquivo de vídeo não encontrado: {video_path}")
        
        cap = cv2.VideoCapture(str(video_path))
        
        if not cap.isOpened():
            raise ValueError(f"Não foi possível abrir o vídeo: {video_path}")
        
        try:
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = frame_count / fps if fps > 0 else 0.0
            
            return {
                "width": width,
                "height": height,
                "fps": fps,
                "frame_count": frame_count,
                "duration": duration
            }
        finally:
            cap.release()
    
    @staticmethod
    def save_frame(frame: np.ndarray, output_path: str | Path) -> None:
        """
        Salva um frame como imagem.
        
        Salva um frame (array numpy) como arquivo de imagem no disco.
        
        Args:
            frame: Array numpy representando o frame (BGR).
            output_path: Caminho onde o frame será salvo.
            
        Raises:
            ValueError: Se o frame estiver vazio ou o formato não for suportado.
            
        Examples:
            >>> frame, _ = next(VideoLoader.load_video("video.mp4"))
            >>> VideoLoader.save_frame(frame, "output/frame_0.jpg")
        """
        if frame is None or frame.size == 0:
            raise ValueError("Frame vazio não pode ser salvo")
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        success = cv2.imwrite(str(output_path), frame)
        
        if not success:
            raise ValueError(f"Não foi possível salvar o frame em: {output_path}")

