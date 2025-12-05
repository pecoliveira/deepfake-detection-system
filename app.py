
import json
import logging
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
import streamlit as st

from src.core.face_extractor import FaceExtractor
from src.core.video_processor import VideoLoader
from src.detectors.ai_model import DeepFakeClassifier
from src.detectors.deterministic import DeterministicDetector

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuração da página
st.set_page_config(
    page_title="Detector de Deepfakes",
    page_icon="🎭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Constantes
SUPPORTED_FORMATS = [".mp4", ".avi", ".mov"]
FRAME_SKIP_OPTIONS = {
    "Cada frame": 1,
    "A cada 5 frames": 5,
    "A cada 10 frames": 10,
    "A cada 20 frames": 20
}


def initialize_session_state() -> None:
    """Inicializa variáveis de sessão do Streamlit."""
    if "processors_initialized" not in st.session_state:
        st.session_state.processors_initialized = False
    if "processing_results" not in st.session_state:
        st.session_state.processing_results = None
    if "video_path" not in st.session_state:
        st.session_state.video_path = None


def load_video_labels() -> Dict[str, str]:
    """
    Carrega o mapeamento de labels de vídeos do arquivo JSON.
    
    Returns:
        Dicionário mapeando nomes de arquivos para labels ('fake' ou 'real').
    """
    labels_file = Path("video_labels.json")
    
    if not labels_file.exists():
        # Criar arquivo padrão se não existir
        default_labels = {
            "video_labels": {},
            "description": "Mapeamento de nomes de arquivos para labels conhecidos. Use 'fake' ou 'real'."
        }
        with open(labels_file, 'w', encoding='utf-8') as f:
            json.dump(default_labels, f, indent=4, ensure_ascii=False)
        return {}
    
    try:
        with open(labels_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            return data.get("video_labels", {})
    except Exception as e:
        logger.error(f"Erro ao carregar labels de vídeo: {e}")
        return {}


def save_video_labels(labels: Dict[str, str]) -> None:
    """
    Salva o mapeamento de labels de vídeos no arquivo JSON.
    
    Args:
        labels: Dicionário mapeando nomes de arquivos para labels.
    """
    labels_file = Path("video_labels.json")
    
    try:
        data = {
            "video_labels": labels,
            "description": "Mapeamento de nomes de arquivos para labels conhecidos. Use 'fake' ou 'real'."
        }
        with open(labels_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
        logger.info(f"Labels de vídeo salvos em: {labels_file}")
    except Exception as e:
        logger.error(f"Erro ao salvar labels de vídeo: {e}")


def get_video_label(filename: str, labels: Dict[str, str]) -> Optional[str]:
    """
    Obtém o label conhecido de um vídeo pelo nome do arquivo.
    
    Args:
        filename: Nome do arquivo de vídeo.
        labels: Dicionário de labels conhecidos.
        
    Returns:
        'fake', 'real' ou None se não houver label conhecido.
    """
    # Tentar busca exata
    if filename in labels:
        return labels[filename].lower()
    
    # Tentar busca parcial (sem extensão)
    filename_no_ext = Path(filename).stem
    if filename_no_ext in labels:
        return labels[filename_no_ext].lower()
    
    # Tentar busca case-insensitive
    for key, value in labels.items():
        if key.lower() == filename.lower() or Path(key).stem.lower() == filename_no_ext.lower():
            return value.lower()
    
    return None


def initialize_processors() -> Tuple[FaceExtractor, DeterministicDetector, DeepFakeClassifier]:
    """
    Inicializa e retorna as classes de processamento.
    
    Returns:
        Tupla contendo (FaceExtractor, DeterministicDetector, DeepFakeClassifier)
    """
    logger.info("Inicializando processadores...")
    
    face_extractor = FaceExtractor()
    deterministic_detector = DeterministicDetector()
    
    # Tentar carregar pesos do modelo de IA, se existirem
    weights_path = Path("models") / "deepfake_classifier.h5"
    if weights_path.exists():
        ai_classifier = DeepFakeClassifier(weights_path=str(weights_path))
    else:
        ai_classifier = DeepFakeClassifier()
    
    return face_extractor, deterministic_detector, ai_classifier


def save_uploaded_file(uploaded_file) -> Path:
    """
    Salva arquivo enviado em arquivo temporário.
    
    Args:
        uploaded_file: Arquivo enviado pelo Streamlit.
        
    Returns:
        Caminho do arquivo temporário salvo.
    """
    with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as tmp_file:
        tmp_file.write(uploaded_file.read())
        return Path(tmp_file.name)


def process_video(
    video_path: Path,
    face_extractor: FaceExtractor,
    deterministic_detector: DeterministicDetector,
    ai_classifier: DeepFakeClassifier,
    progress_bar: Any,
    frame_skip: int = 5,
    ai_weight: float = 0.1
) -> Dict:
    """
    Processa vídeo frame por frame, detectando faces e analisando deepfakes.
    
    Args:
        video_path: Caminho do arquivo de vídeo.
        face_extractor: Extrator de faces.
        deterministic_detector: Detector determinístico.
        ai_classifier: Classificador de IA.
        progress_bar: Barra de progresso do Streamlit.
        frame_skip: Processar a cada N frames (padrão: 5).
        
    Returns:
        Dicionário com resultados do processamento.
    """
    results = {
        "total_frames": 0,
        "processed_frames": 0,
        "frames_with_faces": 0,
        "frames_suspect": 0,
        "probabilities": [],
        "deterministic_results": [],
        "frame_numbers": [],
        "confidence_scores": [],
        "details_list": [],
        "ai_weight": ai_weight
    }
    
    try:
        # Obter propriedades do vídeo
        video_props = VideoLoader.get_video_properties(video_path)
        results["total_frames"] = video_props["frame_count"]
        
        logger.info(f"Processando vídeo: {video_path.name}")
        logger.info(f"Total de frames: {results['total_frames']}")
        logger.info(f"Processando a cada {frame_skip} frame(s)")
        
        frame_count = 0
        processed_count = 0
        
        for frame, frame_number in VideoLoader.load_video(video_path):
            frame_count += 1
            
            # Processar apenas a cada N frames
            if frame_number % frame_skip != 0:
                continue
            
            processed_count += 1
            
            # Atualizar barra de progresso
            progress = frame_count / results["total_frames"]
            progress_bar.progress(min(progress, 1.0))
            
            # Detectar faces no frame (com filtro de qualidade)
            faces = face_extractor.detect_faces(frame, min_face_size=50)
            
            if not faces:
                continue
            
            results["frames_with_faces"] += 1
            
            # Processar cada face detectada (limitar a maior face se houver múltiplas)
            # Ordenar por área (w * h) e pegar a maior
            faces_sorted = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
            
            # Processar apenas a maior face por frame (evita processamento excessivo)
            for face_coords in faces_sorted[:1]:  # Apenas a maior face
                try:
                    # Extrair face
                    face_img = face_extractor.extract_face(frame, face_coords)
                    
                    # Análise determinística
                    det_result = deterministic_detector.detect(face_img)
                    
                    # Análise de IA
                    ai_probability = ai_classifier.predict(face_img)
                    
                    # Calcular pontuação combinada
                    combined_score = calculate_combined_score(ai_probability, det_result, ai_weight)
                    
                    # Armazenar resultados
                    results["probabilities"].append(ai_probability)
                    results["confidence_scores"].append(det_result.get("confidence", 0.0))
                    results["deterministic_results"].append(det_result["is_suspect"])
                    results["frame_numbers"].append(frame_number)
                    results["details_list"].append(det_result.get("details", ""))
                    
                    # Contar como suspeito se ambos indicarem
                    if det_result["is_suspect"] or ai_probability > 0.7:
                        results["frames_suspect"] += 1
                    
                except Exception as e:
                    logger.error(f"Erro ao processar face no frame {frame_number}: {e}")
                    continue
            
            results["processed_frames"] = processed_count
        
        progress_bar.progress(1.0)
        logger.info(f"Processamento concluído: {results['processed_frames']} frames processados")
        
    except Exception as e:
        logger.error(f"Erro durante processamento do vídeo: {e}")
        st.error(f"Erro ao processar vídeo: {str(e)}")
        raise
    
    return results


def calculate_combined_score(ai_probability: float, det_result: Dict, ai_weight: float = 0.1) -> float:
    """
    Calcula pontuação combinada entre IA e detector determinístico.
    
    Args:
        ai_probability: Probabilidade de ser fake da IA (0.0-1.0).
        det_result: Resultado do detector determinístico.
        ai_weight: Peso da IA na decisão final (0.0-1.0). Padrão: 0.1.
        
    Returns:
        Pontuação combinada (0.0-1.0).
    """
    det_weight = 1.0 - ai_weight
    
    # Obter score do detector determinístico
    if det_result["is_suspect"]:
        det_score = det_result.get("confidence", 0.5)
    else:
        det_score = 0.0
    
    # Calcular score final combinado
    final_score = (ai_probability * ai_weight) + (det_score * det_weight)
    
    return min(1.0, max(0.0, final_score))


def display_results(results: Dict, ai_trained: bool, ai_weight: float = 0.1) -> None:
    """
    Exibe resultados do processamento em formato visual.
    
    Args:
        results: Dicionário com resultados do processamento.
        ai_trained: Indica se a IA foi treinada.
        ai_weight: Peso da IA usado na decisão (0.0-1.0).
    """
    st.header("📊 Resultados da Análise")
    
    if not results.get("probabilities") or len(results.get("probabilities", [])) == 0:
        st.info("Nenhuma face foi detectada no vídeo durante o processamento. Tente processar novamente ou use outro vídeo.")
        st.write(f"- **Total de frames no vídeo:** {results.get('total_frames', 0)}")
        st.write(f"- **Frames processados:** {results.get('processed_frames', 0)}")
        st.write(f"- **Frames com rostos detectados:** {results.get('frames_with_faces', 0)}")
        return
    
    # Calcular estatísticas melhoradas
    avg_ai_prob = np.mean(results["probabilities"]) if results["probabilities"] else 0.0
    std_ai_prob = np.std(results["probabilities"]) if len(results["probabilities"]) > 1 else 0.0
    avg_confidence = np.mean(results["confidence_scores"]) if results["confidence_scores"] else 0.0
    reality_probability = (1.0 - avg_ai_prob) * 100
    
    # Calcular confiança combinada
    combined_confidence = (reality_probability / 100) * (1 - std_ai_prob) if std_ai_prob > 0 else reality_probability / 100
    
    # KPIs - Métricas principais
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        # Probabilidade de Realidade (melhorado)
        color = "normal" if reality_probability > 50 else "inverse"
        confidence_level = "Alta" if std_ai_prob < 0.15 else "Média" if std_ai_prob < 0.25 else "Baixa"
        
        st.metric(
            label="🎯 Probabilidade de Realidade",
            value=f"{reality_probability:.1f}%",
            delta=f"{reality_probability - 50:+.1f}%"
        )
        if reality_probability < 50:
            st.caption(f"⚠️ Vídeo suspeito ({confidence_level} confiança)")
        else:
            st.caption(f"✅ Vídeo autêntico ({confidence_level} confiança)")
    
    with col2:
        # Nível de Confiança da IA
        confidence_pct = avg_confidence * 100 if ai_trained else 0.0
        st.metric(
            label="🤖 Confiança da IA",
            value=f"{confidence_pct:.1f}%",
            delta=None
        )
    
    with col3:
        # Frames com rostos
        st.metric(
            label="👤 Frames com Rostos",
            value=results["frames_with_faces"],
            delta=f"{results['processed_frames']} processados"
        )
    
    with col4:
        # Frames suspeitos
        suspect_pct = (results["frames_suspect"] / results["frames_with_faces"] * 100) if results["frames_with_faces"] > 0 else 0
        st.metric(
            label="🚨 Frames Suspeitos",
            value=results["frames_suspect"],
            delta=f"{suspect_pct:.1f}% do total"
        )
    
    # Mostrar qual método está dominando a decisão
    det_weight = 1.0 - ai_weight
    if ai_weight > 0.5:
        dominant_method = "Inteligência Artificial"
        dominant_percent = ai_weight * 100
    elif det_weight > 0.5:
        dominant_method = "Análise de Bordas e Espectro de Frequências (Determinístico)"
        dominant_percent = det_weight * 100
    else:
        dominant_method = "Combinação Equilibrada (IA e Determinístico)"
        dominant_percent = 50.0
    
    st.info(f"📊 **Decisão baseada majoritariamente em:** {dominant_method} ({dominant_percent:.0f}%)")
    
    st.divider()
    
    # Gráfico de variação da confiança
    if len(results["probabilities"]) > 1:
        st.subheader("📈 Variação da Probabilidade ao Longo do Tempo")
        
        # Criar DataFrame para o gráfico com suavização para melhor visualização
        probabilities = [p * 100 for p in results["probabilities"]]
        
        # Aplicar média móvel simples para suavizar (janela de 3)
        smoothed_probs = []
        window_size = min(3, len(probabilities))
        for i in range(len(probabilities)):
            start_idx = max(0, i - window_size // 2)
            end_idx = min(len(probabilities), i + window_size // 2 + 1)
            smoothed_probs.append(np.mean(probabilities[start_idx:end_idx]))
        
        chart_data = pd.DataFrame({
            "Frame": results["frame_numbers"],
            "Probabilidade de Fake (%)": probabilities,
            "Suavizado": smoothed_probs,
            "Linha de Corte (50%)": [50] * len(results["frame_numbers"])
        })
        
        st.line_chart(
            chart_data,
            x="Frame",
            y=["Probabilidade de Fake (%)", "Suavizado", "Linha de Corte (50%)"]
        )
        
        st.caption("Gráfico mostrando a probabilidade de ser fake ao longo dos frames processados (linha suavizada para melhor visualização).")
    
    # Detalhes dos indícios visuais (melhorado)
    if results["details_list"]:
        st.subheader("🔍 Indícios Visuais Detectados")
        
        # Agrupar detalhes únicos e contar frequência
        details_count = {}
        for detail in results["details_list"]:
            if detail:
                details_count[detail] = details_count.get(detail, 0) + 1
        
        # Ordenar por frequência
        sorted_details = sorted(details_count.items(), key=lambda x: x[1], reverse=True)
        
        if sorted_details:
            for detail, count in sorted_details:
                frequency_pct = (count / len(results["details_list"])) * 100
                
                if "Possível Deepfake" in detail:
                    st.error(f"❌ **{detail}** (aparece em {frequency_pct:.1f}% das análises)")
                elif "Nenhum artefato" in detail:
                    st.success(f"✅ **{detail}** (aparece em {frequency_pct:.1f}% das análises)")
                else:
                    st.info(f"ℹ️ **{detail}** (aparece em {frequency_pct:.1f}% das análises)")
        else:
            st.info("Nenhum indício visual significativo foi detectado.")
    
    # Estatísticas detalhadas
    with st.expander("📋 Estatísticas Detalhadas"):
        st.write(f"- **Total de frames no vídeo:** {results['total_frames']}")
        st.write(f"- **Frames processados:** {results['processed_frames']}")
        st.write(f"- **Taxa de processamento:** {(results['processed_frames'] / results['total_frames'] * 100) if results['total_frames'] > 0 else 0:.1f}%")
        st.write(f"- **Frames com rostos detectados:** {results['frames_with_faces']}")
        st.write(f"- **Frames classificados como suspeitos:** {results['frames_suspect']}")
        st.write(f"- **Taxa de suspeitos:** {(results['frames_suspect'] / results['frames_with_faces'] * 100) if results['frames_with_faces'] > 0 else 0:.1f}%")
        st.write(f"- **Probabilidade média de fake (IA):** {avg_ai_prob * 100:.2f}%")
        st.write(f"- **Desvio padrão (IA):** {std_ai_prob * 100:.2f}%")
        st.write(f"- **Confiança média (Determinístico):** {avg_confidence * 100:.2f}%")
        st.write(f"- **Probabilidade de realidade:** {reality_probability:.2f}%")


def main():
    """Função principal da aplicação Streamlit."""
    # Inicializar estado da sessão
    initialize_session_state()
    
    # Título principal
    st.title("🎭 Detector de Deepfakes - UNDB 4.0")
    st.markdown("---")
    st.markdown(
        "Sistema híbrido de detecção de deepfakes combinando análise visual determinística "
        "e inteligência artificial para identificar vídeos manipulados."
    )
    
    # Sidebar - Configurações
    with st.sidebar:
        st.header("⚙️ Configurações")
        
        frame_skip_option = st.selectbox(
            "Frequência de processamento:",
            options=list(FRAME_SKIP_OPTIONS.keys()),
            index=2,  # Default: A cada 10 frames
            help="Processar mais frames = mais preciso, mas mais lento"
        )
        frame_skip = FRAME_SKIP_OPTIONS[frame_skip_option]
        
        # Slider para peso da decisão híbrida
        ai_weight = st.slider(
            "Peso da Decisão (Híbrido)",
            min_value=0.0,
            max_value=1.0,
            value=0.1,
            step=0.05,
            help="Define o quanto confiamos na IA versus nos Métodos Clássicos. 0.0 = 100% Determinístico, 1.0 = 100% IA"
        )
        
        # Mostrar valores atuais
        det_weight = 1.0 - ai_weight
        st.caption(f"**Atual:** {ai_weight*100:.0f}% IA | {det_weight*100:.0f}% Determinístico")
        
        st.divider()
        
        st.markdown("### 📝 Sobre")
        st.markdown(
            """
            Este sistema utiliza:
            - **Detector Determinístico**: Análise de bordas e espectro de frequências
            - **Classificador de IA**: MobileNetV2 com transfer learning
            
            Os resultados são indicativos e devem ser validados por especialistas.
            """
        )
    
    # Upload de vídeo
    st.header("📁 Upload de Vídeo")
    uploaded_file = st.file_uploader(
        "Selecione um arquivo de vídeo:",
        type=["mp4", "avi", "mov"],
        help="Formatos suportados: MP4, AVI, MOV"
    )
    
    if uploaded_file is not None:
        # Salvar arquivo temporário
        if st.session_state.video_path is None or st.session_state.video_path.name != uploaded_file.name:
            st.session_state.video_path = save_uploaded_file(uploaded_file)
            st.session_state.processing_results = None
            st.session_state.processors_initialized = False
        
        # Mostrar informações do vídeo
        st.success(f"✅ Arquivo carregado: **{uploaded_file.name}**")
        
        try:
            video_props = VideoLoader.get_video_properties(st.session_state.video_path)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Resolução", f"{video_props['width']}x{video_props['height']}")
            with col2:
                st.metric("FPS", f"{video_props['fps']:.2f}")
            with col3:
                st.metric("Duração", f"{video_props['duration']:.2f}s")
        except Exception as e:
            st.error(f"Erro ao ler propriedades do vídeo: {e}")
        
        # Botão de processamento
        if st.button("🚀 Iniciar Análise", type="primary", use_container_width=True):
            # Inicializar processadores
            if not st.session_state.processors_initialized:
                with st.spinner("Inicializando processadores..."):
                    try:
                        st.session_state.face_extractor, st.session_state.det_detector, st.session_state.ai_classifier = initialize_processors()
                        st.session_state.processors_initialized = True
                        st.session_state.ai_trained = st.session_state.ai_classifier.is_trained
                    except Exception as e:
                        st.error(f"Erro ao inicializar processadores: {e}")
                        logger.error(f"Erro na inicialização: {e}")
                        st.stop()
            
            # Processar vídeo
            st.header("🔄 Processando Vídeo...")
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            try:
                status_text.text("Detectando faces e analisando frames...")
                
                results = process_video(
                    video_path=st.session_state.video_path,
                    face_extractor=st.session_state.face_extractor,
                    deterministic_detector=st.session_state.det_detector,
                    ai_classifier=st.session_state.ai_classifier,
                    frame_skip=frame_skip,
                    progress_bar=progress_bar,
                    ai_weight=ai_weight
                )
                
                status_text.text("✅ Processamento concluído!")
                st.session_state.processing_results = results
                
                # Pequeno delay para mostrar mensagem de sucesso
                import time
                time.sleep(0.5)
                status_text.empty()
                progress_bar.empty()
                
                st.balloons()
                
            except Exception as e:
                st.error(f"Erro durante processamento: {e}")
                logger.error(f"Erro no processamento: {e}", exc_info=True)
        
        # Exibir resultados se disponíveis (depois do processamento)
        if st.session_state.processing_results is not None:
            ai_trained = getattr(st.session_state, "ai_trained", False)
            used_ai_weight = st.session_state.processing_results.get("ai_weight", ai_weight)
            display_results(st.session_state.processing_results, ai_trained, used_ai_weight)
    
    else:
        st.info("👆 Por favor, faça upload de um arquivo de vídeo para começar a análise.")
        
        # Mostrar instruções
        with st.expander("📖 Como usar"):
            st.markdown(
                """
                1. **Faça upload** de um arquivo de vídeo (MP4, AVI ou MOV)
                2. Configure a **frequência de processamento** na barra lateral
                3. Clique em **"Iniciar Análise"**
                4. Aguarde o processamento (pode levar alguns minutos)
                5. Visualize os **resultados** e estatísticas
                
                **Dicas:**
                - Vídeos menores processam mais rápido
                - Processar todos os frames é mais preciso, mas mais lento
                - Os resultados são indicativos, não definitivos
                """
            )


if __name__ == "__main__":
    main()

