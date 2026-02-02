import streamlit as st
import cv2
import numpy as np
from PIL import Image
from sklearn.cluster import KMeans
import pandas as pd
import io

# -------------------------
# Funciones de filtros
# -------------------------
def apply_grayscale(image):
    """Convierte a escala de grises"""
    if len(image.shape) == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image

def apply_sketch(image):
    """Aplica efecto sketch"""
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    inv = 255 - gray
    blur = cv2.GaussianBlur(inv, (21, 21), 0)
    sketch = cv2.divide(gray, 255 - blur, scale=256)
    return sketch

def apply_sepia(image):
    """Aplica efecto sepia"""
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    
    kernel = np.array([[0.272, 0.534, 0.131],
                       [0.349, 0.686, 0.168],
                       [0.393, 0.769, 0.189]])
    sepia = cv2.transform(image, kernel)
    return np.clip(sepia, 0, 255).astype(np.uint8)

def apply_invert(image):
    """Invierte los colores"""
    return cv2.bitwise_not(image)

def apply_blur(image):
    """Aplica desenfoque gaussiano"""
    if len(image.shape) == 3:
        return cv2.GaussianBlur(image, (21, 21), 0)
    else:
        return cv2.GaussianBlur(image, (21, 21), 0)

def apply_color_reduction(image, n_colors=8, pencil=False):
    """Reduce la paleta de colores"""
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    
    img_data = image.reshape((-1, 3))
    kmeans = KMeans(n_clusters=n_colors, n_init=5, random_state=42)
    labels = kmeans.fit_predict(img_data)
    centers = np.uint8(kmeans.cluster_centers_)
    reduced = centers[labels].reshape(image.shape)

    if pencil:
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        edges = cv2.Canny(gray, 50, 150)
        edges = cv2.bitwise_not(edges)
        edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        reduced = cv2.addWeighted(reduced, 0.9, edges, 0.2, 0)

    return reduced, centers, labels

# -------------------------
# Función para aplicar múltiples filtros en el ORDEN DE SELECCIÓN
# -------------------------
def apply_filters_selection_order(image, selected_filters, selection_order, n_colors=8):
    """
    Aplica los filtros en el ORDEN en que fueron seleccionados por el usuario
    """
    current_image = image.copy()
    hex_colors = []
    applied_filters = []
    
    # Aplicar filtros en el orden de selección
    for filter_key in selection_order:
        if selected_filters.get(filter_key, False):
            
            if filter_key == 'grayscale':
                current_image = apply_grayscale(current_image)
                applied_filters.append("Escala de grises")
                
            elif filter_key == 'sketch':
                current_image = apply_sketch(current_image)
                applied_filters.append("Sketch")
                
            elif filter_key == 'sepia':
                current_image = apply_sepia(current_image)
                applied_filters.append("Sepia")
                
            elif filter_key == 'invert':
                current_image = apply_invert(current_image)
                applied_filters.append("Invertido")
                
            elif filter_key == 'blur':
                current_image = apply_blur(current_image)
                applied_filters.append("Blur")
                
            elif filter_key == 'color_reduce':
                if len(current_image.shape) == 2:
                    current_image = cv2.cvtColor(current_image, cv2.COLOR_GRAY2BGR)
                current_image, centers, labels = apply_color_reduction(current_image, n_colors, pencil=False)
                hex_colors = extract_colors(centers, labels)
                applied_filters.append("Reducir colores")
                
            elif filter_key == 'pencil_effect':
                if len(current_image.shape) == 2:
                    current_image = cv2.cvtColor(current_image, cv2.COLOR_GRAY2BGR)
                current_image, centers, labels = apply_color_reduction(current_image, n_colors, pencil=True)
                hex_colors = extract_colors(centers, labels)
                applied_filters.append("Reducir colores con lápiz")
    
    return current_image, hex_colors, applied_filters

# -------------------------
# Función para descargar imagen
# -------------------------
def create_download_button(image, filename="imagen_procesada.png"):
    """Crea un botón de descarga para la imagen procesada"""
    if len(image.shape) == 3:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image_rgb)
    else:
        pil_image = Image.fromarray(image)
    
    img_buffer = io.BytesIO()
    pil_image.save(img_buffer, format="PNG", quality=95)
    img_buffer.seek(0)
    
    st.download_button(
        label="📥 Descargar imagen procesada",
        data=img_buffer,
        file_name=filename,
        mime="image/png",
        use_container_width=True
    )

# -------------------------
# Utilidades
# -------------------------
def rgb_to_hex(color):
    return "#{:02x}{:02x}{:02x}".format(int(color[2]), int(color[1]), int(color[0]))

def extract_colors(centers, labels):
    unique, counts = np.unique(labels, return_counts=True)
    hex_colors = []
    for i, count in zip(unique, counts):
        hex_colors.append((rgb_to_hex(centers[i]), int(count)))
    return hex_colors

def count_unique_colors(image):
    """Cuenta la cantidad de colores únicos en una imagen"""
    if len(image.shape) == 3:
        pixels = image.reshape(-1, 3)
        unique_colors = np.unique(pixels, axis=0)
        return len(unique_colors)
    else:
        unique_values = np.unique(image)
        return len(unique_values)

def show_color_palette(hex_colors, title, mode="Tabla"):
    """Muestra la paleta de colores con título"""
    st.subheader(title)
    
    if mode == "Tabla":
        df = pd.DataFrame(hex_colors, columns=["Color HEX", "Pixeles"])
        st.dataframe(df, use_container_width=True, height=200)
    else:
        cols = st.columns(3)
        for i, (hex_code, count) in enumerate(hex_colors):
            with cols[i % 3]:
                st.markdown(
                    f"""
                    <div style='text-align:center; font-family:Arial;'>
                        <div style='width:100%; height:60px; background:{hex_code};
                                    border-radius:8px; border:1px solid #666;'></div>
                        <div style='margin-top:4px; font-family:monospace; font-size:11px;'>
                            {hex_code}<br><small>{count} px</small>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

def get_image_info(image, title, original_dimensions=None):
    """Obtiene información detallada de la imagen"""
    if len(image.shape) == 3:
        height, width, channels = image.shape
        color_mode = "Color"
    else:
        height, width = image.shape
        channels = 1
        color_mode = "Grises"
    
    unique_colors = count_unique_colors(image)
    
    info = {
        "title": title,
        "current_dimensions": f"{width} × {height}",
        "original_dimensions": original_dimensions if original_dimensions else f"{width} × {height}",
        "color_mode": color_mode,
        "channels": channels,
        "unique_colors": f"{unique_colors:,}",
        "size_kb": (image.nbytes / 1024) if hasattr(image, 'nbytes') else "N/A"
    }
    
    return info

def display_image_with_detailed_info(image, info, hex_colors=None, palette_mode="Tabla", show_download=False):
    """Muestra la imagen con información detallada compacta"""
    st.subheader(info["title"])
    
    if len(image.shape) == 3:
        st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), use_container_width=True)
    else:
        st.image(image, use_container_width=True, clamp=True)
    
    if show_download:
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"imagen_procesada_{timestamp}.png"
        create_download_button(image, filename)
    
    st.write("**📊 Información de la imagen:**")
    
    col1, col2, col3, col4 = st.columns([1, 1, 1, 1])
    
    with col1:
        st.markdown(f"**Original:**  \n`{info['original_dimensions']}`")
        st.markdown(f"**Actual:**  \n`{info['current_dimensions']}`")
    
    with col2:
        st.markdown(f"**Modo:**  \n`{info['color_mode']}`")
        st.markdown(f"**Canales:**  \n`{info['channels']}`")
    
    with col3:
        st.markdown(f"**Colores únicos:**  \n`{info['unique_colors']}`")
    
    with col4:
        if info['size_kb'] != "N/A":
            st.markdown(f"**Tamaño:**  \n`{info['size_kb']:.1f} KB`")
        else:
            st.markdown("**Tamaño:**  \n`N/A`")
    
    if hex_colors:
        show_color_palette(hex_colors, f"🎨 Paleta de colores principales ({len(hex_colors)} colores)", mode=palette_mode)

def extract_colors_from_image(image, n_colors=8):
    """Extrae los colores principales de una imagen usando K-means"""
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    
    n_colors = min(n_colors, 20)
    img_data = image.reshape((-1, 3))
    n_colors = min(n_colors, len(np.unique(img_data, axis=0)))
    
    if n_colors > 1:
        kmeans = KMeans(n_clusters=n_colors, n_init=5, random_state=42)
        labels = kmeans.fit_predict(img_data)
        centers = np.uint8(kmeans.cluster_centers_)
        return extract_colors(centers, labels)
    else:
        avg_color = np.mean(img_data, axis=0)
        return [(rgb_to_hex(avg_color), len(img_data))]

# -------------------------
# App
# -------------------------
st.title("🎨 Editor de Imágenes - Orden por Selección")

uploaded_file = st.file_uploader("Sube una imagen", type=["jpg", "jpeg", "png"])
if uploaded_file:
    img_pil = Image.open(uploaded_file)
    original_dimensions = f"{img_pil.width} × {img_pil.height}"
    img = np.array(img_pil)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    # Sidebar
    st.sidebar.header("🎛️ Configuración de Filtros")
    
    keep_ratio = st.sidebar.checkbox("Mantener proporción", value=True)
    width = st.sidebar.number_input("Ancho de visualización", min_value=2, value=min(img.shape[1], 800))
    height = st.sidebar.number_input("Alto de visualización", min_value=2, value=min(img.shape[0], 600))
    
    if keep_ratio:
        aspect_ratio = img.shape[0] / img.shape[1]
        height = int(width * aspect_ratio)

    # Inicializar estado de sesión para el orden de selección
    if 'selection_order' not in st.session_state:
        st.session_state.selection_order = []
    
    if 'filters' not in st.session_state:
        st.session_state.filters = {
            'grayscale': False,
            'sketch': False,
            'sepia': False,
            'invert': False,
            'blur': False,
            'color_reduce': False,
            'pencil_effect': False
        }

    st.sidebar.subheader("✅ Selecciona los filtros (orden de selección)")
    
    # Mapeo de nombres amigables
    filter_names = {
        'grayscale': "Escala de grises",
        'sketch': "Sketch", 
        'sepia': "Sepia",
        'invert': "Invertido",
        'blur': "Blur",
        'color_reduce': "Reducir colores",
        'pencil_effect': "Reducir colores con lápiz"
    }
    
    # Checkboxes que actualizan el orden de selección
    for filter_key, filter_name in filter_names.items():
        current_value = st.session_state.filters[filter_key]
        new_value = st.sidebar.checkbox(filter_name, value=current_value, key=f"checkbox_{filter_key}")
        
        # Si el estado cambió, actualizar el orden de selección
        if new_value != current_value:
            st.session_state.filters[filter_key] = new_value
            if new_value:
                # Agregar al orden de selección si se activa
                if filter_key not in st.session_state.selection_order:
                    st.session_state.selection_order.append(filter_key)
            else:
                # Remover del orden de selección si se desactiva
                if filter_key in st.session_state.selection_order:
                    st.session_state.selection_order.remove(filter_key)
            st.rerun()

    # Configuración adicional para reducción de colores
    n_colors = 8
    if st.session_state.filters['color_reduce'] or st.session_state.filters['pencil_effect']:
        n_colors = st.sidebar.number_input("Número de colores para reducción", min_value=2, value=8, step=1)
    
    palette_mode = st.sidebar.radio("Modo de visualización de paleta", ["Tabla", "Visual"])
    
    # Mostrar orden actual de selección
    st.sidebar.subheader("🔄 Orden de aplicación actual")
    if st.session_state.selection_order:
        for i, filter_key in enumerate(st.session_state.selection_order, 1):
            st.sidebar.write(f"{i}. {filter_names[filter_key]}")
    else:
        st.sidebar.info("Aún no has seleccionado filtros")
    
    # Botón para limpiar todos los filtros
    if st.sidebar.button("🧹 Limpiar todos los filtros"):
        for key in st.session_state.filters:
            st.session_state.filters[key] = False
        st.session_state.selection_order = []
        st.rerun()

    # Redimensionar imagen
    img_resized = cv2.resize(img, (int(width), int(height)))
    
    # Verificar si hay filtros seleccionados
    any_filter_selected = any(st.session_state.filters.values())
    
    # Extraer colores de la imagen original
    original_hex_colors = extract_colors_from_image(img_resized, n_colors=8)
    
    if any_filter_selected and st.session_state.selection_order:
        # Aplicar filtros en el ORDEN DE SELECCIÓN
        processed, processed_hex_colors, applied_filters = apply_filters_selection_order(
            img_resized, st.session_state.filters, st.session_state.selection_order, n_colors
        )
    else:
        processed = img_resized
        processed_hex_colors = original_hex_colors
        applied_filters = []

    # Mostrar imágenes
    st.subheader("🖼️ Análisis Comparativo de Imágenes")
    
    col1, col2 = st.columns(2)
    
    with col1:
        original_info = get_image_info(img_resized, "Imagen Original", original_dimensions)
        display_image_with_detailed_info(img_resized, original_info, original_hex_colors, palette_mode, show_download=False)
    
    with col2:
        if any_filter_selected and st.session_state.selection_order:
            processed_info = get_image_info(processed, "Imagen Procesada", original_dimensions)
            display_image_with_detailed_info(processed, processed_info, processed_hex_colors, palette_mode, show_download=True)
            
            st.success(f"**✅ Filtros aplicados en orden de selección:**")
            for i, filter_key in enumerate(st.session_state.selection_order, 1):
                if st.session_state.filters[filter_key]:
                    st.write(f"{i}. {filter_names[filter_key]}")
            
            st.info("💡 **El orden importa:** Los filtros se aplican en el orden en que los seleccionaste")
        else:
            processed_info = get_image_info(processed, "Imagen Original (sin filtros)", original_dimensions)
            display_image_with_detailed_info(processed, processed_info, original_hex_colors, palette_mode, show_download=False)
            st.info("💡 Selecciona filtros en el orden que desees aplicarlos")

    # Estadísticas comparativas
    if any_filter_selected and st.session_state.selection_order:
        st.subheader("📈 Resumen de Cambios")
        
        stats_col1, stats_col2, stats_col3, stats_col4 = st.columns(4)
        
        original_colors = count_unique_colors(img_resized)
        processed_colors = count_unique_colors(processed)
        
        with stats_col1:
            color_change = processed_colors - original_colors
            st.metric("Cambio en colores", f"{processed_colors:,}", delta=f"{color_change:+,}")
        
        with stats_col2:
            original_pixels = img_resized.shape[0] * img_resized.shape[1]
            processed_pixels = processed.shape[0] * processed.shape[1]
            st.metric("Píxeles totales", f"{processed_pixels:,}")
        
        with stats_col3:
            if len(processed.shape) == 3:
                processed_channels = processed.shape[2]
            else:
                processed_channels = 1
            st.metric("Canales", processed_channels)
        
        with stats_col4:
            if hasattr(img_resized, 'nbytes') and hasattr(processed, 'nbytes'):
                size_diff = (processed.nbytes - img_resized.nbytes) / 1024
                st.metric("Tamaño", f"{size_diff:+.1f} KB")

else:
    st.info("👆 Sube una imagen para comenzar el análisis")