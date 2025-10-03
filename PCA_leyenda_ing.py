# pip install PyPDF2 pandas numpy matplotlib scikit-learn

import os
import re
import csv
import textwrap
from collections import Counter
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
from PyPDF2 import PdfReader

# --- STOPWORDS ---
STOPWORDS_ES = {
    'de', 'la', 'que', 'el', 'en', 'y', 'a', 'los', 'del', 'se', 'las', 'por', 'un', 'para', 'con', 'no', 'una', 'su', 'al',
    'lo', 'como', 'más', 'pero', 'sus', 'le', 'ya', 'o', 'este', 'sí', 'porque', 'esta', 'entre', 'cuando', 'muy', 'sin',
    'sobre', 'también', 'me', 'hasta', 'hay', 'donde', 'quien', 'desde', 'todo', 'nos', 'durante', 'todos', 'uno', 'les',
    'ni', 'contra', 'otros', 'ese', 'eso', 'ante', 'ellos', 'e', 'esto', 'mí', 'antes', 'algunos', 'qué', 'unos', 'yo',
    'otro', 'otras', 'otra', 'él', 'tanto', 'esa', 'estos', 'mucho', 'quienes', 'nada', 'muchos', 'cual', 'poco', 'ella',
    'estar', 'estas', 'algunas', 'algo', 'nosotros', 'mi', 'mis', 'tú', 'te', 'ti', 'tu', 'tus', 'ellas', 'nosotras',
    'vosotros', 'vosotras', 'os', 'mío', 'mía', 'míos', 'mías', 'tuyo', 'tuya', 'tuyos', 'tuyas', 'suyo', 'suya',
    'suyos', 'suyas', 'nuestro', 'nuestra', 'nuestros', 'nuestras', 'vuestro', 'vuestra', 'vuestros', 'vuestras',
    'esos', 'esas', 'ser', 'ir', 'tener', 'hacer', 'poder', 'decir', 'ver', 'dar', 'saber', 'querer', 'llegar',
    'pasar', 'deber', 'sentir', 'seguir', 'encontrar', 'llevar', 'dejar', 'volver', 'venir', 'parecer', 'creer',
    'hablar', 'necesitar', 'entender', 'trabajar', 'empezar', 'utilizar', 'llamar', 'pensar', 'esperar', 'conocer'
}

STOPWORDS_EN = {
    'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll", "you'd", 'your',
    'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it',
    "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this',
    'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had',
    'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while',
    'of', 'at', 'by', 'for', 'with', 'through', 'during', 'before', 'after', 'above', 'below', 'up', 'down', 'in', 'out',
    'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how',
    'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own',
    'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now',
    'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn',
    "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't",
    'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren',
    "weren't", 'won', "won't", 'wouldn', "wouldn't","from","one","may","also","many","into","could","would","about",
    "journal","used","however","use","even","review","citation","download","shrink","ieee"
}

TODAS_STOPWORDS = STOPWORDS_ES | STOPWORDS_EN

def limpiar_texto(texto):
    texto = texto.lower()
    texto = re.sub(r'[^a-záéíóúüñ\s]', ' ', texto)
    texto = re.sub(r'\s+', ' ', texto).strip()
    return texto

def leer_pdf(ruta):
    try:
        reader = PdfReader(ruta)
        texto = ""
        for page in reader.pages:
            extracted = page.extract_text()
            if extracted:
                texto += extracted
        return texto
    except Exception as e:
        print(f"⚠️ Error al leer PDF {ruta}: {e}")
        return ""

def leer_txt_csv(ruta):
    try:
        if ruta.lower().endswith('.csv'):
            with open(ruta, 'r', encoding='utf-8', errors='ignore') as f:
                reader = csv.reader(f)
                return ' '.join(' '.join(row) for row in reader)
        else:
            with open(ruta, 'r', encoding='utf-8', errors='ignore') as f:
                return f.read()
    except Exception as e:
        print(f"⚠️ Error al leer {ruta}: {e}")
        return ""

def main():
    print("📊 Análisis PCA de palabras en archivos (TXT, CSV, PDF)")
    carpeta = input("Ruta de la carpeta: ").strip()
    while not os.path.isdir(carpeta):
        print("❌ Carpeta no válida.")
        carpeta = input("Ruta de la carpeta: ").strip()

    todos_los_textos = []
    total_archivos = 0

    for dirpath, _, files in os.walk(carpeta):
        for nombre in files:
            ruta = os.path.join(dirpath, nombre)
            if nombre.lower().endswith('.pdf'):
                texto = leer_pdf(ruta)
                total_archivos += 1
                print(f"📄 {ruta}")
            elif nombre.lower().endswith(('.txt', '.csv')):
                texto = leer_txt_csv(ruta)
                total_archivos += 1
                print(f"📄 {ruta}")
            else:
                continue
            if texto.strip():
                todos_los_textos.append(limpiar_texto(texto))

    if not todos_los_textos:
        print("❌ No se encontró texto en los archivos.")
        return

    # Combinar y procesar
    texto_global = ' '.join(todos_los_textos)
    palabras = [p for p in texto_global.split() if p not in TODAS_STOPWORDS and len(p) > 2]
    if not palabras:
        print("❌ No hay palabras significativas tras filtrar stopwords.")
        return

    # Top palabras
    top_n = 50
    contador = Counter(palabras)
    palabras_top = [palabra for palabra, _ in contador.most_common(top_n)]

    # Matriz de co-ocurrencia (ventana = 5)
    index = {w: i for i, w in enumerate(palabras_top)}
    cooc = np.zeros((len(palabras_top), len(palabras_top)))
    for i, palabra in enumerate(palabras):
        if palabra not in index:
            continue
        idx1 = index[palabra]
        start = max(0, i - 5)
        end = min(len(palabras), i + 6)
        for j in range(start, end):
            if i == j:
                continue
            vecina = palabras[j]
            if vecina in index:
                idx2 = index[vecina]
                cooc[idx1, idx2] += 1

    # Normalizar y aplicar PCA
    cooc_norm = normalize(cooc, norm='l2', axis=1)
    pca = PCA(n_components=2)
    coords = pca.fit_transform(cooc_norm)

    # Clustering real (colores reales)
    n_clusters = min(8, len(palabras_top))
    kmeans = KMeans(n_clusters=n_clusters, random_state=42).fit(coords)
    colors = kmeans.labels_

    # --- Obtener palabras por cluster ---
    unique_clusters = np.unique(colors)
    cluster_words = {}
    
    for cluster_id in unique_clusters:
        # Obtener palabras de este cluster
        palabras_cluster = [palabras_top[i] for i in range(len(palabras_top)) if colors[i] == cluster_id]
        cluster_words[cluster_id] = palabras_cluster

    # --- Graficar ---
    plt.figure(figsize=(18, 14))

    # Aumentamos el tamaño de los puntos para hacer más visibles las zonas de color
    scatter = plt.scatter(coords[:, 0], coords[:, 1], c=colors, cmap='tab10', s=300, alpha=0.8, edgecolors='k', linewidth=0.3)

    # Etiquetar palabras SIN recuadros
    for i, palabra in enumerate(palabras_top):
        plt.text(coords[i, 0], coords[i, 1], palabra, fontsize=9, ha='center', va='center',
                color='black', weight='normal')  # Texto negro y en negrita para mejor visibilidad

    # Título
    plt.title(f'PCA of most frequent words (based on {total_archivos} files)\n'
            f'PC1: {pca.explained_variance_ratio_[0]:.1%} var, PC2: {pca.explained_variance_ratio_[1]:.1%} var',
            fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('First principal component (PC1)', fontsize=14)
    plt.ylabel('Second principal component (PC2)', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.5)

    # --- Agregar números de cluster en los centroides ---
    # Obtener centroides de los clusters
    centroids = kmeans.cluster_centers_
    
    # Dibujar el número de cada cluster en su centroide
    for i, centroid in enumerate(centroids):
        plt.text(centroid[0], centroid[1], str(i), fontsize=16, ha='center', va='center',
                color='red', weight='bold',
                bbox=dict(boxstyle="circle,pad=0.3", facecolor='white', alpha=0.8, edgecolor='red'))

    # --- Leyenda con número de cluster y palabras ---
    # Construir texto para la leyenda
    legend_text = "Semantic clusters:\n"
    for cluster_id in sorted(unique_clusters):
        palabras_cluster = cluster_words[cluster_id]
        # Limitar a 10 palabras por línea para mejor visualización
        palabras_str = ', '.join(palabras_cluster[:10])
        if len(palabras_cluster) > 10:
            palabras_str += f" (+{len(palabras_cluster)-10} más)"
        legend_text += f"Cluster {cluster_id}: {palabras_str}\n"

    # Dibujar leyenda en la esquina superior izquierda
    plt.text(0.02, 0.98, legend_text, fontsize=9, 
             bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.95, edgecolor='black'),
             transform=plt.gca().transAxes,  # Usar coordenadas del eje actual
             verticalalignment='top', 
             horizontalalignment='left',
             family='monospace')  # Fuente monoespaciada para mejor alineación

    # --- Agregar cuadro de texto con relaciones filosóficas dentro del gráfico ---
           # --- Agregar cuadro de texto con relaciones filosóficas dentro del gráfico ---
    text_box = """\
    --Relationship with philosophical and cognitive themes:
    • Consciousness AND artificial intelligence: The question of whether a machine can be conscious is central to the philosophy of AI (Chalmers, 1995).
    • Phenomenology AND machine learning: Can an algorithm "experience" something? Or at least model phenomenological structures? (McClelland, 2017)
    • Qualia AND computational models: Qualia are the subjective qualities of experience (e.g., "what it feels like to see red"). Exploring this combination allows us to investigate whether there are computational models that attempt to simulate, explain, or even deny the relevance of qualia. (Chalmers, 1995)
    • Intentionality AND AI: We can adopt an "intentional stance" toward complex systems, even if they lack consciousness (Dennett D. C., 1990)
    • Emotion AND neuroeducation: Neuroeducation studies how the brain learns, and emotion is a key modulator of attention, memory, and motivation. This combination explores literature on how to integrate neuroscientific knowledge of emotions into educational environments (human or AI-assisted) (Damasio A., 1994) and affective computing (Picard, 1997)
    • Global Neural Workspace AND simulation: Dehaene's Global Neural Workspace Theory (GNWT) proposes a neuroscientific model of consciousness as "global access" to information. Computationally simulating this model allows for testing hypotheses about how consciousness might emerge in artificial architectures (Dehaene S., 2014).
    • Embodied cognition AND AI limitations: Embodied cognition argues that the mind is not just brain software, but emerges from the interaction between body and environment (Varela, Thompson, & Rosch, 1991).
    • Theory of Mind AND machine consciousness: "Theory of mind" is the ability to attribute mental states to others (Dennett D., 1989).
    • ​​Neuroplasticity AND emotional modulation: Neuroplasticity is the brain's ability to reorganize; emotions modulate this process (e.g., stress inhibits plasticity; motivation enhances it) (Picard, 1997). In educational or adaptive AI settings, understanding this interaction allows for the design of systems that optimize affectively regulated learning (Pradeep et al., 2024).
    • Affective computing AND phenomenological gap: There is a "phenomenological gap": although a machine simulates emotion, it does not feel it (Picard, 1997).
    """
            
    # Formatear el texto con ajuste de línea
    wrapped_text = textwrap.fill(text_box, width=60)
    
    # Posicionar el cuadro de texto más arriba en el borde inferior de la gráfica
    plt.text(0.65, 0.15, wrapped_text, fontsize=8, 
            bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.95, edgecolor='black'),
            transform=plt.gca().transAxes,  # Usar coordenadas del eje actual
            verticalalignment='bottom', 
            horizontalalignment='left',
            wrap=True)

    # Ajustar los límites de los ejes para dejar espacio al cuadro de texto
    plt.xlim(coords[:, 0].min() - 0.5, coords[:, 0].max() + 0.5)
    plt.ylim(coords[:, 1].min() - 0.5, coords[:, 1].max() + 0.5)

    plt.tight_layout()
    plt.savefig("pca_con_leyenda.tiff", dpi=800, bbox_inches='tight', format='tiff')
    plt.savefig("pca_con_leyenda.png", dpi=800, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    main()