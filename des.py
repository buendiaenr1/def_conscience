   # pip install requests beautifulsoup4
 # pip install selenium webdriver-manager
#   pip install chardet


# se usa:      python des.py "https://philpapers.org/s/intentionality%20AND%20ai"
# funciona en CHROME


import requests
from bs4 import BeautifulSoup
import urllib.parse

def descargar_y_guardar_texto(url, nombre_archivo="pagina.txt"):
    try:
        # Hacer la petición GET
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        respuesta = requests.get(url, headers=headers, timeout=10)
        respuesta.raise_for_status()  # Lanza excepción si hay error HTTP

        # Opción 1: Guardar HTML completo
        # with open(nombre_archivo, 'w', encoding='utf-8') as f:
        #     f.write(respuesta.text)

        # Opción 2: Extraer solo texto legible (recomendado para lectura)
        soup = BeautifulSoup(respuesta.content, 'html.parser')

        # Eliminar scripts y estilos
        for script in soup(["script", "style"]):
            script.decompose()

        # Extraer texto
        texto = soup.get_text()

        # Limpiar texto: quitar espacios extra
        lineas = (line.strip() for line in texto.splitlines())
        partes = (frase.strip() for line in lineas for frase in line.split("  "))
        texto_limpio = '\n'.join(parte for parte in partes if parte)

        # Guardar en archivo
        with open(nombre_archivo, 'w', encoding='utf-8') as f:
            f.write(texto_limpio)

        print(f"✅ Contenido guardado en '{nombre_archivo}'")

    except requests.exceptions.RequestException as e:
        print(f"❌ Error al acceder a la URL: {e}")
    except Exception as e:
        print(f"❌ Error inesperado: {e}")

# --- Programa principal ---
if __name__ == "__main__":
    url_usuario = input("📌 Por favor, ingresa la URL: ").strip()

    if not url_usuario:
        print("❌ No ingresaste ninguna URL.")
    else:
        # Generar nombre de archivo basado en la URL (opcional)
        parsed_url = urllib.parse.urlparse(url_usuario)
        nombre_base = parsed_url.netloc + parsed_url.path.replace("/", "_")
        if not nombre_base or nombre_base == "":
            nombre_base = "pagina"
        nombre_archivo = nombre_base + ".txt"

        descargar_y_guardar_texto(url_usuario, nombre_archivo)