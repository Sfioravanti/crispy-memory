import tifffile
import numpy as np
from lxml import etree
import os

# --- Configura qui il percorso del tuo file ---
# Usa la 'r' prima delle virgolette per evitare errori con la barra rovesciata \
# Assicurati di aggiungere l'estensione del file (.tif o .tiff) alla fine!
file_path = r'C:\Users\Lavoro\Desktop\SEM_thesis\11_2024\U_25\SDC_U25_A1_1.5um_20K_01.tif'

# --- FUNZIONE PRINCIPALE: ESTRE IL RAPPORTO PIXEL/MICRON ---
def get_pixels_per_micron(path):
    """
    Estrae il rapporto pixel/micron dai metadati OME-XML di un file TIFF.
    """
    if not os.path.exists(path):
        print(f"Errore: il file non esiste al percorso '{path}'")
        return None

    try:
        with tifffile.TiffFile(path) as tif:
            # Legge la prima pagina dell'immagine
            page = tif.pages[0]

            # Controlla se il tag ImageDescription contiene metadati OME-XML
            if 'ImageDescription' in page.tags:
                description = page.tags['ImageDescription'].value
                
                # Se la descrizione è un XML, tenta di estrarre il pixel size
                if description.startswith('<?xml'):
                    root = etree.fromstring(description.encode('utf-8'))
                    namespaces = {'ome': 'http://www.openmicroscopy.org/Schemas/OME/2016-06'}
                    
                    # Cerca il tag PixelSizeX (dimensione di un pixel in micron)
                    pixelsize_x_node = root.find(".//ome:PixelSizeX", namespaces=namespaces)
                    
                    if pixelsize_x_node is not None:
                        pixel_size_micron = float(pixelsize_x_node.get('ID'))
                        pixels_per_micron = 1.0 / pixel_size_micron
                        return pixels_per_micron
                    else:
                        print("Errore: Nessun metadato PixelSizeX trovato.")
            
            print("Avviso: Nessun metadato di calibrazione trovato. Ritorno a un valore predefinito.")
            return 1.0  # Valore predefinito se la scala non viene trovata

    except Exception as e:
        print(f"Si è verificato un errore durante la lettura del file: {e}")
        return None


# --- ESECUZIONE DELLO SCRIPT ---
if __name__ == "__main__":
    
    # 1. Estrai il rapporto pixel/micron
    pixels_per_micron_ratio = get_pixels_per_micron(file_path)
    
    if pixels_per_micron_ratio is not None:
        print(f"\nScala di conversione: {pixels_per_micron_ratio:.2f} pixel per micron.")
        
        # 2. Esempio di calcolo
        # Supponiamo di aver misurato un oggetto e che la sua lunghezza sia 250 pixel.
        distanza_in_pixel = 250
        distanza_in_micron = distanza_in_pixel / pixels_per_micron_ratio
        
        print(f"La lunghezza di {distanza_in_pixel} pixel corrisponde a {distanza_in_micron:.2f} micron.")
    else:
        print("Impossibile procedere con i calcoli.")
