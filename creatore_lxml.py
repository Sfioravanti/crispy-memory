import tifffile
import numpy as np
import os
from lxml import etree

def estrai_proprieta_estese(file_path):
    """
    Estrae proprietà e metadati dettagliati da un file TIFF,
    restituendo un dizionario con tutte le informazioni.
    """
    dati = {}
    try:
        with tifffile.TiffFile(file_path) as tif:
            # Assumiamo che l'immagine principale sia nella prima pagina
            page = tif.pages[0]
            
            # --- 1. Proprietà standard dell'immagine ---
            altezza_pixel, larghezza_pixel = page.shape[-2:]
            dati['DimensioneTotaleImmagine_Pixel'] = f"{larghezza_pixel}x{altezza_pixel}"
            dati['ProfonditaDiBit'] = page.dtype.itemsize * 8
            
            # Estrazione della Risoluzione (DPI)
            x_res = page.tags.get('XResolution')
            y_res = page.tags.get('YResolution')
            if x_res and y_res:
                x_res_val = x_res.value[0] / x_res.value[1]
                y_res_val = y_res.value[0] / y_res.value[1]
                dati['Risoluzione_DPI'] = f"{x_res_val:.2f}x{y_res_val:.2f}"
            else:
                dati['Risoluzione_DPI'] = "Non Trovata"

            # --- 2. Dati specifici del Microscopio (FEI) ---
            fei_tags = tif.fei_metadata
            
            if fei_tags and 'Scan' in fei_tags:
                dati['CampoImmagine_Metri'] = {
                    'Orizzontale': fei_tags['Scan'].get('HorFieldsize'),
                    'Verticale': fei_tags['Scan'].get('VerFieldsize')
                }
                
                pixel_width_meters = fei_tags['Scan'].get('PixelWidth')
                if pixel_width_meters:
                    dati['ScalaDiConversione_MicronPerPixel'] = pixel_width_meters / 1e-6
                else:
                    dati['ScalaDiConversione_MicronPerPixel'] = "Non Trovata"
            else:
                dati['DatiMicroscopio_FEI'] = "Non Trovati"

            # --- 3. Proprietà del Fascio EBeam ---
            if fei_tags and 'EBeam' in fei_tags:
                dati['FascioEbeam'] = fei_tags['EBeam']
            else:
                dati['FascioEbeam'] = "Non Trovato"

    except Exception as e:
        print(f"Si è verificato un errore durante la lettura dei metadati: {e}")
        return None
    return dati
 
def crea_xml_riepilogo(file_path, dati_immagine):
    """Crea e salva un file XML con i dati estratti."""
    if not dati_immagine:
        print("Nessun dato da salvare.")
        return

    # Estrae il nome del file senza estensione per il nome del tag root
    nome_base_file = os.path.splitext(os.path.basename(file_path))[0]
    
    root = etree.Element(nome_base_file)
    
    # Aggiunge i dati al file XML
    for key, value in dati_immagine.items():
        if isinstance(value, dict):
            sezione = etree.SubElement(root, key)
            for sub_key, sub_value in value.items():
                etree.SubElement(sezione, sub_key).text = str(sub_value)
        else:
            etree.SubElement(root, key).text = str(value)
            
    # Salva il file XML
    nome_file_xml = f"{nome_base_file}.xml"
    albero = etree.ElementTree(root)
    with open(nome_file_xml, "wb") as f:
        albero.write(f, pretty_print=True, xml_declaration=True, encoding='UTF-8')
    
    print(f"Riepilogo salvato con successo nel file: {nome_file_xml}")


# --- Istruzioni per l'uso ---

# Sostituisci 'il_tuo_file.tif' con il nome del tuo file immagine
nome_file_immagine = r'C:\Users\Lavoro\Desktop\SPCT_FIORAVANTI\T_9\SPC_T9_A1_0.5um_20K_01.tif'

dati_estratti = estrai_proprieta_estese(nome_file_immagine)
if dati_estratti:
    crea_xml_riepilogo(nome_file_immagine, dati_estratti)
