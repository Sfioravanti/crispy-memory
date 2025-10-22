import tifffile
import numpy as np

def estrai_proprieta_estese(file_path):
    """
    Estrae proprietà e metadati dettagliati da un file TIFF,
    incluse le informazioni standard e quelle del fascio EBeam.
    """
    try:
        with tifffile.TiffFile(file_path) as tif:
            # Assumiamo che l'immagine principale sia nella prima pagina
            page = tif.pages[0]
            
            # --- 1. Proprietà standard dell'immagine ---
            print("--- Proprietà Standard dell'Immagine ---")
            altezza_pixel, larghezza_pixel = page.shape[-2:]
            print(f"Dimensione totale immagine: {larghezza_pixel} x {altezza_pixel} pixel")
            
            profondita_bit = page.dtype.itemsize * 8
            print(f"Profondità di bit: {profondita_bit} bit")

            # Estrazione della Risoluzione (spesso in DPI)
            x_res = page.tags.get('XResolution')
            y_res = page.tags.get('YResolution')
            if x_res and y_res:
                x_res_val = x_res.value[0] / x_res.value[1]
                y_res_val = y_res.value[0] / y_res.value[1]
                print(f"Risoluzione: {x_res_val:.2f} DPI (X), {y_res_val:.2f} DPI (Y)")
            else:
                print("Risoluzione (DPI) non trovata.")

            # --- 2. Dati specifici del Microscopio (FEI) ---
            print("\n--- Dati del Microscopio (FEI) ---")
            fei_tags = tif.fei_metadata
            
            if fei_tags and 'Scan' in fei_tags:
                # Campo Immagine (Field of View)
                hor_fieldsize = fei_tags['Scan'].get('HorFieldsize')
                ver_fieldsize = fei_tags['Scan'].get('VerFieldsize')
                if hor_fieldsize and ver_fieldsize:
                    print(f"Campo immagine (Field of View): {hor_fieldsize:.2e} m (orizzontale), {ver_fieldsize:.2e} m (verticale)")
                
                # Scala di Conversione
                pixel_width_meters = fei_tags['Scan'].get('PixelWidth')
                if pixel_width_meters:
                    pixel_per_micron = pixel_width_meters / 1e-6
                    print(f"Scala di conversione: {pixel_per_micron:.6f} micron per pixel")
            else:
                print("Dati del Microscopio (FEI) non trovati.")
                
            # --- 3. Proprietà del Fascio EBeam ---
            print("\n--- Proprietà del Fascio (EBeam) ---")
            if fei_tags and 'EBeam' in fei_tags:
                ebeam_properties = fei_tags['EBeam']
                for key, value in ebeam_properties.items():
                    print(f"  {key}: {value}")
            else:
                print("Proprietà del fascio 'EBeam' non trovate.")

    except FileNotFoundError:
        print(f"Errore: Il file '{file_path}' non è stato trovato. Assicurati che sia nella stessa cartella dello script.")
    except Exception as e:
        print(f"Si è verificato un errore durante la lettura dei metadati: {e}")

# Sostituisci 'il_tuo_file.tif' con il nome del tuo file immagine
nome_file_immagine =  r'C:\Users\Lavoro\Desktop\SEM_thesis\11_2024\U_25\SDC_U25_A1_1.5um_20K_01.tif'

estrai_proprieta_estese(nome_file_immagine)
