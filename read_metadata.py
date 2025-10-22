import tifffile
import json

# Sostituisci 'il_tuo_file.tif' con il nome del tuo file TIFF
file_path = r'C:\Users\Lavoro\Desktop\SEM_thesis\11_2024\U_25\SDC_U25_A1_1.5um_20K_01.tif'

try:
    with tifffile.TiffFile(file_path) as tif:
        # I metadati sono spesso nel primo "ifd" (Image File Directory)
        metadata = tif.pages[0].tags.get('ImageDescription')

        if metadata is not None:
            # Tenta di decodificare il metadato se è in formato JSON o testo
            description = metadata.value
            try:
                # Prova a decodificare come JSON, un formato comune per i metadati
                metadata_dict = json.loads(description)
                print("Metadati trovati (JSON):")
                print(json.dumps(metadata_dict, indent=4))
            except json.JSONDecodeError:
                # Se non è JSON, stampalo come testo semplice
                print("Metadati trovati (Testo):")
                print(description)
        else:
            print("Nessun metadato 'ImageDescription' trovato nel file.")

except FileNotFoundError:
    print(f"Errore: Il file '{file_path}' non è stato trovato. Assicurati che sia nella stessa cartella dello script.")
except Exception as e:
    print(f"Si è verificato un errore: {e}")
