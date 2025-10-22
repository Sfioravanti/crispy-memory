import tifffile
import json

file_path = r'C:\Users\Lavoro\Desktop\SPCT_FIORAVANTI\T_96\SPC_T96_A1_0.5um_20K_01.tif'

try:
    with tifffile.TiffFile(file_path) as tif:
        print(f"Lettura di tutti i tag dal file: {file_path}")
        print("-------------------------------------------------")
        for page in tif.pages:
            for tag in page.tags.values():
                # Stampa il nome e il valore di ogni tag
                print(f"Tag: {tag.name} -> Valore: {tag.value}")
                
                # Se il valore è molto grande, stampa solo un estratto
                if isinstance(tag.value, (bytes, str)) and len(tag.value) > 200:
                    print(f"    (Valore troppo lungo, stampato solo l'inizio: {tag.value[:200]}...)")
            print("\n-------------------------------------------------")
except FileNotFoundError:
    print(f"Errore: Il file '{file_path}' non è stato trovato.")
except Exception as e:
    print(f"Si è verificato un errore: {e}")
