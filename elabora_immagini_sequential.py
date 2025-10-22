import tifffile
import cv2
import os
import numpy as np
from lxml import etree

# Variabili globali per memorizzare i punti del mouse e l'immagine
punti_selezionati = []
immagine_originale = None
immagine_modificabile = None
window_name = ""

def on_mouse(event, x, y, flags, param):
    """
    Funzione di callback per catturare gli eventi del mouse e disegnare.
    """
    global punti_selezionati, immagine_modificabile, window_name

    if event == cv2.EVENT_LBUTTONDOWN:
        if len(punti_selezionati) < 3:
            punti_selezionati.append((x, y))
            print(f"Punto selezionato: ({x}, {y}) - Clic {len(punti_selezionati)} di 3")

            # Disegna un piccolo cerchio sul punto cliccato
            cv2.circle(immagine_modificabile, (x, y), 5, (0, 255, 0), -1)
            cv2.imshow(window_name, immagine_modificabile)
        else:
            print("Hai già selezionato 3 punti. Premi 'c' per calcolare o 'z' per annullare.")


def calcola_gradino_di_scavo(immagine, fattore_di_scala_um_per_px):
    """
    Permette all'utente di selezionare due punti sulla superficie superiore
    e un punto sul fondo per calcolare il gradino.
    """
    global punti_selezionati, immagine_originale, immagine_modificabile, window_name
    punti_selezionati = []
    immagine_originale = immagine.copy()
    immagine_modificabile = immagine.copy()
    window_name = "Misura Gradino"
    
    cv2.imshow(window_name, immagine_modificabile)
    cv2.setMouseCallback(window_name, on_mouse)

    print("\n--- Misura del Gradino ---")
    print("1. Seleziona 2 punti sulla superficie superiore (riferimento).")
    print("2. Seleziona il 3° punto sul fondo dello scavo.")
    print("Premi 'c' per calcolare, 'z' per annullare, 'q' per saltare.")

    risultato = {"gradino_di_scavo_um": "Non misurato"}
    while True:
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('c'):
            if len(punti_selezionati) >= 3:
                p_riferimento1, p_riferimento2, p_scavo = punti_selezionati[0], punti_selezionati[1], punti_selezionati[2]
                
                # Calcolo del gradino basato sulla distanza verticale in pixel
                distanza_verticale_px = abs(p_riferimento1[1] - p_scavo[1])
                gradino_um = distanza_verticale_px * fattore_di_scala_um_per_px
                
                risultato["gradino_di_scavo_um"] = f"{gradino_um:.2f}"
                
                cv2.line(immagine_modificabile, p_riferimento1, p_riferimento2, (0, 255, 0), 2)
                cv2.circle(immagine_modificabile, p_scavo, 5, (0, 0, 255), -1)
                cv2.putText(immagine_modificabile, f"Gradino: {gradino_um:.2f} um", (p_riferimento1[0], p_riferimento1[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.imshow(window_name, immagine_modificabile)
                print(f"Gradino misurato: {gradino_um:.2f} um")
                
                cv2.waitKey(0)
                break
            else:
                print("Devi selezionare 3 punti prima di calcolare.")
        
        elif key == ord('z'):
            if len(punti_selezionati) > 0:
                punti_selezionati.pop()
                print(f"Ultimo punto annullato. Punti rimanenti: {len(punti_selezionati)}")
                immagine_modificabile = immagine_originale.copy()
                for p in punti_selezionati:
                    cv2.circle(immagine_modificabile, p, 5, (0, 255, 0), -1)
                cv2.imshow(window_name, immagine_modificabile)
            else:
                print("Non ci sono punti da annullare.")
        
        elif key == ord('q'):
            break
    
    cv2.destroyAllWindows()
    return risultato

def calcola_angolo_di_scavo(immagine):
    """
    Permette all'utente di selezionare tre punti e calcola l'angolo.
    """
    global punti_selezionati, immagine_originale, immagine_modificabile, window_name
    punti_selezionati = []
    immagine_originale = immagine.copy()
    immagine_modificabile = immagine.copy()
    window_name = "Misura Angolo"
    
    cv2.imshow(window_name, immagine_modificabile)
    cv2.setMouseCallback(window_name, on_mouse)

    print("\n--- Misura dell'Angolo ---")
    print("1. Seleziona 2 punti per la linea di base dello scavo.")
    print("2. Seleziona il 3° punto per la pendenza della parete.")
    print("Premi 'c' per calcolare, 'z' per annullare, 'q' per saltare.")

    risultato = {"angolo_di_scavo_gradi": "Non misurato"}
    while True:
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('c'):
            if len(punti_selezionati) >= 3:
                p1, p2, p3 = punti_selezionati[0], punti_selezionati[1], punti_selezionati[2]
                
                v1 = np.array([p2[0] - p1[0], p2[1] - p1[1]])
                v2 = np.array([p3[0] - p2[0], p3[1] - p2[1]])
                
                prodotto_scalare = np.dot(v1, v2)
                magnitudine1 = np.linalg.norm(v1)
                magnitudine2 = np.linalg.norm(v2)
                
                angolo_rad = np.arccos(prodotto_scalare / (magnitudine1 * magnitudine2))
                angolo_gradi = np.degrees(angolo_rad)
                
                risultato["angolo_di_scavo_gradi"] = f"{angolo_gradi:.2f}"
                cv2.line(immagine_modificabile, p1, p2, (0, 255, 0), 2)
                cv2.line(immagine_modificabile, p2, p3, (0, 255, 0), 2)
                cv2.putText(immagine_modificabile, f"Angolo: {angolo_gradi:.2f} gradi", (p1[0], p1[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.imshow(window_name, immagine_modificabile)
                print(f"Angolo misurato: {angolo_gradi:.2f} gradi")
                
                cv2.waitKey(0)
                break
            else:
                print("Devi selezionare 3 punti prima di calcolare.")
        
        elif key == ord('z'):
            if len(punti_selezionati) > 0:
                punti_selezionati.pop()
                print(f"Ultimo punto annullato. Punti rimanenti: {len(punti_selezionati)}")
                immagine_modificabile = immagine_originale.copy()
                for p in punti_selezionati:
                    cv2.circle(immagine_modificabile, p, 5, (0, 255, 0), -1)
                cv2.imshow(window_name, immagine_modificabile)
            else:
                print("Non ci sono punti da annullare.")
        
        elif key == ord('q'):
            break
            
    cv2.destroyAllWindows()
    return risultato

def elabora_immagini_sequential(cartella_base):
    """
    Itera su tutte le immagini ed esegue l'elaborazione su ognuna.
    """
    risultati_finali = []
    
    if not os.path.exists(cartella_base):
        print(f"Errore: La cartella '{cartella_base}' non esiste.")
        return

    for dirpath, _, filenames in os.walk(cartella_base):
        for filename in filenames:
            if filename.lower().endswith(('.tif', '.tiff')):
                percorso_completo = os.path.join(dirpath, filename)
                print(f"\n--- Elaborazione di: {filename} ---")
                
                try:
                    with tifffile.TiffFile(percorso_completo) as tif:# Controlla se i metadati FEI esistono e li estrae
                        fei_metadata = tif.fei_metadata
                        # Inizializza il fattore di scala a None
                        fattore_di_scala_um_per_px = None
    
                    if fei_metadata and 'Scan' in fei_metadata and 'PixelWidth' in fei_metadata['Scan']:
                    # Estrai il valore di PixelWidth (che è in metri)
                        pixel_width_m = fei_metadata['Scan']['PixelWidth']
        
                    # Converte da metri a micron (1 metro = 1.000.000 micron)
                        fattore_di_scala_um_per_px = pixel_width_m * 1e6
                        # Converte da metri a micron
                        fattore_di_scala_um_per_px = pixel_width_m * 1e6 if pixel_width_m else 1.0

                    immagine = tifffile.imread(percorso_completo)
                    immagine_norm = cv2.normalize(immagine, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
                    
                    risultati_immagine = {"file_name": filename}
                    
                    # Esegue il calcolo del gradino
                    risultati_gradino = calcola_gradino_di_scavo(immagine_norm.copy(), fattore_di_scala_um_per_px)
                    risultati_immagine.update(risultati_gradino)
                    
                    # Esegue il calcolo dell'angolo
                    risultati_angolo = calcola_angolo_di_scavo(immagine_norm.copy())
                    risultati_immagine.update(risultati_angolo)
                    
                    risultati_finali.append(risultati_immagine)
                    
                    print(f"Elaborazione per {filename} completata. Passaggio al prossimo file...")

                except Exception as e:
                    print(f"Errore durante l'elaborazione di {filename}: {e}")
                    
    # Salva i risultati in un file XML
    salva_risultati_xml(risultati_finali, 'riepilogo_misure.xml')
    print("\nProcesso completato. I risultati sono stati salvati.")

def salva_risultati_xml(dati, nome_file):
    """
    Salva una lista di dizionari in un unico file XML.
    """
    root = etree.Element("RiepilogoMisure")
    
    for item in dati:
        misura_element = etree.SubElement(root, "Misura", filename=item['file_name'])
        for key, value in item.items():
            if key != 'file_name':
                etree.SubElement(misura_element, key).text = str(value)
    
    albero = etree.ElementTree(root)
    with open(nome_file, "wb") as f:
        albero.write(f, pretty_print=True, xml_declaration=True, encoding='UTF-8')

# Esegui lo script principale
cartella_di_lavoro = r"C:\Users\Lavoro\Desktop\SPCT_FIORAVANTI"
elabora_immagini_sequential(cartella_di_lavoro)
