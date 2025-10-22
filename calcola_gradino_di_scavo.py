import tifffile
import cv2
import os
import numpy as np
import json
from lxml import etree

# Variabili globali per memorizzare i punti del mouse e l'immagine
punti_selezionati = []
immagine_originale = None

def on_mouse(event, x, y, flags, param):
    """
    Funzione di callback per catturare gli eventi del mouse.
    """
    global punti_selezionati, immagine_originale

    if event == cv2.EVENT_LBUTTONDOWN:
        punti_selezionati.append((x, y))
        print(f"Punto selezionato: ({x}, {y}) - Clic {len(punti_selezionati)} di 3")

        # Disegna un piccolo cerchio sul punto cliccato
        cv2.circle(immagine_originale, (x, y), 5, (0, 255, 0), -1)
        cv2.imshow('Misura Gradino', immagine_originale)


def calcola_gradino_di_scavo(immagine):
    """
    Permette all'utente di selezionare due punti sulla superficie superiore
    e un punto sul fondo per calcolare il gradino.
    """
    global punti_selezionati, immagine_originale
    punti_selezionati = []
    immagine_originale = immagine.copy()
    
    cv2.imshow('Misura Gradino', immagine_originale)
    cv2.setMouseCallback('Misura Gradino', on_mouse)

    print("\n--- Calcolo Gradino di Scavo ---")
    print("1. Seleziona 2 punti sulla superficie superiore (il riferimento).")
    print("2. Seleziona il 3° punto sul fondo dello scavo.")
    print("Premi 'c' per calcolare, 'q' per saltare.")

    risultato = None
    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == ord('c'):
            if len(punti_selezionati) >= 3:
                p_riferimento1 = punti_selezionati[0]
                p_riferimento2 = punti_selezionati[1]
                p_scavo = punti_selezionati[2]

                # Assicurati che l'immagine sia in scala di grigi
                if immagine.ndim > 2:
                    immagine_mono = cv2.cvtColor(immagine, cv2.COLOR_BGR2GRAY)
                else:
                    immagine_mono = immagine
                
                # Calcola l'intensità media dei pixel sulla linea di riferimento
                num_punti = 100
                x = np.linspace(p_riferimento1[0], p_riferimento2[0], num_punti).astype(int)
                y = np.linspace(p_riferimento1[1], p_riferimento2[1], num_punti).astype(int)
                intensita_pixel_riferimento = immagine_mono[y, x]
                intensita_media_riferimento = np.mean(intensita_pixel_riferimento)
                
                # Ottieni l'intensità del pixel sul fondo dello scavo
                intensita_scavo = immagine_mono[p_scavo[1], p_scavo[0]]
                
                gradino_di_scavo_intensita = intensita_media_riferimento - intensita_scavo
                
                risultato = {"gradino_di_scavo_intensita": gradino_di_scavo_intensita}
                
                # Disegna le linee e i punti per visualizzare la selezione
                cv2.line(immagine_originale, p_riferimento1, p_riferimento2, (0, 255, 0), 2)
                cv2.circle(immagine_originale, p_scavo, 5, (0, 0, 255), -1)
                
                testo = f"Gradino: {gradino_di_scavo_intensita:.2f}"
                cv2.putText(immagine_originale, testo, (p_riferimento1[0], p_riferimento1[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                cv2.imshow('Misura Gradino', immagine_originale)
                print(testo)
                punti_selezionati = []
                break
            else:
                print("Devi selezionare 3 punti prima di calcolare.")
        
        elif key == ord('q'):
            risultato = {"gradino_di_scavo_intensita": "Saltato dall'utente"}
            break
            
    cv2.destroyAllWindows()
    return risultato

def elabora_immagini_sequenziale(cartella_base):
    """
    Itera su tutte le immagini ed esegue l'elaborazione su ognuna.
    """
    risultati_finali = []
    
    for dirpath, _, filenames in os.walk(cartella_base):
        for filename in filenames:
            if filename.lower().endswith(('.tif', '.tiff')):
                percorso_completo = os.path.join(dirpath, filename)
                print(f"\n--- Elaborazione di: {filename} ---")
                
                try:
                    immagine = tifffile.imread(percorso_completo)
                    
                    # Normalizza l'immagine per la visualizzazione in OpenCV
                    immagine_norm = cv2.normalize(immagine, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
                    
                    # Esegui il calcolo dell'angolo
                    risultati_immagine = calcola_angolo_di_scavo(immagine_norm.copy())
                    
                    # Aggiungi il nome del file ai risultati
                    risultati_immagine['file_name'] = filename
                    risultati_finali.append(risultati_immagine)

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
# Sostituisci questo percorso con il tuo percorso principale
cartella_di_lavoro = r"C:\Users\Lavoro\Desktop\SPCT_FIORAVANTI"
elabora_immagini_sequenziale(cartella_di_lavoro)
