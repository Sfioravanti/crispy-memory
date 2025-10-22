import tifffile
import cv2
import os
import numpy as np
import xml.etree.ElementTree as ET
from lxml import etree
import datetime
import math

# ==============================================================================
# VARIABILI GLOBALI E STATO
# ==============================================================================

punti_selezionati = [] # Da P1 (indice 0) a P10 (indice 9)
immagine_originale = None
immagine_modificabile = None
window_name = "Misurazione Interattiva"

# Fattori di scala e stato di interazione
scale_factor_um_per_px = 1.0 # Fattore di scala fisico (µm/px)
PIXEL_TO_UNIT_FACTOR = 1.0 # Alias per scale_factor_um_per_px
misurazioni_salvate_parziali = {}
fase_corrente = 0 # 0: Gradino/Angolo (P1-P4), 1: Trincea (P5-P6), 2: Micro Left (P7-P8), 3: Micro Right (P9-P10)

# Variabili di stato usate nel codice
current_mode = 'NONE'       # Modo corrente: 'GRADINO', 'TRINCEA_LINE', 'TRINCEA_P', 'MICRO_L', 'MICRO_R', 'SCALA'
punti_per_fase = [4, 2, 2, 2] # Numero di punti richiesti in ogni fase

# Le variabili di pan e zoom sono state rimosse per rispettare la richiesta dell'utente.

# ==============================================================================
# FUNZIONI DI UTILITÀ (CALCOLO, SCALA, XML)
# ==============================================================================

def calcola_distanza(p1, p2):
    """Calcola la distanza euclidea tra due punti (x, y)."""
    dist_px = math.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
    dist_um = dist_px * PIXEL_TO_UNIT_FACTOR
    return dist_um

def calcola_angolo(p1, p2, p3):
    """Calcola l'angolo in gradi tra i vettori (P2-P1) e (P2-P3)."""
    # Vettore A: P1 -> P2
    # Vettore B: P3 -> P2
    # L'angolo in un gradino è spesso l'angolo formato da P1-P2 e P2-P3.
    # Usiamo P2 come vertice.
    
    # Vettore P2 -> P1
    v1 = np.array([p1[0] - p2[0], p1[1] - p2[1]])
    # Vettore P2 -> P3
    v2 = np.array([p3[0] - p2[0], p3[1] - p2[1]])

    # Prodotto scalare
    dot_product = np.dot(v1, v2)
    # Magnitudini
    mag_v1 = np.linalg.norm(v1)
    mag_v2 = np.linalg.norm(v2)

    if mag_v1 == 0 or mag_v2 == 0:
        return 0.0 # Evita divisione per zero
    
    # Coseno dell'angolo
    cos_angle = dot_product / (mag_v1 * mag_v2)
    # Protezione contro errori di floating point
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    
    angle_rad = np.arccos(cos_angle)
    angle_deg = np.degrees(angle_rad)
    
    # L'angolo restituito è l'angolo interno tra i due segmenti
    return angle_deg


def calcola_risultati_fase(fase):
    """Calcola i risultati basati sulla fase corrente e sui punti selezionati."""
    
    risultati = {}
    
    # Punti sempre 10 totali
    p1 = punti_selezionati[0] if len(punti_selezionati) > 0 else None
    p2 = punti_selezionati[1] if len(punti_selezionati) > 1 else None
    p3 = punti_selezionati[2] if len(punti_selezionati) > 2 else None
    p4 = punti_selezionati[3] if len(punti_selezionati) > 3 else None
    p5 = punti_selezionati[4] if len(punti_selezionati) > 4 else None
    p6 = punti_selezionati[5] if len(punti_selezionati) > 5 else None
    p7 = punti_selezionati[6] if len(punti_selezionati) > 6 else None
    p8 = punti_selezionati[7] if len(punti_selezionati) > 7 else None
    p9 = punti_selezionati[8] if len(punti_selezionati) > 8 else None
    p10 = punti_selezionati[9] if len(punti_selezionati) > 9 else None
    
    # FASE 0: Gradino / Angolo (P1-P4)
    if fase >= 0 and len(punti_selezionati) >= 4:
        # P1: inizio gradino, P2: angolo gradino, P3: fine gradino, P4: punto di controllo
        if p1 and p2 and p3:
            # Larghezza Gradino: P1-P3
            dist_gradino = calcola_distanza(p1, p3)
            risultati['LarghezzaGradino_um'] = f"{dist_gradino:.3f}"
            
            # Angolo P1-P2-P3
            angolo_gradino = calcola_angolo(p1, p2, p3)
            risultati['AngoloGradino_deg'] = f"{angolo_gradino:.2f}"
            
        # Distanza di controllo P2-P4
        if p2 and p4:
            dist_controllo = calcola_distanza(p2, p4)
            risultati['DistControllo_um'] = f"{dist_controllo:.3f}"

    # FASE 1: Trincea (P5-P6)
    if fase >= 1 and len(punti_selezionati) >= 6:
        if p5 and p6:
            # Larghezza Trincea: P5-P6
            dist_trincea = calcola_distanza(p5, p6)
            risultati['LarghezzaTrincea_um'] = f"{dist_trincea:.3f}"

    # FASE 2: Micro Left (P7-P8)
    if fase >= 2 and len(punti_selezionati) >= 8:
        if p7 and p8:
            # Distanza Micro Left: P7-P8
            dist_micro_l = calcola_distanza(p7, p8)
            risultati['MicroLeft_um'] = f"{dist_micro_l:.3f}"

    # FASE 3: Micro Right (P9-P10)
    if fase >= 3 and len(punti_selezionati) >= 10:
        if p9 and p10:
            # Distanza Micro Right: P9-P10
            dist_micro_r = calcola_distanza(p9, p10)
            risultati['MicroRight_um'] = f"{dist_micro_r:.3f}"

    return risultati

def salva_risultati_xml(risultati_sessione, nome_file_xml='riepilogo_misure.xml'):
    """Salva i risultati in un file XML o li aggiunge se esiste."""
    
    if os.path.exists(nome_file_xml):
        try:
            tree = etree.parse(nome_file_xml)
            root = tree.getroot()
        except etree.ParseError:
            print("Avviso: File XML esistente non valido. Creazione di un nuovo file.")
            root = etree.Element("SessioneMisure", data=datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
            tree = etree.ElementTree(root)
    else:
        root = etree.Element("SessioneMisure", data=datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
        tree = etree.ElementTree(root)

    misure_registrate = 0
    for risultati_immagine in risultati_sessione:
        img_node = etree.SubElement(root, "Immagine", nome=risultati_immagine["nome_file"], data_elaborazione=risultati_immagine["data_elaborazione"])
        
        # Aggiungi fattori di scala
        scale_node = etree.SubElement(img_node, "Scala")
        etree.SubElement(scale_node, "UmPerPx").text = f"{risultati_immagine['scale_factor_um_per_px']:.4f}"
        
        # Aggiungi misure
        misure_node = etree.SubElement(img_node, "Misure")
        for k, v in risultati_immagine["misure"].items():
            etree.SubElement(misure_node, k).text = str(v)

        # Aggiungi coordinate (Punti)
        punti_node = etree.SubElement(img_node, "Punti")
        for i, (x, y) in enumerate(risultati_immagine["punti"]):
            etree.SubElement(punti_node, f"P{i+1}", x=str(x), y=str(y))
        
        # Aggiungi note
        etree.SubElement(img_node, "Note").text = risultati_immagine["note"]

        misure_registrate += 1

    # Scrivi il file XML con formattazione
    try:
        xml_string = etree.tostring(root, pretty_print=True, encoding='UTF-8', xml_declaration=True)
        with open(nome_file_xml, 'wb') as f:
            f.write(xml_string)
    except Exception as e:
        print(f"Errore durante la scrittura del file XML: {e}")
        
    return len(root.findall("Immagine"))

def carica_e_prepara_immagine(filepath):
    """Carica l'immagine TIF e ne estrae il fattore di scala se disponibile."""
    global immagine_originale, PIXEL_TO_UNIT_FACTOR, scale_factor_um_per_px

    # Carica l'immagine TIFF con tifffile
    try:
        with tifffile.TiffFile(filepath) as tif:
            immagine_originale = tif.asarray()
            
            # Tenta di leggere i metadati per il fattore di scala
            try:
                # Logica di estrazione del fattore di scala dai metadati (da implementare se necessario)
                
                # SIMULAZIONE FATTORI DI SCALA (SOSTITUIRE CON LA LOGICA REALE SE POSSIBILE)
                scale_factor_um_per_px = 0.05 # Valore predefinito/simulato
                
            except:
                print(f"Avviso: Impossibile leggere il fattore di scala dai metadati. Usato fattore predefinito: {scale_factor_um_per_px} µm/px")

    except Exception as e:
        print(f"Errore caricamento tifffile: {e}")
        return None, 1.0

    # Conversione per OpenCV (normalizzazione a 8 bit se necessario)
    if immagine_originale.ndim == 3 and immagine_originale.shape[2] == 4:
        # Immagine RGBA, converte in BGR
        immagine_originale = cv2.cvtColor(immagine_originale, cv2.COLOR_RGBA2BGR)
    elif immagine_originale.ndim == 2:
        # Immagine a scala di grigi, converte in BGR
        immagine_originale = cv2.cvtColor(immagine_originale, cv2.COLOR_GRAY2BGR)

    # Assicurati che l'immagine sia a 8 bit per OpenCV
    if immagine_originale.dtype != np.uint8:
        # Normalizzazione: trova min/max e mappa a 0-255
        min_val = np.min(immagine_originale)
        max_val = np.max(immagine_originale)
        if (max_val - min_val) > 0:
            immagine_originale = ((immagine_originale - min_val) / (max_val - min_val) * 255).astype(np.uint8)
        else:
            immagine_originale = np.zeros(immagine_originale.shape, dtype=np.uint8)


    PIXEL_TO_UNIT_FACTOR = scale_factor_um_per_px
    return immagine_originale, scale_factor_um_per_px

# ==============================================================================
# GESTIONE VISUALIZZAZIONE E INTERAZIONE
# ==============================================================================

def aggiorna_visualizzazione_immagine():
    """Mostra l'immagine nella finestra (senza pan/zoom aggiuntivi)."""
    global immagine_modificabile, window_name
    
    if immagine_modificabile is not None:
        cv2.imshow(window_name, immagine_modificabile)

def disegna_stato_e_punti(img_to_draw):
    """Disegna i punti, le linee e lo stato sull'immagine."""
    global punti_selezionati, fase_corrente, PIXEL_TO_UNIT_FACTOR
    
    # Crea una copia per il disegno
    img = img_to_draw.copy()
    
    # 1. DISEGNA I PUNTI ESISTENTI
    num_punti = len(punti_selezionati)
    
    for i, (x, y) in enumerate(punti_selezionati):
        p_name = f"P{i+1}"
        color = (0, 0, 255) # Rosso
        
        # Cambio colore a seconda della fase
        if i < 4: color = (0, 0, 255) # Gradino (Rosso)
        elif i < 6: color = (255, 0, 0) # Trincea (Blu)
        elif i < 8: color = (0, 255, 0) # Micro Left (Verde)
        elif i < 10: color = (255, 255, 0) # Micro Right (Giallo-Ciano)
        
        cv2.circle(img, (x, y), 5, color, -1)
        # Etichetta del punto
        cv2.putText(img, p_name, (x + 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    
    # 2. DISEGNA LE MISURAZIONI IN CORSO / LINEE FINITE
    
    # Fase 0: Gradino / Angolo (P1-P4)
    if num_punti >= 2: # P1-P2
        cv2.line(img, punti_selezionati[0], punti_selezionati[1], (100, 100, 255), 2)
    if num_punti >= 3: # P2-P3
        cv2.line(img, punti_selezionati[1], punti_selezionati[2], (100, 100, 255), 2)
    if num_punti >= 4: # P1-P3 (Larghezza Gradino)
        cv2.line(img, punti_selezionati[0], punti_selezionati[2], (0, 255, 255), 1) # Giallo chiaro
    if num_punti >= 4: # P2-P4 (Controllo)
        cv2.line(img, punti_selezionati[1], punti_selezionati[3], (255, 100, 0), 1) # Arancione
        
    # Fase 1: Trincea (P5-P6)
    if num_punti >= 6:
        cv2.line(img, punti_selezionati[4], punti_selezionati[5], (255, 0, 255), 2) # Viola
        
    # Fase 2: Micro Left (P7-P8)
    if num_punti >= 8:
        cv2.line(img, punti_selezionati[6], punti_selezionati[7], (0, 255, 255), 2) # Giallo
        
    # Fase 3: Micro Right (P9-P10)
    if num_punti >= 10:
        cv2.line(img, punti_selezionati[8], punti_selezionati[9], (0, 255, 0), 2) # Verde chiaro

    # TESTO DI STATO (SULL'IMMAGINE)
    # Mostra l'unità di misura
    cv2.putText(img, f"Scala: {PIXEL_TO_UNIT_FACTOR:.4f} um/px", (10, img.shape[0] - 10), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    # Mostra le misure (solo se complete)
    risultati_temp = calcola_risultati_fase(fase_corrente)
    y_offset = 30
    for k, v in risultati_temp.items():
        cv2.putText(img, f"{k}: {v}", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        y_offset += 20
        
    # Rimuove la stampa in console dal loop di disegno!
    # Istruzioni:
    # num_punti = len(punti_selezionati)
    # if fase_corrente == 0:
    #    print(f"\rFase 1: Gradino/Angolo. Inserire punto P{num_punti+1} (totale: 4 punti).", end="")
    # ...

    return img

def mouse_callback(event, x, y, flags, param):
    """Gestisce gli eventi del mouse (solo click per i punti)."""
    global punti_selezionati, fase_corrente, immagine_modificabile, immagine_originale, window_name, current_mode
    
    # La logica di pan, zoom e rotellina è stata rimossa, come richiesto dall'utente.
    
    # Se il click è un punto
    if event == cv2.EVENT_LBUTTONDOWN:
        
        # Determina il punto corrente che l'utente sta cercando di inserire (da P1 a P10)
        num_punti_attuali = len(punti_selezionati)
        
        # LOGICA DI INSERIMENTO PUNTO
        if fase_corrente == 0 and num_punti_attuali < 4: # P1, P2, P3, P4
            punti_selezionati.append((x, y))
            print(f"\nPunto P{len(punti_selezionati)} inserito a ({x}, {y}).")
            if len(punti_selezionati) == 4:
                print("Fase 1 (Gradino/Angolo) completata.")
                fase_corrente = 1 # Passa alla fase successiva
        
        elif fase_corrente == 1 and 4 <= num_punti_attuali < 6: # P5, P6
            punti_selezionati.append((x, y))
            print(f"\nPunto P{len(punti_selezionati)} inserito a ({x}, {y}).")
            if len(punti_selezionati) == 6:
                print("Fase 2 (Trincea) completata.")
                fase_corrente = 2 # Passa alla fase successiva
                
        elif fase_corrente == 2 and 6 <= num_punti_attuali < 8: # P7, P8
            punti_selezionati.append((x, y))
            print(f"\nPunto P{len(punti_selezionati)} inserito a ({x}, {y}).")
            if len(punti_selezionati) == 8:
                print("Fase 3 (Micro Left) completata.")
                fase_corrente = 3 # Passa alla fase successiva
                
        elif fase_corrente == 3 and 8 <= num_punti_attuali < 10: # P9, P10
            punti_selezionati.append((x, y))
            print(f"\nPunto P{len(punti_selezionati)} inserito a ({x}, {y}).")
            if len(punti_selezionati) == 10:
                print("Fase 4 (Micro Right) completata.")
                fase_corrente = 4 # Fase finale
                # Il messaggio di fine viene gestito nel ciclo principale, non qui.
        
        # Aggiorna la visualizzazione dopo aver inserito un punto
        if immagine_originale is not None:
            immagine_modificabile = disegna_stato_e_punti(immagine_originale)
            aggiorna_visualizzazione_immagine()

# ==============================================================================
# PROCESSO PRINCIPALE
# ==============================================================================

def elabora_immagini_sequential(cartella_base):
    """
    Itera su tutte le immagini .tif presenti nelle sottocartelle
    e avvia il processo di misurazione interattiva per ognuna.
    """
    global immagine_originale, immagine_modificabile, punti_selezionati, fase_corrente
    global current_mode, window_name, PIXEL_TO_UNIT_FACTOR, scale_factor_um_per_px
    
    risultati_sessione = []
    
    print(f"Avvio elaborazione nella cartella: {cartella_base}")
    
    try:
        # Crea la finestra
        cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(window_name, mouse_callback)

        # Variabile LOCALE per tracciare l'ultima fase stampata (per evitare ripetizioni)
        last_printed_phase = -1 
        
        # Iterazione dei file
        for root, dirs, files in os.walk(cartella_base):
            for filename in files:
                if filename.lower().endswith(('.tif', '.tiff')):
                    filepath = os.path.join(root, filename)
                    print(f"\n--- Elaborazione: {filename} ---")
                    
                    try:
                        # 1. Carica e Prepara
                        immagine_originale, scale_factor = carica_e_prepara_immagine(filepath)
                        if immagine_originale is None:
                            print(f"Skippando {filename}: Caricamento fallito.")
                            continue
                            
                        # 2. Reset stato per nuova immagine
                        punti_selezionati = []
                        fase_corrente = 0
                        last_printed_phase = -1 # Reset anche la fase stampata
                        PIXEL_TO_UNIT_FACTOR = scale_factor
                        scale_factor_um_per_px = scale_factor
                        
                        # Inizializza l'immagine modificabile con i disegni iniziali
                        immagine_modificabile = disegna_stato_e_punti(immagine_originale)
                        aggiorna_visualizzazione_immagine()
                        
                        # 3. Loop Interattivo
                        while True:
                            # 3.1 Aggiorna l'immagine in loop (disegno dei punti in tempo reale)
                            immagine_modificabile = disegna_stato_e_punti(immagine_originale)
                            aggiorna_visualizzazione_immagine()

                            # 3.2 GESTIONE ISTRUZIONI IN CONSOLE (SOLO SE LA FASE CAMBIA)
                            if fase_corrente != last_printed_phase:
                                last_printed_phase = fase_corrente
                                print("\n" + "="*50)
                                
                                # Calcola quanti punti mancano alla fase
                                punti_mancanti = punti_per_fase[fase_corrente] - (len(punti_selezionati) - sum(punti_per_fase[:fase_corrente]))
                                punto_corrente = sum(punti_per_fase[:fase_corrente]) + 1
                                
                                if fase_corrente == 0:
                                    print("FASE 1: Gradino/Angolo (P1, P2, P3, P4)")
                                    print(f"Inserire punto P{punto_corrente}. (Totale 4 punti richiesti in questa fase)")
                                    
                                elif fase_corrente == 1:
                                    print("FASE 2: Trincea (P5, P6)")
                                    print(f"Inserire punto P{punto_corrente}. (Totale 2 punti richiesti in questa fase)")
                                    
                                elif fase_corrente == 2:
                                    print("FASE 3: Micro Left (P7, P8)")
                                    print(f"Inserire punto P{punto_corrente}. (Totale 2 punti richiesti in questa fase)")
                                    
                                elif fase_corrente == 3:
                                    print("FASE 4: Micro Right (P9, P10)")
                                    print(f"Inserire punto P{punto_corrente}. (Totale 2 punti richiesti in questa fase)")
                                    
                                elif fase_corrente == 4:
                                    print("TUTTE LE FASI COMPLETATE. Premi 'S' per salvare i risultati o 'Q' per saltare.")
                                
                                print("="*50)
                            
                            # 3.3 Gestione Input Tastiera
                            key = cv2.waitKey(1) & 0xFF
                            
                            if key == ord('s') or key == ord('S'):
                                # Salva e passa alla prossima immagine
                                if fase_corrente == 4 and len(punti_selezionati) == 10:
                                    break
                                else:
                                    print("Attenzione: Non tutti i 10 punti sono stati selezionati. Premi 'Q' per saltare o completa i punti.")
                            
                            elif key == ord('q') or key == ord('Q'):
                                # Salta l'immagine e passa alla prossima
                                print("Immagine saltata dall'utente.")
                                break 
                                
                            elif key == ord('r') or key == ord('R'):
                                # Reset: Riavvia la misurazione per l'immagine corrente
                                punti_selezionati = []
                                fase_corrente = 0
                                last_printed_phase = -1 # Stampa nuovamente le istruzioni
                                print("Misurazione resettata. Ricomincia da P1.")

                            # Forziamo l'uscita dalla scansione delle immagini se la finestra viene chiusa
                            if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                                cv2.destroyAllWindows()
                                return risultati_sessione # Uscita pulita
                                
                        # 4. Salvataggio dei risultati (dopo il 'break' dal loop while True)
                        if len(punti_selezionati) == 10:
                            risultati_finali = calcola_risultati_fase(4)
                            print("\n--- RISULTATI MISURAZIONE ---")
                            for k, v in risultati_finali.items():
                                print(f"{k}: {v}")
                            print("-----------------------------")
                            
                            risultati_immagine = {
                                "nome_file": filename,
                                "data_elaborazione": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                "scale_factor_um_per_px": scale_factor_um_per_px,
                                "misure": risultati_finali,
                                "punti": punti_selezionati
                            }
                            
                            # Richiesta di nota (opzionale)
                            note_utente = input(f"Note libere per {filename} (opzionale): ")
                            risultati_immagine["note"] = note_utente if note_utente else "Nessuna nota aggiunta."
                            
                            # Aggiungi alla sessione per il salvataggio finale
                            risultati_sessione.append(risultati_immagine)
                            
                            # Salvataggio immediato (opzionale, ma utile)
                            misure_salvate = salva_risultati_xml(risultati_sessione, 'riepilogo_misure.xml')
                            print(f"Risultato salvato nel file XML. Misure totali registrate: {misure_salvate}")
                    
                    except Exception as e:
                        print(f"Errore durante l'elaborazione di {filename}: {e}")
                        # Continua con il prossimo file
                        
        print("\n==================================================")
        print("Elaborazione di tutte le immagini completata.")
        
    except Exception as e:
        print(f"Errore critico durante l'iterazione delle cartelle: {e}")
        
    return risultati_sessione

        
# ==============================================================================
# ESECUZIONE DELLO SCRIPT
# ==============================================================================

# **MODIFICARE QUESTO PERCORSO CON LA TUA CARTELLA DI LAVORO**
cartella_di_lavoro = r"C:\Users\Lavoro\Desktop\SPCT_FIORAVANTI"
# Assicurati di chiudere tutte le finestre OpenCV alla fine
try:
    elabora_immagini_sequential(cartella_di_lavoro)
except Exception as main_e:
    print(f"Errore generale: {main_e}")
finally:
    cv2.destroyAllWindows()
