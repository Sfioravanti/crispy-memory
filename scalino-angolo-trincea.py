import tifffile
import cv2
import os
import numpy as np
import xml.etree.ElementTree as ET
from lxml import etree
import datetime

# ==============================================================================
# VARIABILI GLOBALI E SETUP
# ==============================================================================
punti_selezionati = []
immagine_originale = None
immagine_modificabile = None
window_name = "Misurazione Interattiva"
scale_factor_um_per_px = 1.0
misurazioni_salvate_parziali = {}
fase_corrente = 0 # 0: Gradino/Angolo, 1: Trincea

# ==============================================================================
# NUOVA FUNZIONE: DISEGNA CROCE
# ==============================================================================
def draw_cross(img, center, size, color, thickness=1):
    """
    Disegna una croce centrata in (x, y) invece di un cerchio.
    """
    x, y = center
    # Linea orizzontale
    cv2.line(img, (x - size, y), (x + size, y), color, thickness)
    # Linea verticale
    cv2.line(img, (x, y - size), (x, y + size), color, thickness)

# ==============================================================================
# FUNZIONI DI UTILITÀ
# ==============================================================================

def to_original_coords(x_screen, y_screen):
    """Converte le coordinate sullo schermo (visualizzate) in coordinate sull'immagine originale."""
    x_orig = int((x_screen - current_pan_offset[0]) / scale_factor)
    y_orig = int((y_screen - current_pan_offset[1]) / scale_factor)
    return x_orig, y_orig

def to_screen_coords(x_orig, y_orig):
    """Converte le coordinate originali in coordinate sullo schermo (visualizzate)."""
    x_screen = int(x_orig * scale_factor + current_pan_offset[0])
    y_screen = int(y_orig * scale_factor + current_pan_offset[1])
    return x_screen, y_orig

# --- Callback per il Mouse (Aggiornata per Logging e Calcolo) ---

def on_mouse(event, x, y, flags, param):
    """
    Funzione di callback per catturare gli eventi del mouse (click, zoom, pan).
    Aggiornata per loggare i risultati Gradino/Angolo usando la nuova logica.
    """
    global punti_selezionati, current_mode, reference_line_params, scan_line_y
    global scale_factor, is_panning, last_mouse_pos, current_pan_offset, filepath_global

    x_orig, y_orig = to_original_coords(x, y) 

    if event == cv2.EVENT_LBUTTONDOWN:
        
        if not is_panning:
            if current_mode in ['GRADINO', 'ANGOLO']:
                if len(punti_selezionati) < 3:
                    punti_selezionati.append((x_orig, y_orig))
                    print(f"Punto selezionato (Originale): ({x_orig}, {y_orig}) - Clic {len(punti_selezionati)} di 3")
                    
                    redraw_image() 

                    if len(punti_selezionati) == 2:
                        # Logica Retta Riferimento P1-P2 (invariata)
                        p1, p2 = punti_selezionati[0], punti_selezionati[1]
                        x1, y1 = p1
                        x2, y2 = p2
                        
                        if x2 != x1:
                            m = (y2 - y1) / (x2 - x1)
                            q = y1 - m * x1
                            reference_line_params = (m, q)
                        else:
                            reference_line_params = (None, x1)

                    if len(punti_selezionati) == 3:
                        # ** CHIAMATA ALLA TUA LOGICA DI CALCOLO **
                        gradino_unit_val, angolo_gradi_val, delta_x_px, delta_y_px = gradino_angolo(punti_selezionati, PIXEL_TO_UNIT_FACTOR) 
                        
                        # Conversione per dettaglio e logging
                        delta_x_unit = delta_x_px * PIXEL_TO_UNIT_FACTOR
                        delta_y_unit = delta_y_px * PIXEL_TO_UNIT_FACTOR

                        RESULTS_LOG.append({
                            'image_path': filepath_global,
                            'mode': current_mode,
                            'angolo_gradi': angolo_gradi_val, 
                            'gradino_unit': gradino_unit_val, 
                            'delta_x_px': delta_x_px,
                            'delta_y_px': delta_y_px,
                            'delta_x_unit': delta_x_unit,
                            'delta_y_unit': delta_y_unit,
                            'scale_factor': PIXEL_TO_UNIT_FACTOR
                        })
                        
                        print(f"Risultato Gradino/Angolo salvato. Gradino (Punto-Retta): {gradino_unit_val:.4f} u, Angolo: {angolo_gradi_val:.2f} deg.") 
                        
                        punti_selezionati = []
                        current_mode = 'NONE' 
                    
                
            elif current_mode == 'TRINCEA_PLANE':
                scan_line_y = y_orig
                print(f"Linea di scansione orizzontale selezionata a Y (Originale)={y_orig}. Premi 'T' per calcolare Trincea Otsu.")
                current_mode = 'TRINCEA'
                redraw_image()
            
            elif current_mode == 'TRINCEA_X_SELECT':
                if reference_line_params is not None and reference_line_params[0] is not None:
                    m, q = reference_line_params
                    x_scan = x_orig
                    scan_line_y = int(m * x_scan + q)
                    print(f"Linea di scansione Y (Originale) calcolata ({scan_line_y}) sulla retta di riferimento. Premi 'T' per calcolo Otsu.")
                    current_mode = 'TRINCEA'
                else:
                    print("Errore: Retta non inclinata o non definita per scansione X.")
                    current_mode = 'NONE'
                redraw_image()
            
            redraw_image()
        
    elif event == cv2.EVENT_RBUTTONDOWN:
        is_panning = True
        last_mouse_pos = (x, y)
        
    elif event == cv2.EVENT_RBUTTONUP:
        is_panning = False
        
    elif event == cv2.EVENT_MOUSEMOVE:
        if is_panning:
            dx = x - last_mouse_pos[0]
            dy = y - last_mouse_pos[1]
            current_pan_offset[0] += dx
            current_pan_offset[1] += dy
            last_mouse_pos = (x, y)
            redraw_image()

    elif event == cv2.EVENT_MOUSEWHEEL:
        zoom_factor = 1.2
        
        if flags > 0:
            scale_factor *= zoom_factor
        else:
            scale_factor /= zoom_factor
            
        scale_factor = max(0.1, min(scale_factor, 10.0))
            
        # Mantenere il punto centrale approssimativo
        x_orig_center, y_orig_center = to_original_coords(immagine_modificabile.shape[1] // 2, immagine_modificabile.shape[0] // 2)
        
        # Ricalcolare l'offset per il nuovo fattore di scala in modo che il centro resti lo stesso
        x_new_screen, y_new_screen = to_screen_coords(x_orig_center, y_orig_center)
        
        # Aggiustare il pan (questo è un aggiustamento grezzo, una logica più precisa userebbe il punto del mouse)
        current_pan_offset[0] -= (x_new_screen - immagine_modificabile.shape[1] // 2)
        current_pan_offset[1] -= (y_new_screen - immagine_modificabile.shape[0] // 2)
        
        redraw_image()

def get_scale_factor(tif):
    """
    Estrae il fattore di scala dai metadati, con priorità ai metadati FEI
    e fallback ai metadati TIFF standard.
    """
    # --- TENTATIVO 1: METADATI PROPRIETARI FEI (Scan -> PixelWidth) ---
    pixel_width_source = None
    pixel_width_value = None

    if tif.fei_metadata and 'Scan' in tif.fei_metadata and 'PixelWidth' in tif.fei_metadata['Scan']:
        pixel_width_value = tif.fei_metadata['Scan']['PixelWidth']
        pixel_width_source = "FEI -> Scan -> PixelWidth"
    elif tif.fei_metadata and 'PixelWidth' in tif.fei_metadata:
        pixel_width_value = tif.fei_metadata['PixelWidth']
        pixel_width_source = "FEI -> PixelWidth (root)"

    if pixel_width_value is not None:
        try:
            # Assicuriamo che il valore sia un float, anche se è una stringa
            pixel_width_m = float(str(pixel_width_value).strip())
            # Conversione da metri a micron (1 metro = 10^6 micron)
            fattore_di_scala_um_per_px = pixel_width_m * 1e6
            
            print(f"Risoluzione trovata in {pixel_width_source}: {fattore_di_scala_um_per_px:.8f} µm/px.")
            return fattore_di_scala_um_per_px
        except (ValueError, TypeError, KeyError):
            print(f"Attenzione: '{pixel_width_source}' non è un valore numerico valido. Proseguo la ricerca.")
            pass # Continua la ricerca con i fallback

    # --- TENTATIVO 2: METADATI TIFF STANDARD (XResolution) ---
    try:
        if 'XResolution' in tif.pages[0].tags:
            # Se la tag XResolution esiste, assumiamo che sia la dimensione del pixel in metri (se le unità sono correttamente impostate)
            # Spesso, per i TIFF scientifici, XResolution è l'inverso della dimensione del pixel
            # Utilizziamo una stima basata sulla presenza del tag, ma il valore esatto dipende dal software
            # Qui usiamo la logica base: se non troviamo il valore FEI, il fallback è 1.0
            print("Nessun dato FEI trovato. Fallback a 1.0 µm/px (verifica i metadati standard se questo non è corretto).")
            return 1.0
    except Exception:
        pass # Continua con il valore di default

    print("Nessun metadato valido trovato. Uso il valore di default 1.0 µm/px.")
    return 1.0

def gradino_angolo(punti, scale_factor):
    """ Calcola il gradino (distanza P1-P3) e l'angolo (P1-P2-P3). """
    if len(punti) < 3:
        return 0.0, 0.0, 0, 0
    
    p1, p2, p3 = np.array(punti[0]), np.array(punti[1]), np.array(punti[2])
    
    # 1. Gradino: Distanza perpendicolare da P3 alla retta P1-P2 (Punto-Retta)
    # v_1 (linea) è il vettore P1 -> P2 (vettore di direzione della retta)
    v_1 = p2 - p1 
    # v_2 (gradino) è il vettore P1 -> P3 (vettore necessario per la formula della distanza punto-retta)
    v_2_gradino = p3 - p1 
    
    line_magnitude = np.linalg.norm(v_1)
    
    gradino_dist_px = 0.0
    
    if line_magnitude == 0:
        # Caso P1 = P2: Il gradino è la distanza euclidea da P1 (o P2) a P3.
        gradino_dist_px = np.linalg.norm(v_2_gradino) 
        
    else:
        # Distanza (in pixels) = |CrossProduct(v_1, v_2_gradino)| / |v_1|
        # Cross product 2D: v_1[0]*v_2_gradino[1] - v_1[1]*v_2_gradino[0]
        cross_product = np.abs(v_1[0] * v_2_gradino[1] - v_1[1] * v_2_gradino[0])
        gradino_dist_px = cross_product / line_magnitude
        
    gradino_unit = gradino_dist_px * scale_factor # <--- gradino_unit
    
    # 2. Angolo: Angolo formato da P1-P2 e P2-P3
    v1 = p1 - p2
    v2 = p3 - p2
    
    dot_product = np.dot(v1, v2)
    magnitudes = np.linalg.norm(v1) * np.linalg.norm(v2)
    if magnitudes == 0:
        angolo_gradi = 0.0
    else:
        cos_angle = dot_product / magnitudes
        # Clamping per evitare ValueError con numeri fuori dal range [-1, 1]
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        angolo_rad = np.arccos(cos_angle)
        angolo_gradi = np.degrees(angolo_rad)
        
    return gradino_um, angolo_gradi

# ==============================================================================
# LOGICA DI MISURAZIONE INTERATTIVA (Solo la parte finale è stata toccata)
# ==============================================================================

def misura_immagine_interattiva(immagine, file_path, scale_factor):
    """
    Funzione principale che gestisce la sequenza Gradino -> Angolo -> Trincea.
    """
    global punti_selezionati, immagine_originale, immagine_modificabile, fase_corrente
    
    # *** INIZIALIZZAZIONE COMPLETA (CORREZIONE ERRORE STRUTTURA/VARIABILE NON DEFINITA) ***
    misurazioni = {
        "gradino_um": None,
        "angolo_gradi": None,
        "trincea_larghezza_um": None,
        "file_name": os.path.basename(file_path),
        "data_misurazione": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    punti_selezionati = []
    immagine_originale = immagine.copy()
    immagine_modificabile = immagine.copy()
    fase_corrente = 0 # Iniziamo sempre dalla Fase 0

    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, on_mouse)
    cv2.imshow(window_name, immagine_modificabile)
    
    # ----------------------------------------------------------------------
    # FASE 0: MISURA GRADINO & ANGOLO (3 PUNTI CONDIVISI)
    # ----------------------------------------------------------------------
    print("\n--- FASE 1/3: Gradino & Angolo (3 Punti) ---")
    print("Seleziona 3 punti: P1, P2 (Vertice), P3.")
    print("Premi 'c' per calcolare, 'z' per annullare l'ultimo punto, 'q' per saltare.")

    loop_attivo = True
    while loop_attivo:
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('c'):
            if len(punti_selezionati) >= 3:
                # ... (Logica di calcolo Gradino/Angolo) ...
                # ... (Logica di visualizzazione Gradino/Angolo) ...
                
                fase_corrente = 1 # Passa alla fase successiva
                break # <-- QUI ESCE DAL CICLO DELLA FASE 0
            else:
                print("Devi selezionare 3 punti prima di calcolare.")
        
        elif key == ord('z'): # Annulla punto
            # ... (Logica 'z' FASE 0) ...
            pass
            
        elif key == ord('q'): return {"file_name": os.path.basename(file_path), "status": "Skipped"}
        elif key == ord('x'): return misurazioni # Salva dati parziali e termina

# ==============================================================================
# INIZIO FASE 1: TRINCEA
# Il codice ESEGUE QUI DOPO IL BREAK DELLA FASE 0
# ==============================================================================

    # Ricalcola i parametri della retta P1-P2 (che ora deve muoversi mantenendo la pendenza)
    x1, y1 = punti_selezionati[0]
    x2, y2 = punti_selezionati[1]
    rows, cols, _ = immagine_modificabile.shape
    
    # Variabili per tracciare lo spostamento
    offset_y = 0  # Spostamento verticale (SU/GIÙ)
    offset_x = 0  # Spostamento orizzontale (SX/DX)
    
    # Istruzioni
    print("\n--- FASE 2/3: Trincea (Posizionamento Linea) ---")
    print("1. Usa frecce SU/GIÙ (o W/S) per la traslazione verticale.")
    print("2. Usa A/D per la traslazione orizzontale.")
    print("3. Premi 'c' per confermare la posizione.")
    
    # PRIMO WHILE TRUE: Movimento e conferma della retta
    while True:
        immagine_trincea = immagine_modificabile.copy()

        # Calcola la retta mobile applicando ENTRAMBI gli offset
        if abs(x2 - x1) < 1e-6: # Linea quasi verticale
            start_point_mobile = (x1 + offset_x, 0)
            end_point_mobile = (x1 + offset_x, rows)
        else: # Linea generale con pendenza
            m = (y2 - y1) / (x2 - x1)
            q_originale = y1 - m * x1
            q_finale = q_originale + offset_y - m * offset_x
            
            y_start = int(m * 0 + q_finale)
            y_end = int(m * (cols - 1) + q_finale)
            
            start_point_mobile = (0, y_start)
            end_point_mobile = (cols - 1, y_end)

        # Disegna la retta mobile (che mantiene la pendenza di P1-P2) in ciano
        cv2.line(immagine_trincea, start_point_mobile, end_point_mobile, (255, 255, 0), 2)
        cv2.putText(immagine_trincea, f"Offset Y: {offset_y} / X: {offset_x} (W/S - A/D)", (cols - 350, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        cv2.imshow(window_name, immagine_trincea)
        
        key = cv2.waitKey(10) & 0xFF
        
        if key == 2490368 or key == ord('w'): offset_y = max(-rows, offset_y - 5)
        elif key == 2621440 or key == ord('s'): offset_y = min(rows, offset_y + 5)
        elif key == ord('a'): offset_x = max(-cols, offset_x - 5)
        elif key == ord('d'): offset_x = min(cols, offset_x + 5)
        elif key == ord('c'):
            print("Posizione della retta Trincea confermata. Inizio selezione P4/P5.")
            break
        elif key == ord('q'): return {"file_name": os.path.basename(file_path), "status": "Skipped"}
        elif key == ord('x'): return misurazioni

    # --- SALVATAGGIO PARAMETRI FINALI PER PROIEZIONE ---
    # ESSENZIALE: Salva la pendenza e l'intercetta finale (q_trincea) per la funzione on_mouse
    if abs(x2 - x1) < 1e-6:  
        m_trincea = None  
        q_trincea = x1 + offset_x # X finale per la retta verticale
    else:
        m_trincea = (y2 - y1) / (x2 - x1)
        q_originale = y1 - m_trincea * x1
        q_trincea = q_originale + offset_y - m_trincea * offset_x
    
    # --- INIZIALIZZAZIONE FASE DI SELEZIONE P4/P5 ---
    punti_selezionati = punti_selezionati[:3] # Manteniamo solo i primi 3 punti
    # Nota: fase_corrente è già 1
    
    misurazione_trincea_completata = False
    
    # Ridisegna lo stato attuale dell'immagine dopo aver confermato la linea
    immagine_modificabile = immagine_originale.copy()
    # Ridisegna P1, P2, P3 e le linee originali del gradino (per contesto)
    p1, p2, p3 = punti_selezionati[0], punti_selezionati[1], punti_selezionati[2]
    cv2.line(immagine_modificabile, p1, p2, (255, 0, 0), 2)
    cv2.line(immagine_modificabile, p2, p3, (255, 0, 0), 2)
    cv2.line(immagine_modificabile, p1, p3, (0, 0, 255), 2)
    for p in punti_selezionati:
         draw_cross(immagine_modificabile, p, 6, (0, 255, 0), 2)

    print("\n--- FASE 3/3: Selezione Punti (Fine-Tuning) ---")
    print("Seleziona 2 punti sulla retta della trincea (P4, P5).")
    print("Dopo aver selezionato entrambi, usa 'a'/'d' per il fine-tuning di P4/P5.")
    print("Premi 'z' per annullare l'ultimo punto o 'c' per calcolare/salvare.")
    
    # SECONDO WHILE TRUE: Selezione e fine-tuning di P4 e P5
    while True: 
        immagine_trincea_click = immagine_modificabile.copy()
        
        # Disegna la retta mobile fissa (Giallo/Ciano)
        cv2.line(immagine_trincea_click, start_point_mobile, end_point_mobile, (255, 255, 0), 2)
        
        # Disegna i punti P4 e P5 già selezionati
        for i, p in enumerate(punti_selezionati[3:]):
            colore = (255, 0, 255)
            # Evidenzia l'ultimo punto per il fine-tuning (se len=4 o len=5)
            if len(punti_selezionati) == i + 4:  
                colore = (0, 255, 255)  
                if not misurazione_trincea_completata:
                    cv2.putText(immagine_trincea_click, "Ultimo punto selezionato. Usa A/D per spostare.", (p[0] + 10, p[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, colore, 1)

            draw_cross(immagine_trincea_click, p, 6, colore, -1)

        # --- Visualizzazione Misure e Istruzioni ---
        if not misurazione_trincea_completata:
            cv2.putText(immagine_trincea_click, f"Seleziona {5 - len(punti_selezionati)} punti (P4, P5) per la larghezza...", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        else:
            # Stato di revisione
            p4, p5 = punti_selezionati[3], punti_selezionati[4]
            cv2.line(immagine_trincea_click, p4, p5, (255, 0, 255), 2)
            cv2.putText(immagine_trincea_click, f"Larghezza: {misurazioni['trincea_larghezza_um']} um", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
            cv2.putText(immagine_trincea_click, "Premi 'c' per SALVARE/ESCI o 'z' per RIFARE i punti (Annulla).", (10, rows - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        cv2.imshow(window_name, immagine_trincea_click)
        key = cv2.waitKey(1) & 0xFF
        
        # ----------------------------------------------------------------------------------
        # LOGICA DI SPOSTAMENTO P4/P5 (A/D) - CORRETTA
        # ----------------------------------------------------------------------------------
        if (key == ord('a') or key == ord('d')) and len(punti_selezionati) >= 4 and not misurazione_trincea_completata:
            
            indice_punto = len(punti_selezionati) - 1  # L'ultimo punto cliccato
            x_corrente, y_corrente = punti_selezionati[indice_punto]
            spostamento_pixel = -5 if key == ord('a') else 5
            
            nuova_x = x_corrente + spostamento_pixel
            
            if m_trincea is None:  # Retta verticale
                print("Impossibile muovere orizzontalmente su retta verticale.")
                continue
            else:
                m = m_trincea
                q = q_trincea
                nuova_y = int(m * nuova_x + q)

            punti_selezionati[indice_punto] = (nuova_x, nuova_y)
        
        # --- Gestione Comandi 'c' e 'z', 'q', 'x' ---

        elif key == ord('c'):
            if len(punti_selezionati) < 5:
                print("Devi selezionare 2 punti (P4, P5) prima di calcolare. Punti attuali:", len(punti_selezionati) - 3)
                continue
                
            if not misurazione_trincea_completata:
                # PRIMA CONFERMA: Esegui il calcolo
                p4, p5 = punti_selezionati[3], punti_selezionati[4]
                distanza_px = np.linalg.norm(np.array(p4) - np.array(p5))
                larghezza_um = distanza_px * scale_factor
                misurazioni["trincea_larghezza_um"] = f"{larghezza_um:.2f}"
                
                print(f"Trincea misurata: {larghezza_um:.2f} um. Revisiona e premi 'c' per uscire.")
                misurazione_trincea_completata = True
            else:
                # SECONDA CONFERMA: Esci dal loop
                break # <-- QUI ESCE DAL CICLO DELLA FASE 1
        
        elif key == ord('z'):
            if len(punti_selezionati) > 3: # Puoi annullare solo P4 o P5
                punti_selezionati.pop()
                misurazione_trincea_completata = False # Reimposta lo stato se annulli
                print(f"Annullato. Punti selezionati: {len(punti_selezionati) - 3}. Seleziona nuovamente i punti.")
            else:
                print("Non ci sono punti P4/P5 da annullare.")
        
        elif key == ord('q'):  
            return {"file_name": os.path.basename(file_path), "status": "Skipped"}
        
        elif key == ord('x'):  
            return misurazioni

# ==============================================================================
# FINE FUNZIONE
# ==============================================================================
    # Ritorna le misurazioni una volta che la FASE 1 è completata
    return misurazioni

# ==============================================================================
# LOGICA DI ESECUZIONE E SALVATAGGIO
# ==============================================================================

def salva_risultati_xml(nuovi_dati, nome_file):
    """ Salva o aggiorna i risultati in un file XML. (Il codice rimane lo stesso)"""
    
    # 1. Carica l'albero esistente o creane uno nuovo
    if os.path.exists(nome_file):
        try:
            tree = ET.parse(nome_file)
            root = tree.getroot()
        except ET.ParseError:
            print(f"Errore nella lettura del file XML esistente. Sovrascrivo un nuovo file.")
            root = ET.Element("RiepilogoMisure")
            tree = ET.ElementTree(root)
    else:
        root = ET.Element("RiepilogoMisure")
        tree = ET.ElementTree(root)
    
    # 2. Aggiunge i nuovi dati
    for item in nuovi_dati:
        misura_element = ET.SubElement(root, "Misura", filename=item['file_name'])
        
        # Raggruppa per data di acquisizione
        data_acq = item.get("data_acquisizione", "Sconosciuta")
        ET.SubElement(misura_element, "DataAcquisizione").text = data_acq
        
        for key, value in item.items():
            if key not in ['file_name', 'data_acquisizione']:
                ET.SubElement(misura_element, key).text = str(value)
    
    # 3. Scrivi l'XML con lxml per una formattazione pulita
    xml_string = ET.tostring(root, encoding='utf-8')
    parser = etree.XMLParser(remove_blank_text=True)
    tree_lxml = etree.fromstring(xml_string, parser)
    
    with open(nome_file, "wb") as f:
        f.write(etree.tostring(tree_lxml, pretty_print=True, xml_declaration=True, encoding='UTF-8'))

def elabora_immagini_sequential(cartella_base):
    """ Itera su tutte le immagini e gestisce la sequenza di misurazione. """
    risultati_sessione = []
    
    if not os.path.exists(cartella_base):
        print(f"Errore: La cartella '{cartella_base}' non esiste.")
        return

    for dirpath, _, filenames in os.walk(cartella_base):
        for filename in filenames:
            if filename.lower().endswith(('.tif', '.tiff')):
                percorso_completo = os.path.join(dirpath, filename)
                print(f"\n==================================================")
                print(f"--- INIZIO ELABORAZIONE: {filename} ---")
                
                try:
                    # 1. Estrazione del fattore di scala
                    with tifffile.TiffFile(percorso_completo) as tif:
                        scale_factor = get_scale_factor(tif)

                    # 2. Caricamento e normalizzazione immagine
                    immagine = tifffile.imread(percorso_completo)
                    immagine_norm = cv2.normalize(immagine, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
                    
                    # 3. Misurazione interattiva
                    risultati_immagine = misura_immagine_interattiva(immagine_norm.copy(), percorso_completo, scale_factor)
                    
                    if risultati_immagine.get("status") == "Skipped":
                        print(f"Immagine {filename} saltata per richiesta utente ('q').")
                    else:
                        risultati_immagine["file_name"] = filename
                        risultati_sessione.append(risultati_immagine)
                        print(f"Risultati per {filename} aggiunti al batch. Prossimo file...")

                except Exception as e:
                    print(f"ERRORE FATALE durante l'elaborazione di {filename}: {e}")
                    
    # 4. Salvataggio finale di tutti i risultati della sessione
    if risultati_sessione:
        salva_risultati_xml(risultati_sessione, 'riepilogo_misure.xml')
        print(f"\nProcesso completato. {len(risultati_sessione)} nuove misurazioni salvate/aggiornate in riepilogo_misure.xml.")
    else:
        print("\nNessuna nuova misurazione completata in questa sessione.")

# ==============================================================================
# ESECUZIONE DELLO SCRIPT
# ==============================================================================

# **MODIFICARE QUESTO PERCORSO CON LA TUA CARTELLA DI LAVORO**
cartella_di_lavoro = r"C:\Users\Lavoro\Desktop\SPCT_FIORAVANTI"
elabora_immagini_sequential(cartella_di_lavoro)
