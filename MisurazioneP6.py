import tifffile
import cv2
import os
import numpy as np
import xml.etree.ElementTree as ET
from lxml import etree
import datetime
import math

# ==============================================================================
# VARIABILI GLOBALI E SETUP
# ==============================================================================
punti_selezionati = []
immagine_originale = None
immagine_modificabile = None
window_name = "Misurazione Interattiva"

# Fattori di scala e stato di interazione
scale_factor_um_per_px = 1.0 # Fattore di scala fisico (µm/px)
PIXEL_TO_UNIT_FACTOR = 1.0 # Alias per scale_factor_um_per_px (usato nella logica di calcolo)
misurazioni_salvate_parziali = {}
fase_corrente = 0 # 0: Gradino/Angolo (P1-P4), 1: Trincea (P5-P6)

# Variabili per la visualizzazione e interazione (Pan)
visual_scale_factor = 1.0 # Fisso a 1.0 (Zoom rimosso come richiesto)
current_pan_offset = [0, 0] # Offset di spostamento per il panning
is_panning = False
last_mouse_pos = (0, 0)

# Variabili di stato usate nel codice
current_mode = 'NONE'         # Modo corrente: 'GRADINO', 'ANGOLO', 'TRINCEA', 'NONE'
reference_line_params = None  # (m, q) o (None, x1) per la retta P1-P2 (Top Surface)
filepath_global = ""
RESULTS_LOG = [] # Per memorizzare i risultati prima del salvataggio XML
SEZIONE_NOTE = "\n\n---\n## Note dell'Analisi\n\n[NOTE]"

# ==============================================================================
# NUOVA FUNZIONE: DISEGNA CROCE
# ==============================================================================
def draw_cross(img, center, size, color, thickness=1):
    """
    Disegna una croce centrata in (x, y).
    """
    x, y = center
    # Linea orizzontale
    cv2.line(img, (x - size, y), (x + size, y), color, thickness)
    # Linea verticale
    cv2.line(img, (x, y - size), (x, y + size), color, thickness)

# ==============================================================================
# FUNZIONI DI UTILITÀ PER COORDINATE (AGGIORNATE SENZA SCALA VISUALE)
# ==============================================================================

def to_original_coords(x_screen, y_screen):
    """Converte le coordinate sullo schermo (visualizzate con pan) in coordinate sull'immagine originale (1:1)."""
    # Poiché lo zoom è rimosso, visual_scale_factor è 1.0
    x_orig = int(x_screen - current_pan_offset[0])
    y_orig = int(y_screen - current_pan_offset[1])
    return x_orig, y_orig

def to_screen_coords(x_orig, y_orig):
    """
    Converte le coordinate originali in coordinate sullo schermo (visualizzate con pan).
    """
    # Poiché lo zoom è rimosso, visual_scale_factor è 1.0
    x_screen = int(x_orig + current_pan_offset[0])
    y_screen = int(y_orig + current_pan_offset[1])
    return x_screen, y_screen

def redraw_image():
    """
    Funzione per ridisegnare l'immagine sullo schermo tenendo conto del panning.
    """
    global immagine_originale, immagine_modificabile, punti_selezionati, window_name
    global current_pan_offset, current_mode, reference_line_params

    if immagine_originale is None:
        return

    # L'immagine di base è l'originale
    immagine_modificabile = immagine_originale.copy() 
    
    # 1. Disegna la retta di riferimento P1-P2 se selezionata
    # Il check va fino a 6 punti (P1-P6)
    if len(punti_selezionati) >= 2:
        p1_orig, p2_orig = punti_selezionati[0], punti_selezionati[1]
        p1_screen = to_screen_coords(*p1_orig)
        p2_screen = to_screen_coords(*p2_orig)
        
        # Disegna la retta di riferimento P1-P2
        cv2.line(immagine_modificabile, p1_screen, p2_screen, (255, 0, 0), 2)

    # 2. Disegna tutti i punti selezionati (P1, P2, P3, P4, P5, P6)
    for i, p_orig in enumerate(punti_selezionati):
        p_screen = to_screen_coords(*p_orig)
        
        # P1, P2 (Retta), P3 (Angolo), P4 (Gradino/Fondo)
        if i < 4:
            color = (0, 255, 0) # Verde per i punti di Gradino/Angolo
        # P5, P6 (Trincea)
        else:
            color = (255, 0, 255) # Viola/Magenta per i punti di Trincea
            
        label = f"P{i+1}"
        
        draw_cross(immagine_modificabile, p_screen, 6, color, 1)
        cv2.putText(immagine_modificabile, label, (p_screen[0] + 10, p_screen[1] - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    # 3. Visualizzazione stato
    cv2.putText(immagine_modificabile, f"Fase {fase_corrente+1}: {current_mode} - Punti Totali: {len(punti_selezionati)}", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    cv2.imshow(window_name, immagine_modificabile)


# --- Callback per il Mouse ---

def on_mouse(event, x, y, flags, param):
    """
    Funzione di callback per catturare gli eventi del mouse (click e pan).
    Aggiornata per la selezione di 4 punti (P1-P4) per la Fase 0 (Gradino/Angolo)
    e 2 punti (P5-P6) per la Fase 1 (Trincea).
    """
    global punti_selezionati, current_mode, reference_line_params, fase_corrente
    global is_panning, last_mouse_pos, current_pan_offset, filepath_global, PIXEL_TO_UNIT_FACTOR

    x_orig, y_orig = to_original_coords(x, y) 

    if event == cv2.EVENT_LBUTTONDOWN:
        
        if not is_panning:
            # Fase 0: Gradino/Angolo (P1, P2, P3, P4)
            if fase_corrente == 0 and len(punti_selezionati) < 4:
                punti_selezionati.append((x_orig, y_orig))
                print(f"Punto selezionato (Originale): ({x_orig}, {y_orig}) - P{len(punti_selezionati)} di 4")
                
                redraw_image() 

                if len(punti_selezionati) == 2:
                    # Logica Retta Riferimento P1-P2
                    p1, p2 = punti_selezionati[0], punti_selezionati[1]
                    x1, y1 = p1
                    x2, y2 = p2
                    
                    if x2 != x1:
                        m = (y2 - y1) / (x2 - x1)
                        q = y1 - m * x1
                        reference_line_params = (m, q) # Retta per Gradino/Angolo
                    else:
                        reference_line_params = (None, x1)

                if len(punti_selezionati) == 4:
                    # Calcolo e log per Gradino/Angolo
                    gradino_unit_val, angolo_gradi_val, delta_x_px, delta_y_px = gradino_angolo(punti_selezionati, PIXEL_TO_UNIT_FACTOR) 
                    
                    delta_x_unit = delta_x_px * PIXEL_TO_UNIT_FACTOR
                    delta_y_unit = delta_y_px * PIXEL_TO_UNIT_FACTOR

                    RESULTS_LOG.append({
                        'image_path': filepath_global,
                        'mode': 'GRADINO_ANGOLO',
                        'angolo_gradi': angolo_gradi_val, 
                        'gradino_unit': gradino_unit_val, 
                        'delta_x_px': delta_x_px,
                        'delta_y_px': delta_y_px,
                        'delta_x_unit': delta_x_unit,
                        'delta_y_unit': delta_y_unit,
                        'scale_factor': PIXEL_TO_UNIT_FACTOR
                    })
                    
                    print(f"Risultato Gradino/Angolo salvato. Gradino (P4-Retta P1-P2): {gradino_unit_val:.4f} u, Angolo (P1-P3-P2): {angolo_gradi_val:.2f} deg.") 
                    
                    current_mode = 'NONE' # Passa a NONE per uscire dal loop di selezione e aspettare 'c'
            
            # Fase 1: Trincea (P5, P6). I punti vengono proiettati sulla retta della trincea.
            elif fase_corrente == 1 and len(punti_selezionati) < 6:
                # La retta di riferimento (reference_line_params) in questa fase contiene
                # i parametri della retta mobile della trincea (m_trincea, q_trincea).

                if reference_line_params is None:
                    print("Errore: Retta di riferimento della trincea non definita.")
                    return

                m, q = reference_line_params 
                
                if m is None: # Retta verticale (x = q)
                    p_proiettato = (q, y_orig) # Si muove solo in Y lungo la retta verticale
                else:
                    # Calcolo del punto proiettato sulla retta y = m*x + q
                    x_click, y_click = x_orig, y_orig
                    
                    # Retta perpendicolare passante per (x_click, y_click)
                    m_perp = -1 / m
                    q_perp = y_click - m_perp * x_click
                    
                    # Intersezione (nuova_x, nuova_y)
                    nuova_x = int((q_perp - q) / (m - m_perp))
                    nuova_y = int(m * nuova_x + q)
                    p_proiettato = (nuova_x, nuova_y)
                    
                punti_selezionati.append(p_proiettato)
                print(f"Punto P{len(punti_selezionati)} selezionato e proiettato su retta: {p_proiettato}")
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

def get_scale_factor(tif):
    """
    Estrae il fattore di scala dai metadati.
    """
    # ... (Logica di estrazione scala invariata) ...
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
            print("Nessun dato FEI valido trovato. Fallback a 1.0 µm/px.")
            return 1.0
    except Exception:
        pass 

    print("Nessun metadato valido trovato. Uso il valore di default 1.0 µm/px.")
    return 1.0

def gradino_angolo(punti, scale_factor):
    """ 
    Calcola il gradino (distanza perpendicolare P4-Retta P1-P2) 
    e l'angolo (Angolo P1-P3-P2, Vertice P3). 
    """
    if len(punti) < 4:
        return 0.0, 0.0, 0, 0
    
    p1, p2, p3, p4 = np.array(punti[0]), np.array(punti[1]), np.array(punti[2]), np.array(punti[3])
    
    # 1. Gradino: Distanza perpendicolare da P4 (Fondo) alla retta P1-P2 (Superficie)
    v_1 = p2 - p1           # Vettore P1 -> P2 (Definizione retta)
    v_step = p4 - p1        # Vettore P1 -> P4 
    
    line_magnitude = np.linalg.norm(v_1)
    
    gradino_dist_px = 0.0
    
    if line_magnitude == 0:
        # Caso P1 = P2: Gradino è la distanza euclidea da P1 a P4.
        gradino_dist_px = np.linalg.norm(v_step) 
        
    else:
        # Formula per Distanza Punto-Retta in 2D: |(x2-x1)(y4-y1) - (y2-y1)(x4-x1)| / line_magnitude
        cross_product = np.abs(v_1[0] * v_step[1] - v_1[1] * v_step[0])
        gradino_dist_px = cross_product / line_magnitude
        
    gradino_unit = gradino_dist_px * scale_factor # <--- gradino_unit
    
    # Delta X e Y per il log (P4 rispetto a P1 per riferimento)
    delta_x_px = p4[0] - p1[0]
    delta_y_px = p4[1] - p1[1]
    
    # 2. Angolo: Angolo formato da P1-P3 e P2-P3 (Vertice P3)
    v1_angle = p1 - p3 # Vettore P3 -> P1
    v2_angle = p2 - p3 # Vettore P3 -> P2
    
    dot_product = np.dot(v1_angle, v2_angle)
    magnitudes = np.linalg.norm(v1_angle) * np.linalg.norm(v2_angle)
    
    if magnitudes == 0:
        angolo_gradi = 0.0
    else:
        cos_angle = dot_product / magnitudes
        # Clamping per evitare ValueError con numeri fuori dal range [-1, 1]
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        angolo_rad = np.arccos(cos_angle)
        angolo_gradi = np.degrees(angolo_rad)
        
    return gradino_unit, angolo_gradi, delta_x_px, delta_y_px

# ==============================================================================
# LOGICA DI MISURAZIONE INTERATTIVA
# ==============================================================================

def misura_immagine_interattiva(immagine, file_path, scale_factor_um_per_px):
    """
    Funzione principale che gestisce la sequenza di misurazione.
    """
    global punti_selezionati, immagine_originale, immagine_modificabile, fase_corrente
    global current_pan_offset, current_mode, PIXEL_TO_UNIT_FACTOR, filepath_global, visual_scale_factor
    global reference_line_params
    
    # *** INIZIALIZZAZIONE COMPLETA E RESET GLOBALI PER NUOVA IMMAGINE ***
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

    # Inizializzazione Globale per la Sessione Immagine (NO ZOOM)
    PIXEL_TO_UNIT_FACTOR = scale_factor_um_per_px
    filepath_global = file_path
    current_pan_offset = [0, 0] # Reset pan
    visual_scale_factor = 1.0   # Zoom disattivato
    current_mode = 'GRADINO'    # Inizia in modalità selezione Gradino/Angolo
    reference_line_params = None # Reset retta di riferimento P1-P2
    
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, on_mouse)
    cv2.imshow(window_name, immagine_modificabile)
    
    # ----------------------------------------------------------------------
    # FASE 0: MISURA GRADINO & ANGOLO (4 PUNTI: P1, P2, P3, P4)
    # ----------------------------------------------------------------------
    print("\n--- FASE 1/2: Gradino & Angolo (4 Punti) ---")
    print("Seleziona 4 punti:")
    print("P1, P2: Retta di riferimento (Superficie Superiore).")
    print("P3: Vertice per la misura dell'angolo (P1-P3-P2).")
    print("P4: Punto sul fondo per la misura del gradino (P4 vs Retta P1-P2).")
    print("Premi 'c' per calcolare e passare alla Trincea, 'z' per annullare l'ultimo punto, 'q' per saltare.")

    loop_attivo = True
    while loop_attivo:
        redraw_image() # Aggiorna la visualizzazione
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('c'):
            if len(punti_selezionati) >= 4:
                # Il calcolo è già avvenuto in on_mouse quando len=4.
                
                # Cerca l'ultimo risultato in RESULTS_LOG 
                latest_result = RESULTS_LOG[-1] if RESULTS_LOG else {}
                misurazioni["gradino_um"] = f"{latest_result.get('gradino_unit', 0.0):.4f}"
                misurazioni["angolo_gradi"] = f"{latest_result.get('angolo_gradi', 0.0):.2f}"
                
                fase_corrente = 1 # Passa alla fase successiva
                break # <-- QUI ESCE DAL CICLO DELLA FASE 0
            else:
                print("Devi selezionare 4 punti prima di calcolare.")
        
        elif key == ord('z'): # Annulla punto
            if len(punti_selezionati) > 0:
                punti_selezionati.pop()
                current_mode = 'GRADINO' # Torna in modalità selezione
                if len(punti_selezionati) < 2:
                    reference_line_params = None
                print(f"Punto annullato. Punti attuali: {len(punti_selezionati)}")
            else:
                print("Nessun punto da annullare.")
            
        elif key == ord('q'): return {"file_name": os.path.basename(file_path), "status": "Skipped"}
        elif key == ord('x'): return misurazioni # Salva dati parziali e termina

    # ==============================================================================
    # INIZIO FASE 1: TRINCEA (P5 e P6)
    # ==============================================================================
    punti_selezionati_gradino = list(punti_selezionati) # Salva P1, P2, P3, P4
    punti_selezionati = punti_selezionati[:4] # Manteniamo solo i primi 4 punti
    
    # Parametri P1-P2 (Base per la retta mobile)
    x1, y1 = punti_selezionati_gradino[0]
    x2, y2 = punti_selezionati_gradino[1]
    rows, cols, _ = immagine_modificabile.shape
    
    # Variabili per tracciare lo spostamento della retta (Traslazione)
    offset_y = 0  # Spostamento verticale (SU/GIÙ)
    offset_x = 0  # Spostamento orizzontale (SX/DX)
    
    # Istruzioni Traslazione Retta
    print("\n--- FASE 2/2 - Parte 1: Trincea (Posizionamento Linea) ---")
    print("1. Usa frecce SU/GIÙ (o W/S) per la traslazione verticale.")
    print("2. Usa A/D per la traslazione orizzontale.")
    print("3. Premi 'c' per confermare la posizione e passare alla selezione P5/P6.")
    
    m_trincea = None
    q_trincea = None

    # PRIMO WHILE TRUE: Movimento e conferma della retta (Spostamento da P1-P2 a Trincea)
    while True:
        immagine_trincea = immagine_originale.copy()

        # Disegna P1, P2, P3, P4 per contesto
        for p in punti_selezionati_gradino:
            draw_cross(immagine_trincea, p, 6, (0, 255, 0), 1)

        # Calcola la retta mobile applicando ENTRAMBI gli offset
        if abs(x2 - x1) < 1e-6: # Linea quasi verticale (x = costante)
            start_point_mobile = (x1 + offset_x, 0)
            end_point_mobile = (x1 + offset_x, rows)
            
            m_trincea = None  
            q_trincea = x1 + offset_x
        else: # Linea generale con pendenza
            m = (y2 - y1) / (x2 - x1)
            q_originale = y1 - m * x1
            # La nuova intercetta 'q_finale' incorpora sia offset_y che la compensazione di m*offset_x
            q_finale = q_originale + offset_y - m * offset_x
            
            y_start = int(m * 0 + q_finale)
            y_end = int(m * (cols - 1) + q_finale)
            
            start_point_mobile = (0, y_start)
            end_point_mobile = (cols - 1, y_end)

            m_trincea = m
            q_trincea = q_finale

        # Disegna la retta mobile (che mantiene la pendenza di P1-P2) in ciano
        cv2.line(immagine_trincea, to_screen_coords(*start_point_mobile), to_screen_coords(*end_point_mobile), (255, 255, 0), 2)
        cv2.putText(immagine_trincea, f"Offset Y: {offset_y} / X: {offset_x} (W/S - A/D)", (cols - 350, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        cv2.imshow(window_name, immagine_trincea)
        
        key = cv2.waitKey(10) & 0xFF
        
        # Uso W/S per traslazione verticale e A/D per traslazione orizzontale
        if key == 2490368 or key == ord('w'): offset_y = max(-rows, offset_y - 5)
        elif key == 2621440 or key == ord('s'): offset_y = min(rows, offset_y + 5)
        elif key == ord('a'): offset_x = max(-cols, offset_x - 5)
        elif key == ord('d'): offset_x = min(cols, offset_x + 5)
        elif key == ord('c'):
            print("Posizione della retta Trincea confermata. Inizio selezione P5/P6.")
            break
        elif key == ord('q'): return {"file_name": os.path.basename(file_path), "status": "Skipped"}
        elif key == ord('x'): return misurazioni

    # --- INIZIALIZZAZIONE FASE DI SELEZIONE P5/P6 ---
    # La retta di riferimento per la fase di click deve essere la retta della trincea finale.
    reference_line_params = (m_trincea, q_trincea)
    current_mode = 'TRINCEA'
    
    misurazione_trincea_completata = False
    
    print("\n--- FASE 2/2 - Parte 2: Selezione Punti P5/P6 (Fine-Tuning) ---")
    print("Clicca per selezionare 2 punti sulla retta della trincea (P5, P6).")
    print("Dopo aver selezionato P5 o P6, usa 'a'/'d' per il fine-tuning di 1 pixel.")
    print("Premi 'z' per annullare l'ultimo punto o 'c' per calcolare/salvare.")
    
    # SECONDO WHILE TRUE: Selezione e fine-tuning di P5 e P6
    while True: 
        immagine_trincea_click = immagine_originale.copy()
        rows, cols, _ = immagine_trincea_click.shape
        
        # 1. Disegna la retta mobile fissa (Giallo/Ciano)
        cv2.line(immagine_trincea_click, to_screen_coords(*start_point_mobile), to_screen_coords(*end_point_mobile), (255, 255, 0), 2)
        
        # 2. Disegna P1, P2, P3, P4 per contesto
        for p in punti_selezionati_gradino:
            draw_cross(immagine_trincea_click, p, 6, (0, 255, 0), 1)
            
        # 3. Disegna i punti P5 e P6 già selezionati
        # L'indice parte da 4, quindi i punti selezionati sono punti_selezionati[4] (P5) e [5] (P6)
        for i in range(4, len(punti_selezionati)):
            p_orig = punti_selezionati[i]
            p_screen = to_screen_coords(*p_orig) 
            colore = (255, 0, 255)
            
            # Evidenzia l'ultimo punto per il fine-tuning se non è completata la misura
            if len(punti_selezionati) - 1 == i and not misurazione_trincea_completata: 
                colore = (0, 255, 255) # Giallo/Ciano per l'ultimo punto modificabile
                cv2.putText(immagine_trincea_click, "Ultimo punto selezionato. Usa A/D per spostare (1px).", (p_screen[0] + 10, p_screen[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, colore, 1)

            draw_cross(immagine_trincea_click, p_screen, 6, colore, 1)

        # --- Visualizzazione Misure e Istruzioni ---
        if len(punti_selezionati) < 6:
             cv2.putText(immagine_trincea_click, f"Seleziona {6 - len(punti_selezionati)} punti (P5, P6) per la larghezza...", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        else:
            # Stato di revisione
            p5, p6 = punti_selezionati[4], punti_selezionati[5]
            p5_s = to_screen_coords(*p5)
            p6_s = to_screen_coords(*p6)
            cv2.line(immagine_trincea_click, p5_s, p6_s, (255, 0, 255), 2)
            cv2.putText(immagine_trincea_click, f"Larghezza: {misurazioni['trincea_larghezza_um']} um", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
            cv2.putText(immagine_trincea_click, "Premi 'c' per SALVARE/ESCI o 'z' per RIFARE i punti P5/P6.", (10, rows - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        cv2.imshow(window_name, immagine_trincea_click)
        key = cv2.waitKey(1) & 0xFF
        
        # ----------------------------------------------------------------------------------
        # LOGICA DI SPOSTAMENTO P5/P6 (A/D) - TUNING 1 PIXEL
        # ----------------------------------------------------------------------------------
        if (key == ord('a') or key == ord('d')) and len(punti_selezionati) >= 5 and not misurazione_trincea_completata:
            
            indice_punto = len(punti_selezionati) - 1  # L'ultimo punto cliccato (P5 o P6)
            x_corrente, y_corrente = punti_selezionati[indice_punto]
            spostamento_pixel = -1 if key == ord('a') else 1 # Modifica: 1 pixel alla volta
            
            nuova_x = x_corrente + spostamento_pixel
            
            if m_trincea is None:  # Retta verticale
                # Muove verticalmente se la retta è verticale (spostamento in Y)
                nuova_y = y_corrente + spostamento_pixel
                punti_selezionati[indice_punto] = (x_corrente, nuova_y)
            else:
                # Muove orizzontalmente (spostamento in X) e ricalcola Y sulla retta
                m = m_trincea
                q = q_trincea
                nuova_y = int(m * nuova_x + q)
                punti_selezionati[indice_punto] = (nuova_x, nuova_y)
                
            print(f"Punto P{indice_punto+1} spostato a: {punti_selezionati[indice_punto]}")
            
        # --- Gestione Comandi 'c' e 'z', 'q', 'x' ---

        elif key == ord('c'):
            if len(punti_selezionati) < 6:
                print("Devi selezionare 2 punti (P5, P6) prima di calcolare. Punti Trincea attuali:", len(punti_selezionati) - 4)
                continue
                
            if not misurazione_trincea_completata:
                # PRIMA CONFERMA: Esegui il calcolo
                p5, p6 = punti_selezionati[4], punti_selezionati[5]
                distanza_px = np.linalg.norm(np.array(p5) - np.array(p6))
                larghezza_um = distanza_px * scale_factor_um_per_px
                misurazioni["trincea_larghezza_um"] = f"{larghezza_um:.4f}"
                
                print(f"Trincea misurata: {larghezza_um:.4f} um. Revisiona e premi 'c' per uscire.")
                misurazione_trincea_completata = True
            else:
                # SECONDA CONFERMA: Esci dal loop
                break # <-- QUI ESCE DAL CICLO DELLA FASE 1
        
        elif key == ord('z'):
            if len(punti_selezionati) > 4: # Puoi annullare solo P5 o P6
                punti_selezionati.pop()
                misurazione_trincea_completata = False # Reimposta lo stato se annulli
                print(f"Annullato. Punti Trincea selezionati: {len(punti_selezionati) - 4}. Seleziona nuovamente i punti.")
            else:
                print("Non ci sono punti P5/P6 da annullare.")
        
        elif key == ord('q'): 
            return {"file_name": os.path.basename(file_path), "status": "Skipped"}
        
        elif key == ord('x'): 
            return misurazioni

    # Ritorna le misurazioni una volta completate entrambe le fasi
    return misurazioni

# ==============================================================================
# LOGICA DI ESECUZIONE E SALVATAGGIO (Aggiornata per salvataggio incrementale)
# ==============================================================================

def salva_risultati_xml(nuovi_dati, nome_file):
    """ Salva o aggiorna i risultati in un file XML. """
    
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
        # Evita di salvare item saltati
        if item.get("status") == "Skipped":
            continue

        misura_element = ET.SubElement(root, "Misura", filename=item['file_name'])
        
        # Raggruppa per data di acquisizione (Se disponibile. Altrimenti usa data misurazione)
        data_acq = item.get("data_acquisizione", item.get("data_misurazione", "Sconosciuta"))
        ET.SubElement(misura_element, "DataMisurazione").text = item.get("data_misurazione", "N/A")
        
        for key, value in item.items():
            if key not in ['file_name', 'data_acquisizione', 'data_misurazione', 'status']: # Aggiunto 'status' da ignorare
                ET.SubElement(misura_element, key).text = str(value)
    
    # 3. Scrivi l'XML con lxml per una formattazione pulita
    if etree:
        xml_string = ET.tostring(root, encoding='utf-8')
        parser = etree.XMLParser(remove_blank_text=True)
        tree_lxml = etree.fromstring(xml_string, parser)
        
        with open(nome_file, "wb") as f:
            f.write(etree.tostring(tree_lxml, pretty_print=True, xml_declaration=True, encoding='UTF-8'))
    else:
        # Usa la funzione di indentazione di ElementTree se lxml non c'è
        ET.indent(tree, space="    ", level=0)
        tree.write(nome_file, encoding='utf-8', xml_declaration=True)
    
    return len([item for item in nuovi_dati if item.get("status") != "Skipped"])
        

def elabora_immagini_sequential(cartella_base):
    """ Itera su tutte le immagini e gestisce la sequenza di misurazione. """
    global RESULTS_LOG
    RESULTS_LOG = [] # Resetta il log all'inizio della sessione
    risultati_sessione = [] # Mantenuto solo per il conteggio totale della sessione
    
    if not os.path.exists(cartella_base):
        print(f"Errore: La cartella '{cartella_base}' non esiste.")
        return

    try:
        for dirpath, _, filenames in os.walk(cartella_base):
            for filename in filenames:
                if filename.lower().endswith(('.tif', '.tiff')):
                    percorso_completo = os.path.join(dirpath, filename)
                    print(f"\n==================================================")
                    print(f"--- INIZIO ELABORAZIONE: {filename} ---")
                    
                    try:
                        # 1. Estrazione del fattore di scala (Assumiamo che le funzioni get_scale_factor e tifffile siano definite)
                        with tifffile.TiffFile(percorso_completo) as tif:
                            scale_factor = get_scale_factor(tif)

                        # 2. Caricamento e normalizzazione immagine (Assumiamo che le variabili/moduli siano definiti)
                        immagine = tifffile.imread(percorso_completo)
                        # Converti in 8-bit per visualizzazione
                        if immagine.dtype != np.uint8:
                            immagine_norm = cv2.normalize(immagine, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
                        else:
                            immagine_norm = immagine
                        
                        # 3. Misurazione interattiva
                        risultati_immagine = misura_immagine_interattiva(immagine_norm.copy(), percorso_completo, scale_factor)
                        
                        cv2.destroyAllWindows() # Chiudi la finestra prima di passare al prossimo file
                        
                        if risultati_immagine.get("status") == "Skipped":
                            print(f"Immagine {filename} saltata per richiesta utente ('q').")
                        else:
                            risultati_immagine["file_name"] = filename
                            
                            # SEZIONE PER LE NOTE (Usando l'input diretto)
                            print("\n--- INSERIMENTO ANNOTAZIONI ---")
                            note_utente = input(f"Note per {filename} (opzionale): ")
                            risultati_immagine["note"] = note_utente if note_utente else "Nessuna nota aggiunta."
                            
                            # ** PUNTO CRITICO: SALVATAGGIO INCREMENTALE DOPO OGNI MISURA **
                            misure_salvate = salva_risultati_xml([risultati_immagine], 'riepilogo_misure.xml')
                            
                            # Aggiorna il contatore per il messaggio finale della sessione
                            risultati_sessione.append(risultati_immagine) 
                            
                            print(f"Risultati per {filename} **salvati e file XML aggiornato**. Prossimo file...")
                            # se vedo che sono più comoda con la finestra aperta cv2.destroyAllWindows()
                    
                    except Exception as e:
                        print(f"ERRORE DURANTE L'ELABORAZIONE DI {filename}: {e}. Passaggio al prossimo file.")
                        cv2.destroyAllWindows()

    except KeyboardInterrupt:
        print("\n\n!!! INTERRUZIONE UTENTE (CTRL+C) RILEVATA. !!!")
        print("Il salvataggio è gestito in modo incrementale; tutti i risultati completati sono già salvati nel file XML.")

    # 4. Messaggio finale (Il salvataggio è stato spostato all'interno del ciclo)
    if risultati_sessione:
        print(f"\nProcesso terminato. {len(risultati_sessione)} nuove misurazioni salvate in riepilogo_misure.xml in questa sessione.")
    else:
        print("\nNessuna nuova misurazione completata in questa sessione.")
        
# ==============================================================================
# ESECUZIONE DELLO SCRIPT
# ==============================================================================

# Definisci la cartella di esempio che contiene i file .tif
# CARTELLA_DI_TEST = "percorso/alla/tua/cartella/di/immagini" 
# print(f"Per eseguire lo script, devi definire la variabile CARTELLA_DI_TEST con il percorso corretto delle immagini.")

# **MODIFICARE QUESTO PERCORSO CON LA TUA CARTELLA DI LAVORO**
cartella_di_lavoro = r"C:\Users\Lavoro\Desktop\SPCT_FIORAVANTI"
# Assicurati di chiudere tutte le finestre OpenCV alla fine
try:
    elabora_immagini_sequential(cartella_di_lavoro)
except Exception as main_e:
    print(f"Errore generale: {main_e}")
finally:
    cv2.destroyAllWindows()
