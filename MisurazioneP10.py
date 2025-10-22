import tifffile
import cv2
import os
import numpy as np
import xml.etree.ElementTree as ET
from lxml import etree
import datetime
import math

# ==============================================================================
# VARIABILI GLOBALI E STATO (PULITO DA PAN/ZOOM)
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
punti_per_fase = [4, 2, 2, 2] # Numero di punti richiesti in ogni fase

# Variabili di stato usate nel codice
current_mode = 'NONE'       # Modo corrente: 'GRADINO', 'TRINCEA_LINE', 'TRINCEA_P', 'MICRO_L', 'MICRO_R', 'NONE'
reference_line_params = None # (m, q) o (None, x1) per la retta P1-P2 (Top Surface)
filepath_global = ""
RESULTS_LOG = [] # Per memorizzare i risultati prima del salvataggio XML
SEZIONE_NOTE = "\n\n---\n## Note dell'Analisi\n\n[NOTE]"

# Le variabili visual_scale_factor, current_pan_offset, is_panning, last_mouse_pos sono state rimosse.

# ==============================================================================
# FUNZIONI DI BASE (coordinate e disegno)
# ==============================================================================

def to_screen_coords(x_orig, y_orig):
    """
    Funzione fittizia: ora le coordinate originali sono le coordinate schermo.
    Mantenuta per coerenza con il codice di disegno se si usava *p_orig.
    """
    return (int(x_orig), int(y_orig))


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
# LOGICA DI DISEGNO (Aggiornata, usa solo coordinate originali)
# ==============================================================================

def redraw_image():
    """
    Funzione per ridisegnare l'immagine sullo schermo senza tenere conto di pan/zoom.
    """
    global immagine_originale, immagine_modificabile, punti_selezionati, window_name
    global current_mode, reference_line_params, fase_corrente

    if immagine_originale is None:
        return

    # L'immagine modificabile è semplicemente una copia dell'originale, non scalata/traslata.
    immagine_modificabile = immagine_originale.copy()  
    
    # 1. Disegna la retta di riferimento P1-P2 (Top Surface)
    if len(punti_selezionati) >= 2:
        # Usiamo direttamente le coordinate originali (punti_selezionati[i])
        p1, p2 = punti_selezionati[0], punti_selezionati[1]
        
        cv2.line(immagine_modificabile, p1, p2, (255, 0, 0), 2) # Blu

    # 2. Disegna la retta della trincea (se definita in Fase 1)
    if fase_corrente >= 1 and reference_line_params is not None:
        m, q = reference_line_params
        rows, cols, _ = immagine_modificabile.shape
        
        if m is None: # Retta verticale (x = q)
            start_p = (q, 0)
            end_p = (q, rows)
        else: # Retta generale (y = mx + q)
            y_start = int(m * 0 + q)
            y_end = int(m * (cols - 1) + q)
            start_p = (0, y_start)
            end_p = (cols - 1, y_end)

        # Retta della trincea in ciano. Usiamo le coordinate originali
        cv2.line(immagine_modificabile, start_p, end_p, (255, 255, 0), 2)


    # 3. Disegna tutti i punti selezionati (P1-P10)
    for i, p_orig in enumerate(punti_selezionati):
        # p_screen è uguale a p_orig
        p_screen = p_orig 
        label = f"P{i+1}"
        
        # P1-P4 (Gradino/Angolo/Faceting)
        if i < 4:
            color = (0, 255, 0) # Verde
        # P5-P6 (Trincea Larghezza)
        elif i < 6:
            color = (0, 255, 255) # Giallo
        # P7-P8 (Micro Trench Left)
        elif i < 8:
            color = (255, 0, 255) # Magenta
        # P9-P10 (Micro Trench Right)
        else:
            color = (0, 0, 255) # Rosso (Aggiornato per evitare conflitto con Linea P1-P2)
            
        # Disegna croce e testo usando le coordinate originali
        draw_cross(immagine_modificabile, p_screen, 6, color, 1)
        cv2.putText(immagine_modificabile, label, (p_screen[0] + 10, p_screen[1] - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        # Disegna la linea di misura per Micro Trench (P7-P8 e P9-P10)
        if i == 7: # Linea per Micro Left
            p7_orig = punti_selezionati[6]
            cv2.line(immagine_modificabile, p7_orig, p_screen, (255, 0, 255), 1)
        elif i == 9: # Linea per Micro Right
            p9_orig = punti_selezionati[8]
            cv2.line(immagine_modificabile, p9_orig, p_screen, (0, 0, 255), 1)

    # 4. Visualizzazione stato
    fasi = ["Gradino/Angolo/Faceting (P1-P4)", "Larghezza Trincea (P5-P6)", 
            "Micro Trench Sinistro (P7-P8)", "Micro Trench Destro (P9-P10)"]
    
    if fase_corrente < len(fasi):
        stato_fase = fasi[fase_corrente]
    else:
        stato_fase = "Misurazione completata"
                 
    cv2.putText(immagine_modificabile, f"Fase {fase_corrente+1} / {len(fasi)}: {stato_fase}", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(immagine_modificabile, f"Punti Totali Selezionati: {len(punti_selezionati)}", (10, 60), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    cv2.imshow(window_name, immagine_modificabile)

# ==============================================================================
# FUNZIONI DI CALCOLO GEOMETRICO
# ==============================================================================

def project_point_onto_line(p, p1, p2):
    """ Calcola la proiezione di un punto (p) sulla retta definita da p1 e p2. """
    p, p1, p2 = np.array(p), np.array(p1), np.array(p2)
    v_line = p2 - p1 # Vettore P1->P2
    
    line_magnitude_sq = np.dot(v_line, v_line)
    
    if line_magnitude_sq < 1e-6: # Punti coincidenti, non c'è retta
        return p1
        
    v_point = p - p1 # Vettore P1->P
    t = np.dot(v_point, v_line) / line_magnitude_sq
    p_projected = p1 + t * v_line
    return p_projected.astype(int)

def gradino_angolo_faceting(punti, scale_factor):
    """ 
    Calcola il gradino (P4 vs P1-P2), l'angolo (P1-P3-P2) e il faceting (distanza orizzontale P4 vs P'4).
    """
    if len(punti) < 4:
        return 0.0, 0.0, 0.0, 0, 0
    
    p1, p2, p3, p4 = np.array(punti[0]), np.array(punti[1]), np.array(punti[2]), np.array(punti[3])
    
    # --- 1. Gradino: Distanza perpendicolare da P4 (Fondo) alla retta P1-P2 (Superficie)
    v_1 = p2 - p1
    v_step = p4 - p1
    line_magnitude = np.linalg.norm(v_1)
    
    gradino_dist_px = 0.0
    if line_magnitude > 1e-6:
        # Distanza Punto-Retta: |(x2-x1)(y4-y1) - (y2-y1)(x4-x1)| / line_magnitude
        cross_product = np.abs(v_1[0] * v_step[1] - v_1[1] * v_step[0])
        gradino_dist_px = cross_product / line_magnitude
    elif np.linalg.norm(v_step) > 0: # Caso P1=P2, usa distanza P1-P4
         gradino_dist_px = np.linalg.norm(v_step)
        
    gradino_unit = gradino_dist_px * scale_factor

    # --- 2. Faceting: Distanza orizzontale (Delta X) tra P4 e la sua proiezione P'4
    p4_projected = project_point_onto_line(p4, p1, p2)
    faceting_px = abs(p4[0] - p4_projected[0])
    faceting_unit = faceting_px * scale_factor
    
    # --- 3. Angolo: Angolo formato da P1-P3 e P2-P3 (Vertice P3)
    v1_angle = p1 - p3 # Vettore P3 -> P1
    v2_angle = p2 - p3 # Vettore P3 -> P2
    
    dot_product = np.dot(v1_angle, v2_angle)
    magnitudes = np.linalg.norm(v1_angle) * np.linalg.norm(v2_angle)
    
    angolo_gradi = 0.0
    if magnitudes > 1e-6:
        cos_angle = np.clip(dot_product / magnitudes, -1.0, 1.0)
        angolo_rad = np.arccos(cos_angle)
        angolo_gradi = np.degrees(angolo_rad)
        
    # Delta X e Y per il log (P4 rispetto a P1 per riferimento)
    delta_x_px = p4[0] - p1[0]
    delta_y_px = p4[1] - p1[1]

    return gradino_unit, angolo_gradi, faceting_unit, delta_x_px, delta_y_px

def microtrenching_calc(p_start, p_end, scale_factor):
    """ Calcola la larghezza del microtrenching come distanza Euclidea tra i due punti. """
    dist_px = np.linalg.norm(np.array(p_end) - np.array(p_start))
    dist_unit = dist_px * scale_factor
    return dist_unit, dist_px

# ==============================================================================
# CALLBACK DEL MOUSE (Pannning e Selezione Punti)
# ==============================================================================

def on_mouse(event, x, y, flags, param):
    """
    Funzione di callback per catturare gli eventi del mouse (click e pan).
    Gestisce la selezione sequenziale di tutti i 10 punti (P1-P10).
    """
    global punti_selezionati, current_mode, reference_line_params, fase_corrente
    global is_panning, last_mouse_pos, current_pan_offset, filepath_global, PIXEL_TO_UNIT_FACTOR

    x_orig, y_orig = to_original_coords(x, y) 

    if event == cv2.EVENT_LBUTTONDOWN:
        
        if not is_panning:
            # Fase 0: Gradino/Angolo/Faceting (P1, P2, P3, P4)
            if fase_corrente == 0 and len(punti_selezionati) < 4:
                punti_selezionati.append((x_orig, y_orig))
                print(f"Punto selezionato (Originale): ({x_orig}, {y_orig}) - P{len(punti_selezionati)} di 4 (Fase 1)")
                redraw_image() 

                if len(punti_selezionati) == 2:
                    p1, p2 = punti_selezionati[0], punti_selezionati[1]
                    x1, y1 = p1
                    x2, y2 = p2
                    
                    if abs(x2 - x1) > 1e-6:
                        m = (y2 - y1) / (x2 - x1)
                        q = y1 - m * x1
                        reference_line_params = (m, q) # Retta P1-P2
                    else:
                        reference_line_params = (None, x1) # Retta verticale

                if len(punti_selezionati) == 4:
                    # Calcolo e log per Gradino/Angolo/Faceting
                    gradino_unit_val, angolo_gradi_val, faceting_unit_val, delta_x_px, delta_y_px = gradino_angolo_faceting(punti_selezionati, PIXEL_TO_UNIT_FACTOR) 
                    
                    delta_x_unit = delta_x_px * PIXEL_TO_UNIT_FACTOR
                    delta_y_unit = delta_y_px * PIXEL_TO_UNIT_FACTOR

                    RESULTS_LOG.append({
                        'image_path': filepath_global,
                        'mode': 'GRADINO_ANGOLO_FACETING',
                        'angolo_gradi': angolo_gradi_val, 
                        'gradino_unit': gradino_unit_val, 
                        'faceting_unit': faceting_unit_val, # Nuovo dato
                        'delta_x_px': delta_x_px,
                        'delta_y_px': delta_y_px,
                        'delta_x_unit': delta_x_unit,
                        'delta_y_unit': delta_y_unit,
                        'scale_factor': PIXEL_TO_UNIT_FACTOR
                    })
                    
                    print(f"Risultato Gradino/Angolo/Faceting salvato. Gradino: {gradino_unit_val:.4f} u, Angolo: {angolo_gradi_val:.2f} deg, Faceting: {faceting_unit_val:.4f} u.") 
                    current_mode = 'NONE' # Aspetta 'c' per passare alla fase successiva
            
            # Fase 1: Trincea Larghezza (P6, P7) - Proiezione sulla retta trincea (m_trincea, q_trincea)
            elif fase_corrente == 1 and len(punti_selezionati) == 5:
                if reference_line_params is None:
                    print("Errore: Retta della trincea non definita. Ritorna al posizionamento con 'z'.")
                    return

                m, q = reference_line_params
                
                # Calcola il punto proiettato sulla retta trincea
                p_proiettato = (0, 0)
                if m is None: # Retta verticale (x = q)
                    p_proiettato = (q, y_orig) 
                elif abs(m) < 1e-6: # Retta orizzontale (y = q)
                    p_proiettato = (x_orig, q)
                else:
                    p_proiettato_np = project_point_onto_line((x_orig, y_orig), (0, q), (100, int(m*100 + q)))
                    p_proiettato = tuple(p_proiettato_np.tolist())
                    
                punti_selezionati.append(p_proiettato)
                print(f"Punto P{len(punti_selezionati)} selezionato e proiettato su retta: {p_proiettato}")
                redraw_image()

            # Fase 2: Micro Trench Sinistro (P8, P9) - Clic diretto (Non proiettato)
            elif fase_corrente == 2 and len(punti_selezionati) >= 7 and len(punti_selezionati) < 9:
                punti_selezionati.append((x_orig, y_orig))
                print(f"Punto selezionato: ({x_orig}, {y_orig}) - P{len(punti_selezionati)} di 2 (Fase 3)")
                redraw_image()

            # Fase 3: Micro Trench Destro (P10, P11) - Clic diretto (Non proiettato)
            elif fase_corrente == 3 and len(punti_selezionati) >= 9 and len(punti_selezionati) < 11:
                punti_selezionati.append((x_orig, y_orig))
                print(f"Punto selezionato: ({x_orig}, {y_orig}) - P{len(punti_selezionati)} di 2 (Fase 4)")
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
    Estrae il fattore di scala dai metadati. (Logica invariata)
    """
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
            fattore_di_scala_um_per_px = pixel_width_m * 1e6
            
            print(f"Risoluzione trovata in {pixel_width_source}: {fattore_di_scala_um_per_px:.8f} µm/px.")
            return fattore_di_scala_um_per_px
        except (ValueError, TypeError, KeyError):
            print(f"Attenzione: '{pixel_width_source}' non è un valore numerico valido. Proseguo la ricerca.")
            pass

    print("Nessun metadato valido trovato. Uso il valore di default 1.0 µm/px.")
    return 1.0

# ==============================================================================
# FUNZIONE PRINCIPALE DI MISURAZIONE INTERATTIVA (Multi-Fase Estesa)
# ==============================================================================

import tifffile
import cv2
import os
import numpy as np
import xml.etree.ElementTree as ET
from lxml import etree
import datetime
import math

punti_selezionati = [] # Da P1 (indice 0) a P10 (indice 9)
immagine_originale = None
immagine_modificabile = None
window_name = "Misurazione Interattiva"

# Fattori di scala e stato di interazione
scale_factor_um_per_px = 1.0 # Fattore di scala fisico (µm/px)
PIXEL_TO_UNIT_FACTOR = 1.0 # Alias per scale_factor_um_per_px
misurazioni_salvate_parziali = {}
# Fasi aggiornate (P1-P5 in Fase 0, P6-P7 in Fase 1, P8-P9 in Fase 2, P10-P11 in Fase 3)
fase_corrente = 0 # 0: Gradino/Angolo/Faceting (P1-P5), 1: Trincea Larghezza (P6-P7), 2: Micro Left (P8-P9), 3: Micro Right (P10-P11)

# Variabili per la visualizzazione e interazione (Pan)
visual_scale_factor = 1.0 
current_pan_offset = [0, 0] # Offset di spostamento per il panning
is_panning = False
last_mouse_pos = (0, 0)

# Variabili di stato usate nel codice
current_mode = 'NONE'     # Modo corrente: 'GRADINO', 'TRINCEA_LINE', 'TRINCEA_P', 'MICRO_L', 'MICRO_R', 'NONE'
reference_line_params = None  # (m, q) o (None, x1) per la retta P1-P2 (Top Surface)
filepath_global = ""
RESULTS_LOG = [] # Per memorizzare i risultati prima del salvataggio XML
SEZIONE_NOTE = "\n\n---\n## Note dell'Analisi\n\n[NOTE]"

def misura_immagine_interattiva(immagine_originale_in, filepath, scale_factor):
    """
    Gestisce il ciclo interattivo per la selezione dei punti di misurazione.
    """
    global immagine_originale, immagine_modificabile, punti_selezionati, window_name
    global fase_corrente, filepath_global, PIXEL_TO_UNIT_FACTOR, reference_line_params
    global RESULTS_LOG, misurazioni_salvate_parziali
    
    # Setup iniziale... (omesso per brevità)
    punti_selezionati = []
    fase_corrente = 0
    filepath_global = filepath
    PIXEL_TO_UNIT_FACTOR = scale_factor
    immagine_originale = immagine_originale_in.copy()
    
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, on_mouse)
    
    # 1. VISUALIZZAZIONE INIZIALE E FIX DEL BUG
    redraw_image() 
    
    # >>> INSERIMENTO DEL FIX QUI <<<
    # Questo attende 100ms per dare tempo al sistema operativo di disegnare la finestra
    # PRIMA che il ciclo while prenda il controllo con waitKey(1).
    cv2.waitKey(100) 
    
    print("Premi 'c' per passare alla fase successiva, 'x' per annullare l'ultima selezione, 's' per salvare e procedere, 'q' per saltare l'immagine.")
    
    # Inizializzazione Log Risultati per l'immagine corrente
    risultato_immagine = {
        'data_misurazione': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'file_name': os.path.basename(filepath),
        'scale_factor': PIXEL_TO_UNIT_FACTOR,
        'gradino_um': None,
        'angolo_gradi': None,
        'faceting_um': None,
        'trincea_larghezza_um': None,
        'micro_left_um': None,
        'micro_right_um': None,
        'status': 'Incomplete'
    }

    # ====================================================================
    # CICLO DI MISURAZIONE INTERATTIVA
    # ====================================================================

    # Fase 0: Gradino/Angolo/Faceting (P1-P5)
    while fase_corrente == 0:
        # Punti da 0 a 4 (totale 5 punti)
        if len(punti_selezionati) == 5:
            # Calcolo e log per Gradino/Angolo/Faceting
            p1, p2, p3, p4, p5 = punti_selezionati[0:5]
            gradino_unit_val, angolo_gradi_val, _, delta_x_prime_px, delta_y_prime_px = gradino_angolo_faceting(punti_selezionati, PIXEL_TO_UNIT_FACTOR) 
            
            # Calcolo del Faceting (distanza proiettata P3 - P5)
            # P5 è già proiettato. Dobbiamo proiettare anche P3 sulla retta P1-P2.
            p3_proj = project_point_onto_line(p3, p1, p2)
            
            # Trasformazione nel sistema solidale per calcolare Faceting come Delta X' tra P3' e P5'
            angle_p1p2 = get_p1p2_angle(p1, p2)
            p3_x_prime, _ = transform_point(p3_proj, p1, angle_p1p2)
            p5_x_prime, _ = transform_point(p5, p1, angle_p1p2) # P5 è proiettato, quindi Y' è ~0

            faceting_px = abs(p3_x_prime - p5_x_prime)
            faceting_unit_val = faceting_px * PIXEL_TO_UNIT_FACTOR

            risultato_immagine.update({
                'gradino_um': gradino_unit_val,
                'angolo_gradi': angolo_gradi_val,
                'faceting_um': faceting_unit_val,
                'delta_x_prime_px': delta_x_prime_px,
                'delta_y_prime_px': delta_y_prime_px
            })
            
            print(f"Risultato Gradino/Angolo/Faceting salvato. Gradino: {gradino_unit_val:.4f} u, Angolo: {angolo_gradi_val:.2f} deg, Faceting (P3'-P5'): {faceting_unit_val:.4f} u.") 
            print("Premi 'c' per procedere alla misurazione Trincea (P6-P7).")
        
        k = cv2.waitKey(1) & 0xFF
        
        # Gestione input 'c', 'x', 's', 'q' ... (omesso per brevità)
        if k == ord('c'):
            if len(punti_selezionati) == 5:
                fase_corrente = 1 # Passa alla Fase 1
                redraw_image()
            else:
                print("Devi prima selezionare 5 punti (P1-P5).")
        # ... (altre condizioni 'x', 's', 'q') ...
        elif k == ord('q'): # Skip image
            risultato_immagine['status'] = 'Skipped'
            return risultato_immagine
        elif k == ord('s'): # Save and exit
            risultato_immagine['status'] = 'Completed' if len(punti_selezionati) >= 5 else 'Incomplete'
            return risultato_immagine
        elif k == ord('x') or k == 27: # Esc or x to undo
            if len(punti_selezionati) > 0:
                punti_selezionati.pop()
                print(f"Ultimo punto rimosso. Punti rimanenti: {len(punti_selezionati)}")
                if len(punti_selezionati) < 2:
                    reference_line_params = None
                redraw_image()
                # Cancella i risultati se si annulla la selezione completa della fase
                if len(punti_selezionati) < 5:
                    risultato_immagine['gradino_um'] = None 
                    risultato_immagine['angolo_gradi'] = None
                    risultato_immagine['faceting_um'] = None
            else:
                print("Nessun punto da rimuovere.")

    # Fase 1: Larghezza Trincea (P6, P7)
    while fase_corrente == 1:
        # Punti da 5 a 6 (totale 2 punti)
        if len(punti_selezionati) == 7:
            p6, p7 = punti_selezionati[5], punti_selezionati[6]
            # La larghezza della trincea è la distanza euclidea tra P6 e P7 (entrambi proiettati)
            trincea_unit_val, trincea_px_val = microtrenching_calc(p6, p7, p1, p2, PIXEL_TO_UNIT_FACTOR) 
            
            risultato_immagine.update({
                'trincea_larghezza_um': trincea_unit_val,
                'trincea_larghezza_px': trincea_px_val
            })
            
            print(f"Risultato Trincea salvato. Larghezza: {trincea_unit_val:.4f} u.")
            print("Premi 'c' per procedere alla misurazione Micro Trench Sinistro (P8-P9).")

        k = cv2.waitKey(1) & 0xFF
        
        # Gestione input 'c', 'x', 's', 'q' ... (omesso per brevità)
        if k == ord('c'):
            if len(punti_selezionati) == 7:
                fase_corrente = 2 # Passa alla Fase 2
                redraw_image()
            else:
                print("Devi prima selezionare 2 punti (P6-P7).")
        elif k == ord('q'): 
            risultato_immagine['status'] = 'Skipped'
            return risultato_immagine
        elif k == ord('s'):
            risultato_immagine['status'] = 'Completed' if len(punti_selezionati) >= 7 else 'Incomplete'
            return risultato_immagine
        elif k == ord('x') or k == 27: 
            if len(punti_selezionati) > 5:
                punti_selezionati.pop()
                print(f"Ultimo punto rimosso. Punti rimanenti: {len(punti_selezionati)}")
                if len(punti_selezionati) < 7:
                    risultato_immagine['trincea_larghezza_um'] = None 
                redraw_image()
            else:
                print("Nessun punto da rimuovere in questa fase.")
                
    # Fase 2: Micro Trench Sinistro (P8, P9)
    while fase_corrente == 2:
        # Punti da 7 a 8 (totale 2 punti)
        if len(punti_selezionati) == 9:
            p8, p9 = punti_selezionati[7], punti_selezionati[8]
            # Il micro trenching viene misurato come differenza di gradino (Y') tra i due punti
            micro_left_unit_val, micro_left_px_val = microtrenching_calc(p8, p9, p1, p2, PIXEL_TO_UNIT_FACTOR) 
            
            risultato_immagine.update({
                'micro_left_um': micro_left_unit_val,
                'micro_left_px': micro_left_px_val
            })
            
            print(f"Risultato Micro Trench Sinistro salvato. Gradino: {micro_left_unit_val:.4f} u.")
            print("Premi 'c' per procedere alla misurazione Micro Trench Destro (P10-P11).")

        k = cv2.waitKey(1) & 0xFF
        
        # Gestione input 'c', 'x', 's', 'q' ... (omesso per brevità)
        if k == ord('c'):
            if len(punti_selezionati) == 9:
                fase_corrente = 3 # Passa alla Fase 3
                redraw_image()
            else:
                print("Devi prima selezionare 2 punti (P8-P9).")
        elif k == ord('q'): 
            risultato_immagine['status'] = 'Skipped'
            return risultato_immagine
        elif k == ord('s'): 
            risultato_immagine['status'] = 'Completed' if len(punti_selezionati) >= 9 else 'Incomplete'
            return risultato_immagine
        elif k == ord('x') or k == 27: 
            if len(punti_selezionati) > 7:
                punti_selezionati.pop()
                print(f"Ultimo punto rimosso. Punti rimanenti: {len(punti_selezionati)}")
                if len(punti_selezionati) < 9:
                    risultato_immagine['micro_left_um'] = None 
                redraw_image()
            else:
                print("Nessun punto da rimuovere in questa fase.")

    # Fase 3: Micro Trench Destro (P10, P11)
    while fase_corrente == 3:
        # Punti da 9 a 10 (totale 2 punti)
        if len(punti_selezionati) == 11:
            p10, p11 = punti_selezionati[9], punti_selezionati[10]
            # Il micro trenching viene misurato come differenza di gradino (Y') tra i due punti
            micro_right_unit_val, micro_right_px_val = microtrenching_calc(p10, p11, p1, p2, PIXEL_TO_UNIT_FACTOR) 
            
            risultato_immagine.update({
                'micro_right_um': micro_right_unit_val,
                'micro_right_px': micro_right_px_val
            })
            
            print(f"Risultato Micro Trench Destro salvato. Gradino: {micro_right_unit_val:.4f} u.")
            print("Misurazione completata. Premi 's' per salvare o 'q' per saltare.")

        k = cv2.waitKey(1) & 0xFF
        
        # Gestione input 's', 'q' ... (omesso per brevità)
        if k == ord('q'): 
            risultato_immagine['status'] = 'Skipped'
            return risultato_immagine
        elif k == ord('s'):
            risultato_immagine['status'] = 'Completed' if len(punti_selezionati) == 11 else 'Incomplete'
            return risultato_immagine
        elif k == ord('x') or k == 27: 
            if len(punti_selezionati) > 9:
                punti_selezionati.pop()
                print(f"Ultimo punto rimosso. Punti rimanenti: {len(punti_selezionati)}")
                if len(punti_selezionati) < 11:
                    risultato_immagine['micro_right_um'] = None 
                redraw_image()
            else:
                print("Nessun punto da rimuovere in questa fase.")
    
    # Ritorna risultati se il ciclo termina senza 's'/'q' (dovrebbe essere gestito da 's')
    return risultato_immagine

# ==============================================================================
# NUOVE UTILITY GEOMETRICHE (Sistema Solidale P1-P2)
# ==============================================================================

def get_p1p2_angle(p1, p2):
    """ Calcola l'angolo (in radianti) della retta P1-P2 rispetto all'asse X globale. """
    return math.atan2(p2[1] - p1[1], p2[0] - p1[0])

def transform_point(p, p1, angle_rad):
    """ 
    Trasforma un punto P(x, y) nel sistema di coordinate X'Y' solidale a P1-P2.
    L'origine è P1, l'asse X' è P1->P2.
    Ritorna (X', Y').
    """
    p, p1 = np.array(p), np.array(p1)
    
    # 1. Traslazione (Origine in P1)
    p_translated = p - p1
    
    # 2. Rotazione (Allinea X' a P1-P2)
    cos_a = math.cos(-angle_rad)
    sin_a = math.sin(-angle_rad)
    
    # x' = x_trans * cos(-a) - y_trans * sin(-a) = x_trans * cos(a) + y_trans * sin(a)
    # y' = x_trans * sin(-a) + y_trans * cos(-a) = -x_trans * sin(a) + y_trans * cos(a)
    
    x_prime = p_translated[0] * cos_a - p_translated[1] * sin_a
    y_prime = p_translated[0] * sin_a + p_translated[1] * cos_a
    
    return x_prime, y_prime

# ==============================================================================
# FUNZIONI DI BASE (coordinate e disegno)
# ==============================================================================

def to_screen_coords(x_orig, y_orig):
    """
    Funzione fittizia: ora le coordinate originali sono le coordinate schermo.
    Mantenuta per coerenza con il codice di disegno se si usava *p_orig.
    """
    return (int(x_orig), int(y_orig))


def draw_cross(img, center, size, color, thickness=1):
    """
    Disegna una croce centrata in (x, y).
    """
    x, y = center
    # Linea orizzontale
    cv2.line(img, (x - size, y), (x + size, y), color, thickness)
    # Linea verticale
    cv2.line(img, (x, y - size), (x, y + size), color, thickness)


def project_point_onto_line(p, p1, p2):
    """ Calcola la proiezione di un punto (p) sulla retta definita da p1 e p2. """
    p, p1, p2 = np.array(p), np.array(p1), np.array(p2)
    v_line = p2 - p1 # Vettore P1->P2
    
    line_magnitude_sq = np.dot(v_line, v_line)
    
    if line_magnitude_sq < 1e-6: # Punti coincidenti
        return p1
        
    v_point = p - p1 # Vettore P1->P
    t = np.dot(v_point, v_line) / line_magnitude_sq
    p_projected = p1 + t * v_line
    return p_projected.astype(int)


# ==============================================================================
# FUNZIONI DI CALCOLO AGGIORNATE
# ==============================================================================

def gradino_angolo_faceting(punti, scale_factor):
    """ 
    Calcola il gradino (Y' di P4 rispetto P1-P2) e l'angolo (P1-P3-P2).
    Il Faceting P3-P5 viene calcolato nel flusso interattivo.
    """
    if len(punti) < 4:
        # Ritorna: gradino_unit, angolo_gradi, Faceting (placeholder), delta_x_prime_px, delta_y_prime_px
        return 0.0, 0.0, 0.0, 0, 0
    
    p1, p2, p3, p4 = np.array(punti[0]), np.array(punti[1]), np.array(punti[2]), np.array(punti[3])
    
    # --- 1. Gradino (Y' di P4): Distanza perpendicolare da P4 alla retta P1-P2
    v_1 = p2 - p1
    line_magnitude = np.linalg.norm(v_1)
    gradino_dist_px = 0.0
    
    if line_magnitude > 1e-6:
        # Distanza Punto-Retta (equivalente alla componente Y' di P4 nel sistema solidale)
        cross_product = np.abs((p2[0]-p1[0])*(p4[1]-p1[1]) - (p2[1]-p1[1])*(p4[0]-p1[0]))
        gradino_dist_px = cross_product / line_magnitude
    elif np.linalg.norm(p4 - p1) > 0: 
         gradino_dist_px = np.linalg.norm(p4 - p1)
        
    gradino_unit = gradino_dist_px * scale_factor

    # --- 2. Angolo: Angolo formato da P1-P3 e P2-P3 (Vertice P3)
    v1_angle = p1 - p3
    v2_angle = p2 - p3 
    
    dot_product = np.dot(v1_angle, v2_angle)
    magnitudes = np.linalg.norm(v1_angle) * np.linalg.norm(v2_angle)
    
    angolo_gradi = 0.0
    if magnitudes > 1e-6:
        cos_angle = np.clip(dot_product / magnitudes, -1.0, 1.0)
        angolo_rad = np.arccos(cos_angle)
        angolo_gradi = np.degrees(angolo_rad)
        
    # --- 3. Dati X'Y' per il log (usando il sistema solidale)
    angle_p1p2 = get_p1p2_angle(p1, p2)
    p4_x_prime, p4_y_prime = transform_point(p4, p1, angle_p1p2)
    
    # Log delle coordinate solidali di P4
    delta_x_prime_px = p4_x_prime
    delta_y_prime_px = p4_y_prime 

    return gradino_unit, angolo_gradi, 0.0, delta_x_prime_px, delta_y_prime_px

def microtrenching_calc(p_start, p_end, p1, p2, scale_factor):
    """ 
    Calcola la distanza tra P_start e P_end come differenza di 'Gradino' (componente Y' nel sistema solidale P1-P2).
    Questo è l'equivalente della distanza tra rette parallele alla retta di riferimento.
    Ritorna la differenza in Y' (gradino).
    """
    p_start, p_end, p1, p2 = np.array(p_start), np.array(p_end), np.array(p1), np.array(p2)
    
    if np.all(p1 == p2):
        # Se la retta P1-P2 è degenere, usiamo la distanza Euclidea
        dist_px = np.linalg.norm(p_end - p_start)
        return dist_px * scale_factor, dist_px
    
    angle_p1p2 = get_p1p2_angle(p1, p2)
    
    # 1. Trasformazione in sistema solidale (X', Y') con P1 come origine
    p_start_x_prime, p_start_y_prime = transform_point(p_start, p1, angle_p1p2)
    p_end_x_prime, p_end_y_prime = transform_point(p_end, p1, angle_p1p2)
    
    # 2. Calcola la differenza di 'Gradino' (Y') tra i due punti
    y_prime_diff_px = abs(p_end_y_prime - p_start_y_prime)
    
    return y_prime_diff_px * scale_factor, y_prime_diff_px

# ==============================================================================
# FUNZIONI DI VISUALIZZAZIONE E INTERAZIONE (SENZA PAN/ZOOM)
# ==============================================================================

def redraw_image():
    """
    Funzione per ridisegnare l'immagine sullo schermo. Tutte le coordinate
    sono considerate coordinate immagine (1:1).
    """
    global immagine_originale, immagine_modificabile, punti_selezionati, window_name
    global current_mode, reference_line_params, fase_corrente

    if immagine_originale is None:
        return

    # L'immagine modificabile è una copia dell'originale
    immagine_modificabile = immagine_originale.copy()  
    
    # 1. Disegna la retta di riferimento P1-P2 (Top Surface)
    if len(punti_selezionati) >= 2:
        p1_orig, p2_orig = punti_selezionati[0], punti_selezionati[1]
        
        # Usiamo direttamente le coordinate originali
        cv2.line(immagine_modificabile, p1_orig, p2_orig, (255, 0, 0), 2) # Blu
        
        # Disegna la linea di gradino P4
        if len(punti_selezionati) >= 4:
            p4_orig = punti_selezionati[3]
            p4_proj_orig = project_point_onto_line(p4_orig, p1_orig, p2_orig)
            cv2.line(immagine_modificabile, p4_orig, 
                     p4_proj_orig.tolist(), (0, 0, 255), 1) # Rosso (Gradino)
                     
        # Disegna la linea di faceting P3-P5 (asse X')
        # NOTA: P5 è già proiettato
        if len(punti_selezionati) >= 5:
            p3_orig = punti_selezionati[2]
            p5_orig = punti_selezionati[4]

            p3_proj_orig = project_point_onto_line(p3_orig, p1_orig, p2_orig)
            cv2.line(immagine_modificabile, p3_proj_orig.tolist(), 
                     p5_orig, (0, 165, 255), 1) # Arancione

    # 2. Disegna la retta della trincea mobile (se in Fase 1, 2 o 3)
    # NB: Questa è la linea che P6 e P7 definiscono per la larghezza.
    if fase_corrente >= 1 and reference_line_params is not None and len(punti_selezionati) < 11:
        m, q = reference_line_params
        rows, cols, _ = immagine_modificabile.shape
        
        # Logica per disegnare la retta della trincea mobile (ciano)
        if m is None:
            start_p = (q, 0)
            end_p = (q, rows)
        else:
            y_start = int(m * 0 + q)
            y_end = int(m * (cols - 1) + q)
            start_p = (0, y_start)
            end_p = (cols - 1, y_end)

        # Usiamo direttamente le coordinate originali (che sono le coordinate schermo)
        cv2.line(immagine_modificabile, start_p, end_p, (255, 255, 0), 2)


    # 3. Disegna tutti i punti selezionati (P1-P11)
    point_names = ['P1', 'P2', 'P3', 'P4', 'P5', 'P6', 'P7', 'P8', 'P9', 'P10', 'P11']
    for i, p_orig in enumerate(punti_selezionati):
        p_screen = p_orig # Le coordinate sono le coordinate schermo 1:1
        label = point_names[i]
        
        # Assegnazione colore in base alla fase
        if i < 2:
            color = (0, 255, 0) # Verde (Riferimento P1, P2)
        elif i < 5:
            color = (0, 255, 255) # Giallo (Faceting P3, P4, P5)
        elif i < 7:
            color = (255, 0, 255) # Magenta (Trincea P6, P7)
        elif i < 9:
            color = (255, 0, 0) # Blu (Micro Left P8, P9)
        else:
            color = (0, 0, 255) # Rosso (Micro Right P10, P11)

        draw_cross(immagine_modificabile, p_screen, 6, color, 1)
        cv2.putText(immagine_modificabile, label, (p_screen[0] + 11, p_screen[1] - 11), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        # Disegna la linea di misura per Micro Trench (P8-P9 e P10-P11)
        if i == 8: # Linea per Micro Left P8-P9
            p8_orig = punti_selezionati[7]
            cv2.line(immagine_modificabile, p8_orig, p_screen, (255, 0, 0), 1)
        elif i == 10: # Linea per Micro Right P10-P11
            p10_orig = punti_selezionati[9]
            cv2.line(immagine_modificabile, p10_orig, p_screen, (0, 0, 255), 1)

    # 4. Visualizzazione stato
    fasi = ["Gradino/Angolo/Faceting (P1-P5)", "Larghezza Trincea (P6-P7)", 
            "Micro Trench Sinistro (P8-P9)", "Micro Trench Destro (P10-P11)"]
    
    if fase_corrente < len(fasi):
        stato_fase = fasi[fase_corrente]
    else:
        stato_fase = "Misurazione completata"
                 
    cv2.putText(immagine_modificabile, f"Fase {fase_corrente+1} / {len(fasi)}: {stato_fase}", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(immagine_modificabile, f"Punti Totali Selezionati: {len(punti_selezionati)}", (10, 60), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    cv2.imshow(window_name, immagine_modificabile)

def on_mouse(event, x, y, flags, param):
    """
    Funzione di callback per catturare gli eventi del mouse (solo click).
    Gestisce la selezione sequenziale dei 11 punti con la logica di proiezione.
    """
    global punti_selezionati, current_mode, reference_line_params, fase_corrente
    global filepath_global, PIXEL_TO_UNIT_FACTOR

    # x e y sono le coordinate originali, poiché non c'è pan/zoom
    x_orig, y_orig = x, y 

    if event == cv2.EVENT_LBUTTONDOWN:
        
        # Punti P1 e P2: definiscono la retta di riferimento
        if len(punti_selezionati) < 2:
            punti_selezionati.append((x_orig, y_orig))
            print(f"Punto selezionato (Originale): ({x_orig}, {y_orig}) - P{len(punti_selezionati)}")
            redraw_image() 

            if len(punti_selezionati) == 2:
                p1, p2 = punti_selezionati[0], punti_selezionati[1]
                x1, y1 = p1
                x2, y2 = p2
                
                if abs(x2 - x1) > 1e-6:
                    m = (y2 - y1) / (x2 - x1)
                    q = y1 - m * x1
                    reference_line_params = (m, q) # Retta P1-P2
                else:
                    reference_line_params = (None, x1) # Retta verticale

        # Fase 0: Gradino/Angolo/Faceting (P3, P4, P5)
        elif fase_corrente == 0 and len(punti_selezionati) < 5:
            # P5 (indice 4) DEVE essere proiettato sulla retta P1-P2
            if len(punti_selezionati) == 4: 
                p1, p2 = punti_selezionati[0], punti_selezionati[1]
                p_proiettato_np = project_point_onto_line((x_orig, y_orig), p1, p2)
                p_proiettato = tuple(p_proiettato_np.tolist())
                punti_selezionati.append(p_proiettato)
                print(f"Punto P5 selezionato e PROIETTATO sulla retta rif.: {p_proiettato}")
            else: # P3, P4
                punti_selezionati.append((x_orig, y_orig))
                print(f"Punto selezionato (Originale): ({x_orig}, {y_orig}) - P{len(punti_selezionati)}")
                
            redraw_image() 

        # Fase 1: Trincea Larghezza (P6, P7) - Proiezione sulla retta trincea mobile
        elif fase_corrente == 1 and len(punti_selezionati) >= 5 and len(punti_selezionati) < 7:
            if reference_line_params is None:
                print("Errore: Retta della trincea non definita.")
                return

            m, q = reference_line_params
            
            # Calcola il punto proiettato sulla retta trincea mobile (che è la retta P1-P2)
            p_proiettato = (0, 0)
            if m is None: # Retta verticale (x = q)
                p_proiettato = (q, y_orig) 
            elif abs(m) < 1e-6: # Retta orizzontale (y = q)
                p_proiettato = (x_orig, q)
            else:
                # Usiamo due punti fittizi sulla retta P1-P2 per proiettare
                p1_ref, p2_ref = punti_selezionati[0], punti_selezionati[1]
                p_proiettato_np = project_point_onto_line((x_orig, y_orig), p1_ref, p2_ref)
                p_proiettato = tuple(p_proiettato_np.tolist())
                
            punti_selezionati.append(p_proiettato)
            print(f"Punto P{len(punti_selezionati)} selezionato e proiettato su retta trincea (P1-P2): {p_proiettato}")
            redraw_image()

        # Fase 2: Micro Trench Sinistro (P8, P9) - Clic diretto
        elif fase_corrente == 2 and len(punti_selezionati) >= 7 and len(punti_selezionati) < 9:
            punti_selezionati.append((x_orig, y_orig))
            print(f"Punto selezionato: ({x_orig}, {y_orig}) - P{len(punti_selezionati)}")
            redraw_image()

        # Fase 3: Micro Trench Destro (P10, P11) - Clic diretto
        elif fase_corrente == 3 and len(punti_selezionati) >= 9 and len(punti_selezionati) < 11:
            punti_selezionati.append((x_orig, y_orig))
            print(f"Punto selezionato: ({x_orig}, {y_orig}) - P{len(punti_selezionati)}")
            redraw_image()
            
    # Rimozione completa della logica di pan/zoom/scroll
    # L'unica logica rimasta è il click sinistro (LBUTTONDOWN)
    pass # Nessuna azione per RBUTTONDOWN, RBUTTONUP, MOUSEMOVE

def get_scale_factor(tif):
    """ Estrae il fattore di scala dai metadati. (Logica invariata) """
    
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
            fattore_di_scala_um_per_px = pixel_width_m * 1e6
            
            print(f"Risoluzione trovata in {pixel_width_source}: {fattore_di_scala_um_per_px:.8f} µm/px.")
            return fattore_di_scala_um_per_px
        except (ValueError, TypeError, KeyError):
            print(f"Attenzione: '{pixel_width_source}' non è un valore numerico valido. Proseguo la ricerca.")
            pass

    print("Nessun metadato valido trovato. Uso il valore di default 1.0 µm/px.")
    return 1.0

# ==============================================================================
# FUNZIONI DI SALVATAGGIO E LOGICA SEQUENZIALE
# ==============================================================================

def salva_risultati_xml(nuovi_dati, nome_file):
    """ Salva o aggiorna i risultati in un file XML. (Logica invariata) """
    
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
    
    for item in nuovi_dati:
        if item.get("status") == "Skipped":
            continue

        misura_element = ET.SubElement(root, "Misura", filename=item['file_name'])
        ET.SubElement(misura_element, "DataMisurazione").text = item.get("data_misurazione", "N/A")
        
        for key, value in item.items():
            if key not in ['file_name', 'data_acquisizione', 'data_misurazione', 'status']: 
                ET.SubElement(misura_element, key).text = str(value)
    
    if etree:
        xml_string = ET.tostring(root, encoding='utf-8')
        parser = etree.XMLParser(remove_blank_text=True)
        tree_lxml = etree.fromstring(xml_string, parser)
        
        with open(nome_file, "wb") as f:
            f.write(etree.tostring(tree_lxml, pretty_print=True, xml_declaration=True, encoding='UTF-8'))
    else:
        ET.indent(tree, space="    ", level=0)
        tree.write(nome_file, encoding='utf-8', xml_declaration=True)
    
    return len([item for item in nuovi_dati if item.get("status") != "Skipped"])
        
def elabora_immagini_sequential(cartella_base):
    """ Itera su tutte le immagini e gestisce la sequenza di misurazione. (Logica invariata) """
    
    global RESULTS_LOG
    RESULTS_LOG = [] 
    risultati_sessione = [] 
    
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
                        with tifffile.TiffFile(percorso_completo) as tif:
                            scale_factor = get_scale_factor(tif)

                        immagine = tifffile.imread(percorso_completo)
                        if immagine.dtype != np.uint8:
                            immagine_norm = cv2.normalize(immagine, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
                        else:
                            immagine_norm = immagine
                        
                        # 3. Misurazione interattiva
                        risultati_immagine = misura_immagine_interattiva(immagine_norm.copy(), percorso_completo, scale_factor)
                        
                        cv2.destroyAllWindows() 
                        
                        if risultati_immagine.get("status") == "Skipped":
                            print(f"Immagine {filename} saltata per richiesta utente ('q').")
                        elif not risultati_immagine.get("gradino_um") and not risultati_immagine.get("trincea_larghezza_um"):
                            print(f"Misurazione di {filename} incompleta ('x' o interruzione). Dati parziali salvati nel log.")
                        else:
                            risultati_immagine["file_name"] = filename
                            
                            # SEZIONE PER LE NOTE ESTESE (Richieste dell'utente)
                            print("\n--- INSERIMENTO ANNOTAZIONI AGGIUNTIVE ---")
                            
                            # RICHIESTA 1: Polimero su parete
                            polimero_parete = input("Presenza di polimero su parete (S/N/D-Dubbio) [N]: ") or "N"
                            risultati_immagine["polimero_su_parete"] = polimero_parete.upper()
                            
                            # RICHIESTA 2: Rideposizione sul fondo
                            rideposizione_fondo = input("Rideposizione di polimero sul fondo scavo (S/N/D-Dubbio) [N]: ") or "N"
                            risultati_immagine["rideposizione_fondo"] = rideposizione_fondo.upper()

                            # RICHIESTA 3: Note libere
                            note_utente = input(f"Note libere per {filename} (opzionale): ")
                            risultati_immagine["note"] = note_utente if note_utente else "Nessuna nota aggiunta."
                            
                            misure_salvate = salva_risultati_xml([risultati_immagine], 'riepilogo_misure.xml')
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
