import tifffile
import cv2
import os
import numpy as np
import sys
from lxml import etree

# --- Variabili Globali ---
punti_selezionati = []
immagine_originale = None
immagine_modificabile = None
window_name = ""
current_mode = None  # 'GRADINO', 'ANGOLO', 'TRINCEA', ecc.

# Parametri della retta di riferimento (m, q) calcolati dai primi 2 punti del gradino.
reference_line_params = None 

scan_line_y = None   # Coordinata Y della linea di scansione/riferimento (utilizzata per la Trincea)
trench_width_data = {} # Dati temporanei per la misurazione della trincea

# --- Funzioni di Utility ---

def salva_risultati_xml(dati, nome_file):
    """
    Salva una lista di dizionari in un unico file XML.
    """
    root = etree.Element("RiepilogoMisure")
    
    for item in dati:
        # Crea un elemento Misura con il nome del file
        misura_element = etree.SubElement(root, "Misura", filename=item.get('file_name', 'N/D'))
        
        # Aggiungi tutti gli altri risultati come sotto-elementi
        for key, value in item.items():
            if key != 'file_name':
                etree.SubElement(misura_element, key).text = str(value)
    
    albero = etree.ElementTree(root)
    try:
        with open(nome_file, "wb") as f:
            albero.write(f, pretty_print=True, xml_declaration=True, encoding='UTF-8')
    except Exception as e:
        print(f"Errore durante il salvataggio XML: {e}")

def get_line_profile(immagine_norm, y_coord):
    """Estrae il profilo dei livelli di grigio lungo la coordinata Y."""
    if y_coord is not None and 0 <= y_coord < immagine_norm.shape[0]:
        return immagine_norm[y_coord, :]
    return None

def plot_profile(profile_data, title="Profilo Intensità"):
    """
    Simula la visualizzazione di un profilo (poiché non possiamo usare matplotlib).
    Stampa i dati chiave del profilo (min, max, media).
    """
    if profile_data is not None:
        print(f"\n--- {title} ---")
        print(f"Min Intensità: {np.min(profile_data):.2f}")
        print(f"Max Intensità: {np.max(profile_data):.2f}")
        print(f"Media Intensità: {np.mean(profile_data):.2f}")
        # In un'applicazione reale, qui si aprirebbe un grafico
    else:
        print("Profilo non disponibile. Selezionare prima una linea di scansione (tasto 'L' o 'X').")

def auto_trench_detect(profile_data, trench_width_data):
    """
    Applica l'algoritmo di Otsu's per la soglia automatica sulla linea selezionata.
    """
    if profile_data is None:
        print("Impossibile eseguire la soglia: selezionare prima una linea di scansione (tasto 'L' o 'X').")
        return False
    
    try:
        # La soglia di Otsu funziona su dati 8-bit
        threshold_value, binarized_line = cv2.threshold(profile_data, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        binarized_line = np.where(binarized_line == 255, 1, 0)
        
        trench_width_data['threshold_metodo'] = 'Otsu_Auto'
        trench_width_data['threshold_valore'] = f"{threshold_value:.2f}"
        trench_width_data['line_binarizzata'] = binarized_line 
        print(f"Soglia Otsu calcolata: {threshold_value:.2f}")
        
        return True
    except Exception as e:
        print(f"Errore nel calcolo Otsu: {e}")
        return False

def manual_threshold(profile_data, trench_width_data, manual_t):
    """Imposta manualmente la soglia."""
    if profile_data is None:
        print("Impossibile eseguire la soglia: selezionare prima una linea di scansione (tasto 'L' o 'X').")
        return False
        
    try:
        T = int(manual_t)
        if not (0 <= T <= 255):
            print("Valore soglia non valido. Deve essere tra 0 e 255 (8-bit).")
            return False

        _, binarized_line = cv2.threshold(profile_data, T, 255, cv2.THRESH_BINARY)
        binarized_line = np.where(binarized_line == 255, 1, 0)
        
        trench_width_data['threshold_metodo'] = 'Manuale'
        trench_width_data['threshold_valore'] = f"{T}"
        trench_width_data['line_binarizzata'] = binarized_line
        print(f"Soglia Manuale impostata: {T}")

        return True
    except ValueError:
        print("Errore: il valore manuale deve essere un numero intero.")
        return False
    except Exception as e:
        print(f"Errore nel calcolo manuale: {e}")
        return False

def calculate_trench_width(trench_width_data, fattore_di_scala_um_per_px):
    """Calcola la larghezza della trincea basandosi sulla linea binarizzata."""
    binarized_line = trench_width_data.get('line_binarizzata')
    
    if binarized_line is None:
        print("Impossibile calcolare la larghezza: eseguire prima il thresholding (tasto 'T' o 'M').")
        return None

    diff = np.diff(binarized_line)
    start_indices = np.where(diff == -1)[0] + 1  # 1 -> 0 (inizio trincea)
    end_indices = np.where(diff == 1)[0] + 1    # 0 -> 1 (fine trincea)

    if not start_indices.size or not end_indices.size:
        trench_width_data['larghezza_trincea_um'] = 'Non_Rilevata'
        print("Trincea non rilevata automaticamente sulla linea di scansione.")
        return 0

    valid_ends = end_indices[end_indices > start_indices[0]]
    if not valid_ends.size:
        trench_width_data['larghezza_trincea_um'] = 'Confine_Finale_Mancante'
        print("Rilevato solo un bordo. Manca il confine finale della trincea.")
        return 0
        
    start_px = start_indices[0]
    end_px = valid_ends[0]
    
    larghezza_px = end_px - start_px
    larghezza_um = larghezza_px * fattore_di_scala_um_per_px
    
    trench_width_data['larghezza_trincea_px'] = f"{larghezza_px}"
    trench_width_data['larghezza_trincea_um'] = f"{larghezza_um:.2f}"
    
    print(f"Larghezza misurata: {larghezza_px} px = {larghezza_um:.2f} um")
    
    return larghezza_um

def get_pixel_intensity(immagine, x, y):
    """
    Ottiene il valore di intensità di un pixel dall'immagine originale normalizzata (8-bit grayscale).
    """
    # Clamp coordinates to ensure they are within bounds
    h, w = immagine.shape[:2]
    x = max(0, min(int(x), w - 1))
    y = max(0, min(int(y), h - 1))

    # L'immagine normalizzata è a canale singolo (grayscale)
    if len(immagine.shape) == 2:
        return immagine[y, x]
    # Se l'immagine è BGR (anche se caricata come grayscale), usa la media
    elif len(immagine.shape) == 3:
        # Se è BGR, è probabile che l'utente stia cliccando sulla display image (BGR)
        # In questo caso, prendiamo la media dei canali
        return np.mean(immagine[y, x])
    # Fallback, dovrebbe essere 8-bit tra 0 e 255
    return 0 

def get_dynamic_line_color(p1, p2, immagine, dark_threshold):
    """Calcola il colore della linea (bianco o rosso) in base alla luminosità media dei punti."""
    x1, y1 = p1
    x2, y2 = p2
    
    intensity_p1 = get_pixel_intensity(immagine, x1, y1)
    intensity_p2 = get_pixel_intensity(immagine, x2, y2)
    avg_intensity = (intensity_p1 + intensity_p2) / 2
    
    # BGR: Bianco (su scuro) o Rosso (su chiaro)
    return (255, 255, 255) if avg_intensity < dark_threshold else (0, 0, 255) 

def draw_extended_line(immagine, m, q, color, thickness=2, text=""):
    """Disegna una retta (y = m*x + q) estesa su tutta la larghezza dell'immagine."""
    h, w = immagine.shape[:2]
    
    # Calcola i punti di inizio e fine retta
    x_min_full = 0
    x_max_full = w - 1
    y_min_full = int(m * x_min_full + q)
    y_max_full = int(m * x_max_full + q)

    # Limita i valori Y ai bordi dell'immagine per un disegno sicuro
    y_min_clamped = np.clip(y_min_full, 0, h - 1)
    y_max_clamped = np.clip(y_max_full, 0, h - 1)
    
    # Disegna la linea
    cv2.line(immagine, (x_min_full, y_min_clamped), (x_max_full, y_max_clamped), color, thickness)
    
    # Aggiungi testo se specificato
    if text:
        # Calcola un punto centrale sulla retta per posizionare il testo (usiamo x=20)
        text_y = int(m * 20 + q)
        text_y = np.clip(text_y, 20, h - 20) # Clamp per stare nei bordi
        # Scegli il colore del testo (generalmente uguale al colore della linea)
        cv2.putText(immagine, text, (10, text_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

def calculate_angle_between_points(p1, p2, p3):
    """
    Calcola l'angolo (in gradi) formato dai segmenti P2-P1 e P2-P3, dove P2 è il vertice.
    """
    p1 = np.array(p1)
    p2 = np.array(p2)
    p3 = np.array(p3)

    # Vettori P2->P1 e P2->P3
    v1 = p1 - p2
    v2 = p3 - p2

    # Prodotto scalare e norma
    dot_product = np.dot(v1, v2)
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)

    if norm_v1 == 0 or norm_v2 == 0:
        return 0.0

    # Coseno dell'angolo
    cosine_angle = dot_product / (norm_v1 * norm_v2)
    
    # Clamp il valore a [-1, 1] per evitare errori di floating point
    cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
    
    # Angolo in radianti e poi in gradi
    angle_rad = np.arccos(cosine_angle)
    angle_deg = np.degrees(angle_rad)
    
    return angle_deg

def leggi_fattore_di_scala_tif(path):
    """
    Legge il fattore di scala (micrometri per pixel) dai metadati del file TIF.

    NOTA: La logica di estrazione del 'pixelwidth' varia in base al software/microscopio. 
    Questa funzione cerca le informazioni standard di risoluzione e fornisce un fallback.
    """
    # Valore di fallback se i metadati non sono disponibili
    default_scale = 1.0 # 1.0 micrometro per pixel
    
    try:
        with tifffile.TiffFile(path) as tif:
            # Tentativo 1: Leggere le risoluzioni standard (Tiff tags 282, 283, 305)
            # XResolution (282) e ResolutionUnit (305) sono spesso usati.
            # ResolutionUnit: 2=Inch, 3=Centimeter
            
            x_res_tag = tif.series[0].keyframe.tags.get(282)
            unit_tag = tif.series[0].keyframe.tags.get(305)

            if x_res_tag and unit_tag:
                x_res = x_res_tag.value
                unit = unit_tag.value
                
                # XResolution è un tuple (numeratore, denominatore) di pixel/unità
                if x_res and x_res[0] != 0:
                    scale_px_per_unit = x_res[0] / x_res[1]
                    scale_unit_per_px = x_res[1] / x_res[0]
                    
                    scale_um_per_px = scale_unit_per_px
                    
                    # 2=Inch (25400 µm), 3=Centimeter (10000 µm)
                    if unit == 3: # Centimetri
                        scale_um_per_px *= 10000
                        unit_str = "Centimetri"
                    elif unit == 2: # Inch
                        scale_um_per_px *= 25400
                        unit_str = "Inch"
                    else:
                        # Se l'unità non è specificata (unit=1) o sconosciuta (ad esempio, 'None' o 'um'),
                        # si assume che 1/Risoluzione sia già in micrometri.
                        unit_str = "Unità non specificata (assumiamo µm)"

                    print(f"Risoluzione TIF letta: {scale_px_per_unit:.2f} pixel/{unit_str}. Scala calcolata: {scale_um_per_px:.4f} µm/px")
                    return scale_um_per_px
            
            # Tentativo 2: Controllare i metadati ImageJ (spesso contengono "um per pixel")
            if tif.imagej_metadata and 'um' in tif.imagej_metadata:
                 print(f"Trovati metadati ImageJ, usando {tif.imagej_metadata['um']} µm/px.")
                 return tif.imagej_metadata['um']
                 
            # Fallback per il fattore di scala non trovato
            print(f"ATTENZIONE: Fattore di scala (pixelwidth) non trovato nei metadati TIF standard. Usando il valore di default: {default_scale} µm/px")
            
    except Exception as e:
        print(f"Attenzione: Errore durante la lettura dei metadati TIF per la scala. Usando il default {default_scale} µm/px. Dettagli: {e}")
        
    return default_scale
# --- Fine Funzioni di Utility Aggiunte/Mancanti ---


def calcola_larghezza_trincea(immagine, fattore_di_scala_um_per_px):
    """
    Misura la larghezza della trincea utilizzando soglia automatica (Otsu)
    su una singola linea di scansione, spostabile con i tasti freccia.
    """
    global punti_selezionati, immagine_originale, immagine_modificabile, window_name, current_mode, scan_line_y, trench_width_data, reference_line_params
    
    # Inizializzazione: immagine_originale qui è l'immagine 8-bit grayscale
    immagine_originale = immagine.copy()
    window_name = "Misura Larghezza Trincea"
    current_mode = 'TRINCEA'
    trench_width_data = {"larghezza_trincea_um": "Non misurata", "threshold_metodo": "N/D", "threshold_valore": "N/D"}

    if reference_line_params is None:
        print("\nATTENZIONE: Retta di riferimento (Gradino) non calcolata. La scansione sarà orizzontale (m=0).")
        m, q = 0.0, immagine_originale.shape[0] // 2 # Centra la Y se non ci sono dati
    else:
        m, q = reference_line_params
        print("\nRetta di riferimento caricata dal Gradino. Usa frecce SU/GIÙ per spostare l'altezza (Q).")

    # Inizializza la linea di scansione Y (usa q come offset iniziale)
    scan_line_q_offset = q # q diventa l'offset che l'utente modifica
    
    # Definisce il passo di spostamento (ad esempio, 2 pixel)
    STEP_SIZE = 2 
    DARK_THRESHOLD = 60

    def redraw_trench_view(m, current_q):
        """Disegna la retta e la linea di scansione sulla base dell'offset attuale."""
        global scan_line_y # <<<< CORREZIONE APPLICATA QUI: da 'nonlocal' a 'global'
        
        # Converte l'immagine grayscale in BGR per il disegno
        immagine_modificabile = cv2.cvtColor(immagine_originale.copy(), cv2.COLOR_GRAY2BGR)
        
        # 1. Calcola il colore dinamico della retta (necessita di due punti per la luminosità, usiamo i confini dell'immagine per la media)
        # Nota: in questa funzione l'immagine_originale è grayscale, ma get_dynamic_line_color è tollerante.
        p_left = (0, int(current_q))
        p_right = (immagine_originale.shape[1] - 1, int(m * (immagine_originale.shape[1] - 1) + current_q))
        
        line_color = get_dynamic_line_color(p_left, p_right, immagine_originale, DARK_THRESHOLD)
        
        # 2. Disegna la retta spostata
        draw_extended_line(immagine_modificabile, m, current_q, line_color, text=f"Linea di Scansione (Q: {current_q:.2f})")
        
        # Per la trincea, la linea di scansione è la retta stessa. Usiamo la Y centrale
        scan_line_y = int(current_q) # Usiamo l'intercetta y come punto di riferimento per l'analisi del profilo

        # 3. Disegna il risultato della soglia (se presente)
        binarized_line = trench_width_data.get('line_binarizzata')
        if binarized_line is not None and scan_line_y is not None:
            # Rileva gli indici di inizio/fine trincea
            diff = np.diff(binarized_line)
            start_indices = np.where(diff == -1)[0] + 1
            end_indices = np.where(diff == 1)[0] + 1
            
            if start_indices.size and end_indices.size and end_indices[0] > start_indices[0]:
                start_px = start_indices[0]
                end_px = end_indices[0]
                
                # Calcola la Y per il disegno (basata sulla X media)
                mid_x = int((start_px + end_px) / 2)
                draw_y = int(m * mid_x + current_q)
                
                # Disegna un segmento di linea sulla retta tra i bordi (Rosso brillante)
                cv2.line(immagine_modificabile, (start_px, draw_y), (end_px, draw_y), (0, 0, 255), 3)
                
                # Aggiunge testo
                larghezza_um = trench_width_data.get('larghezza_trincea_um', 'N/D')
                cv2.putText(immagine_modificabile, f"Larghezza: {larghezza_um} um", (start_px, draw_y - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        cv2.imshow(window_name, immagine_modificabile)
        
    # Disegno iniziale
    redraw_trench_view(m, scan_line_q_offset)

    print("\n--- Misura Larghezza Trincea (T) ---")
    print("Comandi:")
    print("    FRECCE SU/GIÙ: Sposta la linea di scansione (modifica Q).")
    print("    'R': PLOT_PROFILE - Stampa i dettagli del profilo della linea attuale.")
    print("    'T': AUTO_TRENCH_DETECT - Esegue Soglia Otsu sul profilo.")
    print("    'M': MANUAL_THRESHOLD - Imposta soglia manuale (0-255).")
    print("    'W': CALCULATE_WIDTH - Calcola la larghezza dai bordi rilevati.")
    print("    'Q': ESCI/SALTA la misurazione corrente.")
    
    manual_input_mode = False
    temp_threshold_value = ""

    while True:
        key = cv2.waitKey(1) & 0xFF
        
        # Gestione dell'input manuale (per il threshold)
        if manual_input_mode:
            # Codici tasti: 13=INVIO, 8=BACKSPACE, 27=ESC
            if key == 13:  
                if scan_line_y is not None and manual_threshold(get_line_profile(immagine_originale, scan_line_y), trench_width_data, temp_threshold_value):
                    manual_input_mode = False
                elif scan_line_y is None:
                    print("\nErrore: Selezionare prima la linea di scansione (dovrebbe essere già disegnata).")
                    manual_input_mode = False
                temp_threshold_value = ""
                print("\nModalità Soglia Manuale completata. Premi W per calcolare la larghezza.")

            elif key == 8: 
                temp_threshold_value = temp_threshold_value[:-1]
                sys.stdout.write(f"Inserisci Soglia Manuale (0-255) [{temp_threshold_value}]  \r")
                sys.stdout.flush()
            elif 48 <= key <= 57: # Numeri 0-9
                temp_threshold_value += chr(key)
                sys.stdout.write(f"Inserisci Soglia Manuale (0-255) [{temp_threshold_value}]  \r")
                sys.stdout.flush()
            elif key == 27: # ESC per uscire dall'input
                 manual_input_mode = False
                 print("\nModalità Soglia Manuale annullata.")
        
        # Gestione dei comandi da tastiera e frecce (solo se non in modalità input manuale)
        elif not manual_input_mode:
            # Frecce (Gestione Pressione Tasti Speciali)
            if key == 82: # Freccia SU
                scan_line_q_offset -= STEP_SIZE
                scan_line_q_offset = np.clip(scan_line_q_offset, 0, immagine_originale.shape[0] - 1)
                redraw_trench_view(m, scan_line_q_offset)
                print(f"Linea spostata UP: Nuova Q = {scan_line_q_offset:.2f}")

            elif key == 84: # Freccia GIÙ
                scan_line_q_offset += STEP_SIZE
                scan_line_q_offset = np.clip(scan_line_q_offset, 0, immagine_originale.shape[0] - 1)
                redraw_trench_view(m, scan_line_q_offset)
                print(f"Linea spostata DOWN: Nuova Q = {scan_line_q_offset:.2f}")

            elif key == ord('r'):
                if scan_line_y is not None:
                    profile = get_line_profile(immagine_originale, scan_line_y)
                    plot_profile(profile, f"Profilo Y={scan_line_y}")
                else:
                    print("Errore: La linea di scansione non è definita (premi una freccia).")
                
            elif key == ord('t'):
                if scan_line_y is not None:
                    auto_trench_detect(get_line_profile(immagine_originale, scan_line_y), trench_width_data)
                    redraw_trench_view(m, scan_line_q_offset) # Aggiorna la vista con i bordi se rilevati
                else:
                    print("Errore: La linea di scansione non è definita (premi una freccia).")
                    
            elif key == ord('m'):
                manual_input_mode = True
                temp_threshold_value = ""
                sys.stdout.write("Modalità: Inserisci Soglia Manuale (0-255) []. Premi INVIO per confermare. \r")
                sys.stdout.flush()
                
            elif key == ord('w'):
                larghezza_um = calculate_trench_width(trench_width_data, fattore_di_scala_um_per_px)
                if larghezza_um is not None and larghezza_um != 0 and larghezza_um != 'Confine_Finale_Mancante':
                    # Ridisegna per mostrare il risultato finale in rosso permanente
                    redraw_trench_view(m, scan_line_q_offset)
                    cv2.waitKey(0)  
                    break
                
            elif key == ord('q'):
                break
                
            elif key != 255:
                pass


    cv2.destroyAllWindows()
    current_mode = None
    
    if 'line_binarizzata' in trench_width_data:
        del trench_width_data['line_binarizzata']
        
    if trench_width_data.get('larghezza_trincea_um') == 'Non misurata':
        trench_width_data = {"larghezza_trincea_um": "Non misurata"}
    
    return trench_width_data

# --- Callback per il Mouse Generalizzato ---

def on_mouse(event, x, y, flags, param):
    """
    Funzione di callback per catturare gli eventi del mouse e disegnare.
    """
    global punti_selezionati, immagine_modificabile, window_name, current_mode, scan_line_y, immagine_originale, reference_line_params

    # Costante per il contrasto dinamico
    DARK_THRESHOLD = 60 # Soglia di intensità (0-255) sotto la quale si considera scuro

    if event == cv2.EVENT_LBUTTONDOWN:
        
        # 1. Modalità Gradino/Angolo (3 punti necessari)
        if current_mode == 'GRADINO' or current_mode == 'ANGOLO':
            if len(punti_selezionati) < 3:
                punti_selezionati.append((x, y))
                print(f"Punto selezionato: ({x}, {y}) - Clic {len(punti_selezionati)} di 3")
                
                # Resetta l'immagine e ridisegna tutti i punti
                immagine_modificabile = immagine_originale.copy()
                
                # Disegna i punti 
                for px, py in punti_selezionati:
                    intensity = get_pixel_intensity(immagine_originale, px, py)
                    # Colore del punto: Verde 
                    # Usa sempre contrasto dinamico per i punti
                    point_color = (255, 255, 255) if intensity < DARK_THRESHOLD else (0, 0, 0)
                    cv2.circle(immagine_modificabile, (px, py), 5, point_color, -1) 
                    
                # *** LOGICA: DISEGNA LA RETTA DOPO IL SECONDO PUNTO CON CONTRASTO (Solo per Gradino) ***
                if len(punti_selezionati) == 2 and current_mode == 'GRADINO':
                    p1, p2 = punti_selezionati[0], punti_selezionati[1]
                    x1, y1 = p1
                    x2, y2 = p2
                    
                    # 1. Determina il colore della linea in base alla luminosità media dei punti P1 e P2
                    line_color = get_dynamic_line_color(p1, p2, immagine_originale, DARK_THRESHOLD)
                    
                    if x2 != x1:
                        m = (y2 - y1) / (x2 - x1)
                        q = y1 - m * x1
                        draw_extended_line(immagine_modificabile, m, q, line_color, text=f"Retta Riferimento (P1-P2)")
                    else:
                        # Linea verticale
                        y_avg = int((y1 + y2) / 2)
                        cv2.line(immagine_modificabile, (x1, 0), (x1, immagine_modificabile.shape[0] - 1), line_color, 2)
                        cv2.putText(immagine_modificabile, f"Retta Riferimento Verticale", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, line_color, 1)

                # *** LOGICA: DISEGNA I SEGMENTI (Solo per Angolo) ***
                if len(punti_selezionati) >= 2 and current_mode == 'ANGOLO':
                    cv2.line(immagine_modificabile, punti_selezionati[0], punti_selezionati[1], (255, 0, 0), 2)
                if len(punti_selezionati) == 3 and current_mode == 'ANGOLO':
                    cv2.line(immagine_modificabile, punti_selezionati[1], punti_selezionati[2], (255, 0, 0), 2)


                cv2.imshow(window_name, immagine_modificabile)
            else:
                print("Hai già selezionato 3 punti. Premi 'c' per calcolare o 'z' per annullare.")

        # 2. Modalità Selezione Linea Trincea Orizzontale (1 punto necessario) -> Tasto 'L'
        elif current_mode == 'TRINCEA_LINE':
            scan_line_y = y
            print(f"Linea di scansione Y selezionata: {y} (Orizzontale). Premi 'T' per il calcolo Otsu.")
            
            # Disegna la linea di scansione BIANCA sull'immagine modificabile
            immagine_modificabile = immagine_originale.copy()
            cv2.line(immagine_modificabile, (0, y), (immagine_modificabile.shape[1], y), (255, 255, 255), 2)
            cv2.putText(immagine_modificabile, f"Y: {y} (Orizzontale)", (10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            cv2.imshow(window_name, immagine_modificabile)
            current_mode = 'TRINCEA' # Torna alla modalità principale Trincea
            
        # 3. Modalità Selezione Piano Orizzontale (1 punto per definire la Y) -> Tasto 'P' (Non usata in questo script, ma lasciata per completezza)
        elif current_mode == 'TRINCEA_PLANE':
            if len(punti_selezionati) == 0:
                punti_selezionati.append((x, y))
                
                # Colore del punto contrastivo
                intensity = get_pixel_intensity(immagine_originale, x, y)
                point_color = (255, 255, 255) if intensity < DARK_THRESHOLD else (0, 0, 0)
                
                cv2.circle(immagine_modificabile, (x, y), 5, point_color, -1) # Punto con contrasto
                cv2.imshow(window_name, immagine_modificabile)

                # --- DISEGNA LA LINEA DI RIFERIMENTO (GIALLA per Piano) ---
                y_ref = punti_selezionati[0][1]
                line_color = (0, 255, 255) # Giallo fisso per il piano di riferimento

                # Resetta l'immagine modificabile e disegna la linea gialla
                immagine_modificabile = immagine_originale.copy() 
                cv2.line(immagine_modificabile, (0, y_ref), (immagine_modificabile.shape[1], y_ref), line_color, 2) 
                cv2.putText(immagine_modificabile, "Piano Riferimento", (10, y_ref + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, line_color, 1)
                
                cv2.imshow(window_name, immagine_modificabile)
                
                print(f"Piano orizzontale di riferimento disegnato a Y={y_ref}. Premi 'P' per ri-selezionare.")
                punti_selezionati = [] # Resetta i punti per permettere una nuova selezione
                current_mode = 'TRINCEA' # Torna alla modalità principale Trincea
                
        # 4. Modalità Selezione Linea Trincea X-Coordinate (1 punto necessario) -> Tasto 'X'
        elif current_mode == 'TRINCEA_X_SELECT':
            if reference_line_params is None:
                print("Errore: Definire prima la retta di riferimento nel calcolo del gradino ('G'/'c').")
                current_mode = 'TRINCEA'
                return
            
            m, q = reference_line_params
            x_scan = x
            
            # Calcola la Y sulla retta definita
            scan_line_y_calculated = int(m * x_scan + q)
            scan_line_y = scan_line_y_calculated # Set the global horizontal scan line at this specific Y
            
            immagine_modificabile = immagine_originale.copy()
            
            # Disegna la retta completa come riferimento (Verde, fissa)
            x_min_full = 0
            x_max_full = immagine_modificabile.shape[1] - 1
            y_min_full = int(m * x_min_full + q)
            y_max_full = int(m * x_max_full + q)
            
            y_min_clamped = np.clip(y_min_full, 0, immagine_modificabile.shape[0] - 1)
            y_max_clamped = np.clip(y_max_full, 0, immagine_modificabile.shape[0] - 1)
            
            cv2.line(immagine_modificabile, (x_min_full, y_min_clamped), (x_max_full, y_max_clamped), (0, 255, 0), 2)
            
            # Disegna la linea di scansione BIANCA (orizzontale) che passa per il punto X selezionato sulla retta
            cv2.line(immagine_modificabile, (0, scan_line_y_calculated), (immagine_modificabile.shape[1], scan_line_y_calculated), (255, 255, 255), 2)
            
            # Punto selezionato sulla retta (Contrasto Dinamico)
            intensity = get_pixel_intensity(immagine_originale, x_scan, scan_line_y_calculated)
            point_color = (255, 255, 255) if intensity < DARK_THRESHOLD else (0, 0, 0)
            
            cv2.circle(immagine_modificabile, (x_scan, scan_line_y_calculated), 5, point_color, -1) 
            
            cv2.putText(immagine_modificabile, f"Y Scan: {scan_line_y_calculated} (da Retta)", (10, scan_line_y_calculated - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            cv2.imshow(window_name, immagine_modificabile)
            current_mode = 'TRINCEA'
            print(f"Linea di scansione Y selezionata a Y={scan_line_y_calculated} (calcolata dalla retta alla X={x_scan}). Premi 'T' per il calcolo Otsu.")
            
            
# --- Funzioni di Misura Esistenti ---

def calcola_gradino_di_scavo(immagine, fattore_di_scala_um_per_px):
    global punti_selezionati, immagine_originale, immagine_modificabile, window_name, current_mode, reference_line_params
    punti_selezionati = []
    immagine_originale = immagine.copy()
    immagine_modificabile = immagine.copy()
    window_name = "Misura Gradino"
    current_mode = 'GRADINO'
    reference_line_params = None # Reset dei parametri della retta all'inizio della nuova misura
    DARK_THRESHOLD = 60 # Soglia per il contrasto dinamico
    
    cv2.imshow(window_name, immagine_modificabile)
    cv2.setMouseCallback(window_name, on_mouse)

    print("\n--- Misura del Gradino ---")
    print("MODALITA' GRADINO: Seleziona 2 punti (P1, P2) su superficie, 1 punto (P3) su fondo.")
    print("Dopo il 2° punto, vedrai la **retta con contrasto dinamico** (Bianca su scuro, Rossa su chiaro).")
    print("Comandi: 'c' per calcolare, 'z' per annullare, 'q' per saltare l'immagine corrente.")

    risultato = {"gradino_di_scavo_um": "Non misurato"}
    while True:
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('c'):
            if len(punti_selezionati) >= 3:
                p_riferimento1, p_riferimento2, p_scavo = punti_selezionati[0], punti_selezionati[1], punti_selezionati[2]
                x1, y1 = p_riferimento1
                x2, y2 = p_riferimento2
                x3, y3 = p_scavo
                
                # 1. Calcolo dell'equazione della retta (y = m*x + q)
                if x2 == x1:
                    # Linea verticale: Usiamo la Y media
                    m = 0.0
                    q = (y1 + y2) / 2.0
                    print("Attenzione: Punti di riferimento verticali (x1=x2). La retta è considerata orizzontale (Y media).")
                else:
                    m = (y2 - y1) / (x2 - x1)
                    q = y1 - m * x1
                
                reference_line_params = (m, q) # SALVA LA RETTA
                
                # 2. Calcola la Y di riferimento sulla retta alla X del punto di scavo (x3)
                y_ref_at_x3 = m * x3 + q
                
                # 3. Calcolo del gradino: distanza verticale P3 -> Refference Line
                distanza_verticale_px = abs(y_ref_at_x3 - y3)
                gradino_um = distanza_verticale_px * fattore_di_scala_um_per_px
                
                risultato["gradino_di_scavo_um"] = f"{gradino_um:.2f}"
                risultato["retta_riferimento_m"] = f"{m:.4f}"
                risultato["retta_riferimento_q"] = f"{q:.2f}"
                
                print(f"Retta di riferimento calcolata e salvata: Y = {m:.4f} * X + {q:.2f}")
                
                # --- Disegno finale ---
                
                # Resetta l'immagine per il disegno finale
                immagine_modificabile = immagine_originale.copy()
                
                # Determina il colore finale della retta (verde per finalità)
                line_final_color = (0, 255, 0) # Verde fisso per il risultato finale
                
                # Disegna una linea estesa per mostrare l'inclinazione (Verde per finalità)
                x_min_full = 0
                x_max_full = immagine_modificabile.shape[1] - 1
                y_min_full = int(m * x_min_full + q)
                y_max_full = int(m * x_max_full + q)

                # Clamp Y values to image boundaries for safe drawing
                y_min_clamped = np.clip(y_min_full, 0, immagine_modificabile.shape[0] - 1)
                y_max_clamped = np.clip(y_max_full, 0, immagine_modificabile.shape[0] - 1)
                
                cv2.line(immagine_modificabile, (x_min_full, y_min_clamped), (x_max_full, y_max_clamped), line_final_color, 2)
                
                # Disegno del punto di scavo e della linea verticale di misura (Magenta)
                # Punto 3 (scavo)
                intensity_p3 = get_pixel_intensity(immagine_originale, x3, y3)
                point_color_p3 = (255, 255, 255) if intensity_p3 < DARK_THRESHOLD else (0, 0, 0)
                cv2.circle(immagine_modificabile, p_scavo, 5, point_color_p3, -1) 
                
                cv2.line(immagine_modificabile, (x3, y3), (x3, int(y_ref_at_x3)), (255, 0, 255), 1) # Linea verticale di misura Magenta
                
                # Disegna Punti 1 e 2
                intensity_p1 = get_pixel_intensity(immagine_originale, x1, y1)
                point_color_p1 = (255, 255, 255) if intensity_p1 < DARK_THRESHOLD else (0, 0, 0)
                intensity_p2 = get_pixel_intensity(immagine_originale, x2, y2)
                point_color_p2 = (255, 255, 255) if intensity_p2 < DARK_THRESHOLD else (0, 0, 0)
                
                cv2.circle(immagine_modificabile, p_riferimento1, 5, point_color_p1, -1) 
                cv2.circle(immagine_modificabile, p_riferimento2, 5, point_color_p2, -1)

                cv2.putText(immagine_modificabile, f"Gradino: {gradino_um:.2f} um", (x3 + 10, y3), cv2.FONT_HERSHEY_SIMPLEX, 0.7, line_final_color, 2)
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
                
                # Ricarica l'immagine originale
                immagine_modificabile = immagine_originale.copy()
                
                # Ridisegna i punti rimanenti con contrasto dinamico
                for i, p in enumerate(punti_selezionati):
                    px, py = p
                    intensity = get_pixel_intensity(immagine_originale, px, py)
                    point_color = (255, 255, 255) if intensity < DARK_THRESHOLD else (0, 0, 0)
                    cv2.circle(immagine_modificabile, p, 5, point_color, -1)
                
                # Ricalcola e ridisegna la retta se ci sono ancora 2 punti (con contrasto dinamico)
                if len(punti_selezionati) == 2:
                    p1, p2 = punti_selezionati[0], punti_selezionati[1]
                    x1, y1 = p1
                    x2, y2 = p2
                    
                    line_color = get_dynamic_line_color(p1, p2, immagine_originale, DARK_THRESHOLD)
                    
                    if x2 != x1:
                        m = (y2 - y1) / (x2 - x1)
                        q = y1 - m * x1
                        draw_extended_line(immagine_modificabile, m, q, line_color, text=f"Retta Riferimento (P1-P2)")
                    else:
                        y_avg = int((y1 + y2) / 2)
                        cv2.line(immagine_modificabile, (x1, 0), (x1, immagine_modificabile.shape[0] - 1), line_color, 2)
                        cv2.putText(immagine_modificabile, f"Retta Riferimento Verticale", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, line_color, 1)

                cv2.imshow(window_name, immagine_modificabile)
            else:
                # Se non ci sono punti, ricarica solo l'immagine originale
                immagine_modificabile = immagine_originale.copy()
                cv2.imshow(window_name, immagine_modificabile)

        elif key == ord('q'):
            print("Misurazione saltata.")
            break
            
    cv2.destroyAllWindows()
    current_mode = None
    punti_selezionati = []
    
    return risultato
# --- FINE calcola_gradino_di_scavo (parte mancante completata) ---

def calcola_angolo_apertura(immagine, fattore_di_scala_um_per_px):
    """
    Misura l'angolo di apertura tra tre punti selezionati (P1, P2, P3),
    dove P2 è il vertice dell'angolo.
    """
    global punti_selezionati, immagine_originale, immagine_modificabile, window_name, current_mode
    punti_selezionati = []
    immagine_originale = immagine.copy()
    immagine_modificabile = immagine.copy()
    window_name = "Misura Angolo di Apertura"
    current_mode = 'ANGOLO'
    DARK_THRESHOLD = 60
    
    cv2.imshow(window_name, immagine_modificabile)
    cv2.setMouseCallback(window_name, on_mouse)

    print("\n--- Misura Angolo di Apertura ---")
    print("MODALITA' ANGOLO: Seleziona 3 punti (P1, P2, P3) dove P2 è il vertice dell'angolo.")
    print("Comandi: 'c' per calcolare, 'z' per annullare, 'q' per saltare l'immagine corrente.")

    risultato = {"angolo_di_apertura_gradi": "Non misurato"}
    while True:
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('c'):
            if len(punti_selezionati) >= 3:
                p1, p2, p3 = punti_selezionati[0], punti_selezionati[1], punti_selezionati[2]
                
                # Calcola l'angolo (P2 è il vertice)
                angolo_deg = calculate_angle_between_points(p1, p2, p3)
                
                risultato["angolo_di_apertura_gradi"] = f"{angolo_deg:.2f}"
                
                print(f"Angolo misurato: {angolo_deg:.2f} gradi")
                
                # --- Disegno finale ---
                immagine_modificabile = immagine_originale.copy()
                line_color = (255, 0, 0) # Blu fisso per il risultato finale
                
                # Disegna i segmenti P1-P2 e P2-P3
                cv2.line(immagine_modificabile, p1, p2, line_color, 2)
                cv2.line(immagine_modificabile, p2, p3, line_color, 2)
                
                # Disegna i punti
                for p in punti_selezionati:
                    intensity = get_pixel_intensity(immagine_originale, p[0], p[1])
                    point_color = (255, 255, 255) if intensity < DARK_THRESHOLD else (0, 0, 0)
                    cv2.circle(immagine_modificabile, p, 5, point_color, -1) 
                
                # Posiziona il testo vicino al vertice (P2)
                cv2.putText(immagine_modificabile, f"Angolo: {angolo_deg:.2f} deg", (p2[0] + 10, p2[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.7, line_color, 2)
                cv2.imshow(window_name, immagine_modificabile)
                
                cv2.waitKey(0)
                break
            else:
                print("Devi selezionare 3 punti prima di calcolare.")
        
        elif key == ord('z'):
            if len(punti_selezionati) > 0:
                punti_selezionati.pop()
                print(f"Ultimo punto annullato. Punti rimanenti: {len(punti_selezionati)}")
                
                # Ricarica l'immagine originale
                immagine_modificabile = immagine_originale.copy()
                
                # Ridisegna i punti rimanenti con contrasto dinamico
                for p in punti_selezionati:
                    px, py = p
                    intensity = get_pixel_intensity(immagine_originale, px, py)
                    point_color = (255, 255, 255) if intensity < DARK_THRESHOLD else (0, 0, 0)
                    cv2.circle(immagine_modificabile, p, 5, point_color, -1)
                
                # Ridisegna il segmento P1-P2 se ci sono 2 o 3 punti
                if len(punti_selezionati) >= 2:
                    cv2.line(immagine_modificabile, punti_selezionati[0], punti_selezionati[1], (255, 0, 0), 2)
                # Ridisegna il segmento P2-P3 se ci sono 3 punti
                if len(punti_selezionati) == 3:
                    cv2.line(immagine_modificabile, punti_selezionati[1], punti_selezionati[2], (255, 0, 0), 2)

                cv2.imshow(window_name, immagine_modificabile)
            else:
                immagine_modificabile = immagine_originale.copy()
                cv2.imshow(window_name, immagine_modificabile)
        
        elif key == ord('q'):
            print("Misurazione saltata.")
            break
            
    cv2.destroyAllWindows()
    current_mode = None
    punti_selezionati = []
    
    return risultato

# --- Logica di Esecuzione Principale ---

def carica_immagine_tif(path):
    """
    Carica un'immagine .tif, la normalizza a 8-bit e la converte in BGR per OpenCV.
    Restituisce l'immagine normalizzata 8-bit (grayscale) e l'immagine BGR (per il disegno).
    """
    try:
        # Carica il file TIFF utilizzando tifffile
        img_data = tifffile.imread(path)
        
        if img_data is None:
            print(f"Errore: Impossibile caricare l'immagine da {path}")
            return None, None
            
        # Assicurati che l'immagine sia 2D (grayscale)
        if len(img_data.shape) > 2:
            # Se è a colori o ha più canali, prendi il primo o converti in grayscale
            # E' improbabile che tifffile dia BGR ma è un controllo di sicurezza
            if img_data.shape[-1] > 1 and img_data.ndim == 3:
                # Tentativo di conversione in scala di grigi se ha 3 canali
                img_data = cv2.cvtColor(img_data, cv2.COLOR_BGR2GRAY)
            else:
                # Prende il primo slice se ha più dimensioni (ad es. Z-stack)
                img_data = img_data[..., 0] 
        
        # Normalizzazione a 8-bit (0-255)
        # Troviamo min/max per scalare correttamente
        min_val = np.min(img_data)
        max_val = np.max(img_data)
        
        if max_val == min_val:
             # Se l'immagine è piatta, impostala a 0
            normalized_image = np.zeros_like(img_data, dtype=np.uint8)
        else:
            # Scalatura lineare a 0-255 (uint8)
            normalized_image = (img_data - min_val) / (max_val - min_val) * 255
            normalized_image = normalized_image.astype(np.uint8)
        
        # L'immagine normalizzata (normalized_image) è in grayscale (8-bit) per i calcoli.
        
        # Converte l'immagine normalizzata in BGR per visualizzazione/disegno a colori (immagine_display).
        display_image_bgr = cv2.cvtColor(normalized_image, cv2.COLOR_GRAY2BGR)
        
        print(f"Immagine caricata e normalizzata a 8-bit: {path}")
        return normalized_image, display_image_bgr 
        
    except Exception as e:
        print(f"Errore durante il caricamento o la normalizzazione di {path}: {e}")
        return None, None

def main():
    """
    Funzione principale per gestire il caricamento dei file e il ciclo di misurazione.
    Ora accetta la cartella radice e cerca ricorsivamente i file TIF.
    """
    global punti_selezionati, reference_line_params

    # MODIFICA: Ora ci aspettiamo UN solo argomento: la cartella radice da scansionare
    if len(sys.argv) != 2:
        print("Uso: python dete.py <percorso_cartella_principale>")
        print("Il programma cercherà ricorsivamente tutti i file .tif/.tiff al suo interno.")
        sys.exit(1)

    root_directory = sys.argv[1]

    if not os.path.isdir(root_directory):
        print(f"Errore: Il percorso '{root_directory}' non è una directory valida o non esiste.")
        sys.exit(1)

    # 1. Trova ricorsivamente tutti i file TIF
    file_paths = []
    print(f"Scansione ricorsiva di: {root_directory}...")

    for dirpath, dirnames, filenames in os.walk(root_directory):
        for f in filenames:
            # Controlla estensioni in minuscolo
            if f.lower().endswith(('.tif', '.tiff')):
                full_path = os.path.join(dirpath, f)
                file_paths.append(full_path)
    
    # 2. Verifica se sono stati trovati file
    if not file_paths:
        print(f"Nessun file .tif/.tiff trovato in '{root_directory}' e nelle sue sottocartelle.")
        sys.exit(0)

    print(f"Trovati {len(file_paths)} file immagine da elaborare.")

    risultati_totali = []

    # 3. Ciclo di elaborazione per tutti i file trovati
    for file_path in file_paths:
        file_name = os.path.basename(file_path)
        print(f"\n==============================================")
        print(f"Inizio elaborazione file: {file_name} (in: {os.path.dirname(file_path)})")
        print(f"==============================================")
        
        # 0. LEGGI IL FATTORE DI SCALA (pixelwidth) DAI METADATI DEL FILE
        fattore_di_scala_um_per_px = leggi_fattore_di_scala_tif(file_path)

        # 1. Carica l'immagine: grayscale per i calcoli (Trench), BGR per il display (Gradino/Angolo)
        immagine_grayscale, immagine_display = carica_immagine_tif(file_path)

        if immagine_grayscale is None:
            continue

        # Dizionario per i risultati specifici del file
        risultati_file = {'file_name': file_name}
        
        # Resetta i parametri della retta e i punti per ogni nuova immagine
        reference_line_params = None 
        punti_selezionati = []
        
        # --- Ciclo di Misurazione ---
        print("\n--- Scelta della Misura ---")
        print(f"Fattore di scala (pixelwidth) utilizzato: {fattore_di_scala_um_per_px:.4f} µm/px")
        print("Seleziona il tipo di misura da eseguire:")
        print("  'G': Gradino di Scavo (Definisce anche la retta di riferimento)")
        print("  'A': Angolo di Apertura")
        print("  'T': Larghezza Trincea (Richiede Gradino/Retta di riferimento se inclinata)")
        print("  'S': Salta l'immagine corrente e passa alla successiva")

        scelta_valida = False
        while not scelta_valida:
            try:
                # Modifica per input standard in terminale
                scelta = input("Scelta [G/A/T/S]: ").upper()
            except EOFError: 
                scelta = 'S'
            
            if scelta == 'G':
                # Passa l'immagine BGR (per il disegno)
                risultato_gradino = calcola_gradino_di_scavo(immagine_display.copy(), fattore_di_scala_um_per_px)
                risultati_file.update(risultato_gradino)
                scelta_valida = True
            elif scelta == 'A':
                # Passa l'immagine BGR (per il disegno)
                risultato_angolo = calcola_angolo_apertura(immagine_display.copy(), fattore_di_scala_um_per_px)
                risultati_file.update(risultato_angolo)
                scelta_valida = True
            elif scelta == 'T':
                # Passa l'immagine grayscale 8-bit (per il calcolo della soglia)
                risultato_trincea = calcola_larghezza_trincea(immagine_grayscale.copy(), fattore_di_scala_um_per_px)
                risultati_file.update(risultato_trincea)
                scelta_valida = True
            elif scelta == 'S':
                print("Immagine saltata.")
                scelta_valida = True
            else:
                print("Scelta non valida. Riprova.")
        
        if len(risultati_file) > 1: # Se è stato misurato qualcosa
            risultati_totali.append(risultati_file)

    # 4. Salva i risultati in XML alla fine di tutte le immagini
    if risultati_totali:
        output_file = "risultati_misure_totali.xml"
        salva_risultati_xml(risultati_totali, output_file)
        print(f"\n!!! Elaborazione completata. Risultati salvati in {output_file} !!!")
    else:
        print("\nNessuna misurazione completata.")

if __name__ == '__main__':
    # Punto di ingresso dello script
    main()
