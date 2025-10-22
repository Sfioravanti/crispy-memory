import tifffile
import cv2
import os
import numpy as np
import sys
from lxml import etree # Necessario per la gestione dell'XML

# Costanti per la gestione dei tasti (specifiche per OpenCV)
KEY_UP = 82 
KEY_DOWN = 84 
DARK_THRESHOLD = 125 # Soglia di intensità (0-255) per il disegno a contrasto

# --- Variabili Globali ---
punti_selezionati = []
immagine_originale = None # Immagine 8-bit normalizzata (grayscale)
immagine_modificabile = None # Immagine BGR per il disegno (per la visualizzazione)
window_name = ""
current_mode = None # 'GRADINO', 'ANGOLO', 'TRINCEA', 'TRINCEA_LINE', 'TRINCEA_X_SELECT'

# Parametri della retta di riferimento (m, q) calcolati dai primi 2 punti del gradino.
reference_line_params = None

scan_line_y = None # Coordinata Y della linea di scansione/riferimento (per Trincea)
trench_width_data = {} # Dati temporanei per la misurazione della trincea

# --- Funzioni di Utility ---

def salva_risultati_xml(dati, nome_file):
    """
    Salva una lista di dizionari (risultati di misurazione) in un unico file XML.
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
        print(f"Risultati salvati con successo in {nome_file}")
    except Exception as e:
        print(f"Errore durante il salvataggio XML: {e}")

def get_line_profile(immagine_norm, y_coord):
    """Estrae il profilo dei livelli di grigio lungo la coordinata Y."""
    # Clamping for safety
    y_coord = np.clip(y_coord, 0, immagine_norm.shape[0] - 1)
    if y_coord is not None:
        return immagine_norm[y_coord, :]
    return None

def plot_profile(profile_data, title="Profilo Intensità"):
    """
    Simula la visualizzazione di un profilo (poiché non possiamo usare matplotlib).
    Stampa i dati chiave del profilo (min, max, media) in console.
    """
    if profile_data is not None and profile_data.size > 0:
        print(f"\n--- {title} ---")
        print(f"Punti nel profilo: {profile_data.size}")
        print(f"Min Intensità: {np.min(profile_data):.2f}")
        print(f"Max Intensità: {np.max(profile_data):.2f}")
        print(f"Media Intensità: {np.mean(profile_data):.2f}")
    else:
        print("Profilo non disponibile. Selezionare prima una linea di scansione.")

def auto_trench_detect(profile_data, trench_width_data):
    """
    Applica l'algoritmo di Otsu's per la soglia automatica sulla linea selezionata.
    """
    if profile_data is None:
        print("Impossibile eseguire la soglia: selezionare prima una linea di scansione.")
        return False
    
    try:
        # Assicuriamo che i dati siano di tipo adatto per cv2.threshold (np.uint8)
        profile_data_u8 = profile_data.astype(np.uint8)
        
        # Otsu's thresholding
        threshold_value, binarized_line_u8 = cv2.threshold(profile_data_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Converti il risultato binarizzato in 0/1 per l'analisi logica
        binarized_line = np.where(binarized_line_u8 == 255, 1, 0)
        
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
        print("Impossibile eseguire la soglia: selezionare prima una linea di scansione.")
        return False
        
    try:
        T = int(manual_t)
        if not (0 <= T <= 255):
            print("Valore soglia non valido. Deve essere tra 0 e 255 (8-bit).")
            return False

        profile_data_u8 = profile_data.astype(np.uint8)
        _, binarized_line_u8 = cv2.threshold(profile_data_u8, T, 255, cv2.THRESH_BINARY)
        
        # Converti il risultato binarizzato in 0/1 per l'analisi logica
        binarized_line = np.where(binarized_line_u8 == 255, 1, 0)
        
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

    # Vogliamo 1 -> 0 (inizio trincea - bordo sinistro) e 0 -> 1 (fine trincea - bordo destro)
    diff = np.diff(binarized_line)
    start_indices = np.where(diff == -1)[0] + 1  # 1 (esterno) -> 0 (trincea)
    end_indices = np.where(diff == 1)[0] + 1     # 0 (trincea) -> 1 (esterno)

    if not start_indices.size or not end_indices.size:
        trench_width_data['larghezza_trincea_um'] = 'Non_Rilevata'
        print("Trincea non rilevata automaticamente sulla linea di scansione (mancano entrambi i bordi).")
        return 0

    # Troviamo il primo bordo destro che segue il primo bordo sinistro
    valid_ends = end_indices[end_indices > start_indices[0]]
    if not valid_ends.size:
        trench_width_data['larghezza_trincea_um'] = 'Confine_Finale_Mancante'
        print("Rilevato solo un bordo (sinistro). Manca il confine finale della trincea (destro).")
        return 0
        
    start_px = start_indices[0]
    end_px = valid_ends[0]
    
    larghezza_px = end_px - start_px
    larghezza_um = larghezza_px * fattore_di_scala_um_per_px
    
    # Memorizza i risultati con alta precisione (8 cifre decimali)
    trench_width_data['larghezza_trincea_px'] = f"{larghezza_px}"
    trench_width_data['larghezza_trincea_um'] = f"{larghezza_um:.8f}" 
    
    print(f"Larghezza misurata: {larghezza_px} px = {larghezza_um:.8f} um") 
    
    return larghezza_um

def get_pixel_intensity(immagine, x, y):
    """
    Ottiene il valore di intensità di un pixel dall'immagine originale normalizzata (8-bit grayscale).
    """
    h, w = immagine.shape[:2]
    x = max(0, min(int(x), w - 1))
    y = max(0, min(int(y), h - 1))

    # L'immagine normalizzata è a canale singolo (grayscale)
    if len(immagine.shape) == 2:
        return immagine[y, x]
    # Se l'immagine è BGR (immagine modificabile) usiamo la media
    elif len(immagine.shape) == 3:
        return np.mean(immagine[y, x])
    return 0 

def get_dynamic_line_color(p1, p2, immagine, dark_threshold):
    """Calcola il colore della linea (bianco o rosso) in base alla luminosità media dei punti P1 e P2."""
    x1, y1 = p1
    x2, y2 = p2
    
    # Usiamo l'intensità in 8-bit grayscale
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

    v1 = p1 - p2
    v2 = p3 - p2

    dot_product = np.dot(v1, v2)
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)

    if norm_v1 == 0 or norm_v2 == 0:
        return 0.0

    cosine_angle = dot_product / (norm_v1 * norm_v2)
    cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
    
    angle_rad = np.arccos(cosine_angle)
    angle_deg = np.degrees(angle_rad)
    
    return angle_deg

def leggi_fattore_di_scala_tif(path):
    """
    Legge il fattore di scala (micrometri per pixel) dai metadati del file TIF.

    Priorità di ricerca: FEI, OME, ImageJ, Tiff Tags standard.
    """
    default_scale = 1.0 # µm/px
    
    try:
        with tifffile.TiffFile(path) as tif:
            
            # --- TENTATIVO 1: METADATI PROPRIETARI FEI (Scan -> PixelWidth) ---
            pixel_width_source = None
            pixel_width_value = None

            if tif.fei_metadata and 'Scan' in tif.fei_metadata and 'PixelWidth' in tif.fei_metadata['Scan']:
                pixel_width_value = tif.fei_metadata['Scan']['PixelWidth']
                pixel_width_source = "Scan -> PixelWidth"
            elif tif.fei_metadata and 'PixelWidth' in tif.fei_metadata:
                 pixel_width_value = tif.fei_metadata['PixelWidth']
                 pixel_width_source = "PixelWidth (root)"

            if pixel_width_value is not None:
                try:
                    pixel_width_m = float(str(pixel_width_value).strip())
                    # Conversione da metri a micron (1 metro = 10^6 micron)
                    fattore_di_scala_um_per_px = pixel_width_m * 1e6
                    
                    print(f"Risoluzione FEI '{pixel_width_source}' trovata: {fattore_di_scala_um_per_px:.8f} µm/px (convertito da metri).")
                    return fattore_di_scala_um_per_px
                except (ValueError, TypeError, KeyError):
                    print(f"Attenzione: '{pixel_width_source}' non è un valore numerico valido. Proseguo la ricerca.")
                    pass

            # --- TENTATIVO 2: METADATI OME (PhysicalSizeX) ---
            if tif.ome_metadata:
                try:
                    physical_size_x = tif.ome_metadata['Image'][0]['Pixels']['PhysicalSizeX']
                    physical_unit_x = tif.ome_metadata['Image'][0]['Pixels']['PhysicalSizeXUnit']
                    
                    if physical_unit_x in ('µm', 'um'):
                        print(f"Risoluzione OME trovata: {physical_size_x} µm/px.")
                        return float(physical_size_x)
                    elif physical_unit_x == 'nm':
                        scale_um_per_px = float(physical_size_x) / 1000.0
                        print(f"Risoluzione OME trovata: {physical_size_x} nm/px. Scalata a {scale_um_per_px:.8f} µm/px.")
                        return scale_um_per_px
                        
                except (KeyError, IndexError, ValueError):
                    pass
            
            # --- TENTATIVO 3: METADATI IMAGEJ ---
            if tif.imagej_metadata and 'um' in tif.imagej_metadata:
                    scale = float(tif.imagej_metadata['um'])
                    print(f"Trovati metadati ImageJ, usando {scale:.8f} µm/px.")
                    return scale
                    
            # --- TENTATIVO 4: STANDARD Tiff Tags (XResolution) ---
            x_res_tag = tif.series[0].keyframe.tags.get(282)
            unit_tag = tif.series[0].keyframe.tags.get(305)

            if x_res_tag and unit_tag:
                x_res = x_res_tag.value
                unit = unit_tag.value
                
                if x_res and x_res[0] != 0:
                    scale_unit_per_px = x_res[1] / x_res[0]
                    scale_um_per_px = scale_unit_per_px
                    
                    if unit == 3: # Centimetri
                        scale_um_per_px *= 10000
                        unit_str = "Centimetri"
                    elif unit == 2: # Inch
                        scale_um_per_px *= 25400
                        unit_str = "Inch"
                    else:
                        unit_str = "Unità non specificata (assumiamo µm)"

                    print(f"Risoluzione TIF standard letta. Scala calcolata: {scale_um_per_px:.8f} µm/px")
                    return scale_um_per_px
            
            # --- FALLBACK ---
            print(f"ATTENZIONE: Fattore di scala non trovato nei metadati. Usando il valore di default: {default_scale:.8f} µm/px")
            
    except Exception as e:
        print(f"Attenzione: Errore durante la lettura dei metadati TIF per la scala. Usando il default {default_scale:.8f} µm/px. Dettagli: {e}")
        
    return default_scale

# --- Funzioni di Calcolo Principali ---

def calcola_larghezza_trincea(immagine, fattore_di_scala_um_per_px):
    """
    Misura la larghezza della trincea utilizzando soglia automatica (Otsu)
    su una singola linea di scansione, spostabile con i tasti freccia.
    """
    global punti_selezionati, immagine_originale, window_name, current_mode, scan_line_y, trench_width_data, reference_line_params
    
    immagine_originale = immagine.copy() # Immagine 8-bit
    window_name = "Misura Larghezza Trincea"
    current_mode = 'TRINCEA'
    trench_width_data = {"larghezza_trincea_um": "Non misurata", "threshold_metodo": "N/D", "threshold_valore": "N/D"}

    # Inizializza parametri della retta di scansione
    if reference_line_params is None:
        print("\nATTENZIONE: Retta di riferimento (Gradino) non calcolata. La scansione è Orizzontale (m=0).")
        m, q = 0.0, immagine_originale.shape[0] // 2 # Centra la Y
    else:
        m, q = reference_line_params
        print("\nRetta di riferimento caricata dal Gradino. Usa frecce SU/GIÙ per spostare l'altezza (Q).")

    scan_line_q_offset = q # q diventa l'offset che l'utente modifica
    STEP_SIZE = 2 # Passo di spostamento in pixel

    def redraw_trench_view(m, current_q):
        """Disegna la retta e la linea di scansione con i risultati della soglia."""
        nonlocal scan_line_q_offset
        global scan_line_y
        
        # Converte l'immagine grayscale in BGR per il disegno
        immagine_modificabile = cv2.cvtColor(immagine_originale.copy(), cv2.COLOR_GRAY2BGR)
        
        # 1. Calcola il colore dinamico della retta
        p_left = (0, int(current_q))
        p_right = (immagine_originale.shape[1] - 1, int(m * (immagine_originale.shape[1] - 1) + current_q))
        
        line_color = get_dynamic_line_color(p_left, p_right, immagine_originale, DARK_THRESHOLD)
        
        # 2. Disegna la retta spostata (Linea di Scansione)
        draw_extended_line(immagine_modificabile, m, current_q, line_color, text=f"Linea di Scansione (Q: {current_q:.2f})")
        
        # La Y di scansione è l'intercetta sull'asse Y=0 (quindi Q)
        scan_line_y = int(current_q) 

        # 3. Disegna il risultato della soglia (se presente)
        binarized_line = trench_width_data.get('line_binarizzata')
        if binarized_line is not None and scan_line_y is not None:
            # Rileva gli indici di inizio/fine trincea (come in calculate_trench_width)
            diff = np.diff(binarized_line)
            start_indices = np.where(diff == -1)[0] + 1
            end_indices = np.where(diff == 1)[0] + 1
            
            if start_indices.size and end_indices.size and end_indices[0] > start_indices[0]:
                start_px = start_indices[0]
                end_px = end_indices[0]
                
                # Calcola la Y per il disegno usando la media X per precisione
                mid_x = int((start_px + end_px) / 2)
                draw_y_start = int(m * start_px + current_q)
                draw_y_end = int(m * end_px + current_q)
                
                # Disegna un segmento di linea sulla retta tra i bordi (Rosso brillante)
                cv2.line(immagine_modificabile, (start_px, draw_y_start), (end_px, draw_y_end), (0, 0, 255), 3)
                
                # Aggiunge testo al centro del segmento
                larghezza_um = trench_width_data.get('larghezza_trincea_um', 'N/D')
                cv2.putText(immagine_modificabile, f"Larghezza: {larghezza_um} um", (mid_x - 50, int(m*mid_x + current_q) - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        cv2.imshow(window_name, immagine_modificabile)
        
    # Disegno iniziale
    cv2.namedWindow(window_name)
    redraw_trench_view(m, scan_line_q_offset)

    print("\n--- Misura Larghezza Trincea ---")
    print("Comandi:")
    print("    FRECCE SU/GIÙ: Sposta la linea di scansione (modifica Q).")
    print("    'R': Stampa i dettagli del Profilo di intensità della linea attuale.")
    print("    'T': Soglia Otsu Automatica sul profilo.")
    print("    'M': Soglia Manuale (0-255).")
    print("    'W': Calcola la Larghezza dai bordi rilevati (e salva).")
    print("    'Q': Esci/SALTA la misurazione corrente.")
    
    manual_input_mode = False
    temp_threshold_value = ""

    while True:
        key = cv2.waitKey(1) & 0xFF
        
        # Gestione dell'input manuale (per il threshold)
        if manual_input_mode:
            if key == 13: # INVIO
                if scan_line_y is not None and manual_threshold(get_line_profile(immagine_originale, scan_line_y), trench_width_data, temp_threshold_value):
                    manual_input_mode = False
                    redraw_trench_view(m, scan_line_q_offset) 
                else:
                    print("\nErrore nella Soglia Manuale. Riprova.")
                    manual_input_mode = False
                temp_threshold_value = ""
            elif key == 8: # BACKSPACE
                temp_threshold_value = temp_threshold_value[:-1]
                sys.stdout.write(f"Inserisci Soglia Manuale (0-255) [{temp_threshold_value}]  \r")
                sys.stdout.flush()
            elif 48 <= key <= 57: # Numeri 0-9
                temp_threshold_value += chr(key)
                sys.stdout.write(f"Inserisci Soglia Manuale (0-255) [{temp_threshold_value}]  \r")
                sys.stdout.flush()
            elif key == 27: # ESC 
                manual_input_mode = False
                print("\nModalità Soglia Manuale annullata.")
        
        # Gestione dei comandi da tastiera e frecce 
        elif not manual_input_mode:
            
            if key == KEY_UP: # Freccia SU
                scan_line_q_offset -= STEP_SIZE
                scan_line_q_offset = np.clip(scan_line_q_offset, 0, immagine_originale.shape[0] - 1)
                redraw_trench_view(m, scan_line_q_offset)
                
            elif key == KEY_DOWN: # Freccia GIÙ
                scan_line_q_offset += STEP_SIZE
                scan_line_q_offset = np.clip(scan_line_q_offset, 0, immagine_originale.shape[0] - 1)
                redraw_trench_view(m, scan_line_q_offset)

            elif key == ord('r'):
                if scan_line_y is not None:
                    profile = get_line_profile(immagine_originale, scan_line_y)
                    plot_profile(profile, f"Profilo Y={scan_line_y}")
                else:
                    print("Errore: La linea di scansione non è definita (premi una freccia).")
                
            elif key == ord('t'):
                if scan_line_y is not None:
                    auto_trench_detect(get_line_profile(immagine_originale, scan_line_y), trench_width_data)
                    redraw_trench_view(m, scan_line_q_offset) # Aggiorna la vista
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
                    redraw_trench_view(m, scan_line_q_offset)
                    cv2.waitKey(0) # Attende qualsiasi tasto per chiudere la finestra dopo il calcolo finale
                    break
                
            elif key == ord('q') or key == 27: # Q o ESC
                break
            
            # Gestione dei tasti freccia non mappati come 82/84 (per ambienti diversi)
            elif key >= 0 and key != 255 and key != KEY_UP and key != KEY_DOWN:
                pass


    cv2.destroyAllWindows()
    current_mode = None
    
    # Rimuovi dati temporanei non necessari per l'output finale
    if 'line_binarizzata' in trench_width_data:
        del trench_width_data['line_binarizzata']
        
    if trench_width_data.get('larghezza_trincea_um') == 'Non misurata':
        trench_width_data = {"larghezza_trincea_um": "Non misurata"}
    
    return trench_width_data

# --- Callback per il Mouse Generalizzato ---

def on_mouse(event, x, y, flags, param):
    """
    Funzione di callback per catturare gli eventi del mouse e disegnare in base al current_mode.
    """
    global punti_selezionati, immagine_modificabile, window_name, current_mode, scan_line_y, immagine_originale, reference_line_params

    if event == cv2.EVENT_LBUTTONDOWN:
        
        # 1. Modalità Gradino/Angolo (3 punti necessari)
        if current_mode in ('GRADINO', 'ANGOLO'):
            if len(punti_selezionati) < 3:
                punti_selezionati.append((x, y))
                print(f"Punto selezionato: ({x}, {y}) - Clic {len(punti_selezionati)} di 3")
                
                # Resetta l'immagine e ridisegna tutti i punti
                immagine_modificabile = cv2.cvtColor(immagine_originale.copy(), cv2.COLOR_GRAY2BGR)
                
                # Disegna i punti
                for i, (px, py) in enumerate(punti_selezionati):
                    intensity = get_pixel_intensity(immagine_originale, px, py)
                    point_color = (0, 255, 0) if intensity < DARK_THRESHOLD else (55, 255, 55) # Verde
                    cv2.circle(immagine_modificabile, (px, py), 5, point_color, -1) 
                    cv2.putText(immagine_modificabile, str(i + 1), (px + 7, py + 7), cv2.FONT_HERSHEY_SIMPLEX, 0.5, point_color, 1)

                # --- LOGICA DI DISEGNO (INTERMEDIA) ---
                if len(punti_selezionati) == 2:
                    p1, p2 = punti_selezionati[0], punti_selezionati[1]
                    x1, y1 = p1
                    x2, y2 = p2
                    
                    line_color = get_dynamic_line_color(p1, p2, immagine_originale, DARK_THRESHOLD)
                    
                    if x2 != x1:
                        m = (y2 - y1) / (x2 - x1)
                        q = y1 - m * x1
                        if current_mode == 'GRADINO':
                             draw_extended_line(immagine_modificabile, m, q, line_color, text=f"Retta Riferimento (P1-P2) - m: {m:.2f}")
                        
                        # Memorizza i parametri per la trincea (anche se siamo in ANGLE, i primi due punti sono la retta di base)
                        reference_line_params = (m, q)
                        
                    elif current_mode == 'GRADINO':
                        # Linea verticale
                        cv2.line(immagine_modificabile, (x1, 0), (x1, immagine_modificabile.shape[0] - 1), line_color, 2)
                        cv2.putText(immagine_modificabile, f"Retta Riferimento Verticale", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, line_color, 1)
                        reference_line_params = None # Non calcolabile in m, q standard

                # Disegna i segmenti per l'angolo (sempre visibili in modalità ANGLE)
                if len(punti_selezionati) >= 2 and current_mode == 'ANGOLO':
                    cv2.line(immagine_modificabile, punti_selezionati[0], punti_selezionati[1], (255, 0, 0), 2) # Segmento P1-P2 (Blu)
                if len(punti_selezionati) == 3 and current_mode == 'ANGOLO':
                    cv2.line(immagine_modificabile, punti_selezionati[1], punti_selezionati[2], (255, 0, 0), 2) # Segmento P2-P3 (Blu)
                    
                cv2.imshow(window_name, immagine_modificabile)
            else:
                print("Hai già selezionato 3 punti. Premi 'c' per calcolare o 'z' per annullare.")
            
        # 2. Modalità Selezione Linea Trincea Orizzontale (1 punto necessario) -> Tasto 'L'
        elif current_mode == 'TRINCEA_LINE':
            scan_line_y = y
            print(f"Linea di scansione Y selezionata: {y} (Orizzontale). Premi 'T' per il calcolo Otsu.")
            
            immagine_modificabile = cv2.cvtColor(immagine_originale.copy(), cv2.COLOR_GRAY2BGR)
            cv2.line(immagine_modificabile, (0, y), (immagine_modificabile.shape[1], y), (255, 255, 255), 2)
            cv2.putText(immagine_modificabile, f"Y: {y} (Orizzontale)", (10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            cv2.imshow(window_name, immagine_modificabile)
            current_mode = 'TRINCEA' # Torna alla modalità principale Trincea
            
        # 3. Modalità Selezione Linea Trincea X-Coordinate (1 punto necessario) -> Tasto 'X'
        elif current_mode == 'TRINCEA_X_SELECT':
            if reference_line_params is None:
                print("Errore: Definire prima la retta di riferimento nel calcolo del gradino ('G'/'c').")
                current_mode = 'TRINCEA'
                return
            
            m, q = reference_line_params
            x_scan = x
            
            # Calcola la Y sulla retta definita
            scan_line_y_calculated = int(m * x_scan + q)
            scan_line_y = scan_line_y_calculated # La scansione viene eseguita su questa Y orizzontale
            
            immagine_modificabile = cv2.cvtColor(immagine_originale.copy(), cv2.COLOR_GRAY2BGR)
            
            # Disegna la retta di riferimento completa (Verde chiaro)
            draw_extended_line(immagine_modificabile, m, q, (150, 255, 150), 1, "Retta Base")
            
            # Disegna la linea di scansione BIANCA (orizzontale) che passa per il punto X selezionato sulla retta
            cv2.line(immagine_modificabile, (0, scan_line_y_calculated), (immagine_modificabile.shape[1], scan_line_y_calculated), (255, 255, 255), 2)
            
            # Punto selezionato sulla retta (Rosso)
            cv2.circle(immagine_modificabile, (x_scan, scan_line_y_calculated), 5, (0, 0, 255), -1) 
            
            cv2.putText(immagine_modificabile, f"Y Scan: {scan_line_y_calculated} (da Retta a X={x_scan})", (10, scan_line_y_calculated - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            cv2.imshow(window_name, immagine_modificabile)
            current_mode = 'TRINCEA'
            print(f"Linea di scansione Y selezionata a Y={scan_line_y_calculated} (calcolata dalla retta alla X={x_scan}). Premi 'T' per il calcolo Otsu.")

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
        # Il valore float restituito è il più preciso possibile (dipendente da Python)
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
        # Stampa il fattore di scala con la nuova precisione
        print(f"Fattore di scala (pixelwidth) utilizzato: {fattore_di_scala_um_per_px:.8f} µm/px")
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
