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
# Fasi aggiornate (P1-P5 in Fase 0, P6-P7 in Fase 1, P8-P9 in Fase 2, P10-P11 in Fase 3)
fase_corrente = 0 # 0: Gradino/Angolo/Faceting (P1-P5), 1: Trincea Larghezza (P6-P7), 2: Micro Left (P8-P9), 3: Micro Right (P10-P11)

# Variabili per la visualizzazione e interazione (Pan)
visual_scale_factor = 1.0 
current_pan_offset = [0, 0] # Offset di spostamento per il panning
is_panning = False
last_mouse_pos = (0, 0)

# Variabili di stato usate nel codice

reference_line_params = None  # (m, q) o (None, x1) per la retta P1-P2 (Top Surface)
filepath_global = ""

def misura_immagine_interattiva(immagine_originale_in, filepath, scale_factor):
    """
    Gestisce il ciclo interattivo per la selezione dei punti di misurazione.
    """
    global immagine_originale, immagine_modificabile, punti_selezionati, window_name
    global fase_corrente, filepath_global, PIXEL_TO_UNIT_FACTOR, reference_line_params
    
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
    
    print("Premi 'c' per passare alla fase successiva, 'x' per annullare l'ultima selezione, 's' per salvare e procedere, 'q' per saltare l'immagine.") #voglio estendere queti comandi a tutti i punti.
    
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

    print(f"Selezionare P1 P2 per definire la retta di riferimento, p3 p4 per la parete di scavo, p5 per identificare il faceting")
    risultati_fase_0_stampati = False # Nuovo flag per tracciare lo stato
    p1 = None # Inizializzazione necessaria per le fasi successive
    p2 = None


    # Fase 0: Gradino/Angolo/Faceting (P1-P5)
    while fase_corrente == 0:
        # Ridondanza: Selezionare P1 P2 per definire la retta di riferimento, p3 p4 per la parete di scavo, p5 per identificare il faceting
        
        # LOGICA DI CALCOLO (Si attiva solo una volta raggiunti i 5 punti)
        if len(punti_selezionati) == 5 and not risultati_fase_0_stampati:
            # Calcolo e log per Gradino/Angolo/Faceting
            p1, p2, p3, p4, p5 = punti_selezionati[0:5]
            gradino_unit_val, angolo_gradi_val, _, delta_x_prime_px, delta_y_prime_px = gradino_angolo_faceting(punti_selezionati, PIXEL_TO_UNIT_FACTOR) 

            # P1 è l'origine (0, 0)
            p1_x_prime, p1_y_prime = (0.0, 0.0)

            #angolo di riferimento
            angle_p1p2 = get_p1p2_angle(p1, p2)

            # Trasformazione P2 (definisce l'asse X')
            p2_x_prime, p2_y_prime = transform_point(p2, p1, angle_p1p2)

            # Calcolo del Faceting (distanza proiettata P3 - P5)
            p3_proj = project_point_onto_line(p3, p1, p2)

            # Trasformazione P4 (fondo)
            p4_x_prime, p4_y_prime = transform_point(p4, p1, angle_p1p2) 
            
            # Trasformazione nel sistema solidale
            
            p3_x_prime, p3_y_prime = transform_point(p3_proj, p1, angle_p1p2)
            p5_x_prime, p5_y_prime = transform_point(p5, p1, angle_p1p2)

            faceting_px = abs(p3_x_prime - p5_x_prime)
            faceting_unit_val = faceting_px * PIXEL_TO_UNIT_FACTOR

            risultato_immagine.update({
                'gradino_um': gradino_unit_val,
                'angolo_gradi': angolo_gradi_val,
                'faceting_um': faceting_unit_val,
                'delta_x_prime_um': delta_x_prime_px * PIXEL_TO_UNIT_FACTOR,
                'delta_y_prime_um': delta_y_prime_px * PIXEL_TO_UNIT_FACTOR,

                # NUOVE CHIAVI AGGIUNTE: X' dei punti che definiscono il Faceting
                'p3_x_prime_px': p3_x_prime, 'p3_y_prime_px': p3_y_prime,
                'p5_x_prime_px': p5_x_prime, 'p5_y_prime_px': p5_y_prime,
                'p3_x_prime_um': p3_x_prime * PIXEL_TO_UNIT_FACTOR, 'p3_y_prime_um': p3_y_prime * PIXEL_TO_UNIT_FACTOR,
                'p5_x_prime_um': p5_x_prime * PIXEL_TO_UNIT_FACTOR, 'p5_y_prime_um': p5_y_prime * PIXEL_TO_UNIT_FACTOR
            })
            
            # Aggiornamento del dizionario risultato_immagine
            risultato_immagine.update({
                'p1_x_prime_px': p1_x_prime, 'p1_y_prime_px': p1_y_prime,
                'p2_x_prime_px': p2_x_prime, 'p2_y_prime_px': p2_y_prime,
                'p4_x_prime_px': p4_x_prime, 'p4_y_prime_px': p4_y_prime,
                # Le versioni in um le calcoliamo subito:
                'p1_x_prime_um': p1_x_prime * PIXEL_TO_UNIT_FACTOR, 'p1_y_prime_um': p1_y_prime * PIXEL_TO_UNIT_FACTOR,
                'p2_x_prime_um': p2_x_prime * PIXEL_TO_UNIT_FACTOR, 'p2_y_prime_um': p2_y_prime * PIXEL_TO_UNIT_FACTOR,
                'p4_x_prime_um': p4_x_prime * PIXEL_TO_UNIT_FACTOR, 'p4_y_prime_um': p4_y_prime * PIXEL_TO_UNIT_FACTOR,
            })
            
            print(f"Risultato Gradino/Angolo/Faceting salvato. Gradino: {gradino_unit_val:.4f} um, Angolo: {angolo_gradi_val:.2f} deg, Faceting (P3'-P5'): {faceting_unit_val:.4f} um.") 
            print("Premi 'c' per procedere alla misurazione Trincea (P6-P7).")
            risultati_fase_0_stampati = True
        
        # GESTIONE INPUT (Deve essere sempre eseguita nel loop)
        k = cv2.waitKey(50) & 0xFF
        
        if k == ord('c'):
            if len(punti_selezionati) == 5:
                if risultati_fase_0_stampati: # Assicurati che i calcoli siano stati fatti
                    fase_corrente = 1 # Passa alla Fase 1
                    print("\nPassaggio alla Fase 1: Selezionare 2 punti (P6-P7) per la larghezza della Trincea.")
                    redraw_image()
                else:
                    print("Calcoli ancora non completati. Premi 'c' di nuovo.") # Non dovrebbe accadere se il flag è gestito bene
            else:
                print(f"Devi selezionare 5 punti (P1-P5). Punti attuali: {len(punti_selezionati)}.")
                
        # Input di Skip e Save (Q e S)
        elif k == ord('q'): # Skip image
            risultato_immagine['status'] = 'Skipped'
            return risultato_immagine
        elif k == ord('s'): # Save and exit
            risultato_immagine['status'] = 'Completed' if len(punti_selezionati) >= 5 else 'Incomplete'
            return risultato_immagine
        elif k == ord('x') or k == 27: # Esc or x to undo
            if len(punti_selezionati) > 0:
                punti_selezionati.pop()
                punti_rimanenti = len(punti_selezionati)
                print(f"Ultimo punto rimosso. Punti rimanenti: {punti_rimanenti}")
                
                # Reset dei flag di stampa e dei dati se annullata la fase
                if punti_rimanenti < 5:
                    risultati_fase_0_stampati = False
                    risultato_immagine['gradino_um'] = None 
                    risultato_immagine['angolo_gradi'] = None
                    risultato_immagine['faceting_um'] = None
                if punti_rimanenti < 2:
                    reference_line_params = None
                    
                redraw_image()
            else:
                print("Nessun punto da rimuovere.")
        # --- BLOCCO AGGIUNTO: SPOSTAMENTO DEL PUNTO CORRENTE (Tasti i, k, j, l) ---
        # Si applica solo se almeno un punto è stato selezionato.
        if len(punti_selezionati) > 0:
            
            # Codici tasti di spostamento richiesti
            UP_KEY = ord('i') 
            DOWN_KEY = ord('k')
            LEFT_KEY = ord('j')
            RIGHT_KEY = ord('l')
            
            # Definizione dello step di spostamento (in pixel)
            MOVE_STEP_PX = 1 # Spostamento fine
            
            # Ottieni l'indice dell'ultimo punto selezionato (il punto corrente)
            idx_punto_corrente = len(punti_selezionati) - 1
            current_x, current_y = punti_selezionati[idx_punto_corrente]
            
            # Spostamento
            x_delta, y_delta = 0, 0
            
            # NOTA: Usiamo k (il tasto premuto) direttamente
            if k == UP_KEY:
                y_delta = -MOVE_STEP_PX
            elif k == DOWN_KEY:
                y_delta = MOVE_STEP_PX
            elif k == LEFT_KEY:
                x_delta = -MOVE_STEP_PX
            elif k == RIGHT_KEY:
                x_delta = MOVE_STEP_PX
            
            if x_delta != 0 or y_delta != 0:
                # Applica il nuovo spostamento
                new_x = current_x + x_delta
                new_y = current_y + y_delta
                
                # Sostituisci l'ultimo punto nella lista
                # Questo è cruciale per aggiornare la posizione del punto!
                punti_selezionati[idx_punto_corrente] = (new_x, new_y)
                
                # Gestione speciale per P5 (indice 4) che è sempre proiettato
                # Se il punto spostato è P1 o P2, P5 (se esiste) deve essere ricalcolato e aggiornato.
                if len(punti_selezionati) >= 5 and (idx_punto_corrente == 0 or idx_punto_corrente == 1):
                    # Ricalcola la retta di riferimento (reference_line_params)
                    p1, p2 = punti_selezionati[0], punti_selezionati[1]
                    x1, y1 = p1
                    x2, y2 = p2
                    
                    if abs(x2 - x1) > 1e-6:
                        m = (y2 - y1) / (x2 - x1)
                        q = y1 - m * x1
                        
                        reference_line_params = (m, q)
                    else:
                        
                        reference_line_params = (None, x1)
                        
                    # Ricalcola e aggiorna P5 (l'indice 4) sulla nuova retta
                    p5_originale = punti_selezionati[4] # P5 è memorizzato come punto proiettato
                    # Per spostare P1 o P2, non vogliamo che P5 rimanga lo stesso, ma che sia riproiettato
                    # Sostituisci P5 con la sua nuova proiezione sulla nuova retta P1-P2.
                    p5_proiettato_np = project_point_onto_line(p5_originale, p1, p2)
                    punti_selezionati[4] = tuple(p5_proiettato_np.tolist())
                    
                    print(f"P{idx_punto_corrente+1} spostato. La retta di riferimento e P5 sono stati aggiornati.")

                elif len(punti_selezionati) == 11 and idx_punto_corrente >= 9:
                    # Gestione per P10 e P11
                    print(f"P{idx_punto_corrente+1} spostato.")
                    pass
                else:
                    print(f"P{idx_punto_corrente+1} spostato. Coordinate: ({new_x}, {new_y})")
                    
                # Ridisegna l'immagine per visualizzare lo spostamento
                redraw_image()
        
        # --- FINE BLOCCO AGGIUNTO ---        

        # Se l'immagine viene aggiornata dalla callback del mouse, è necessario ridisegnare
        redraw_image()


    # Fase 1: Larghezza Trincea (P6, P7)
    risultati_fase_1_stampati = False
    while fase_corrente == 1:
        # LOGICA DI CALCOLO (Si attiva solo una volta raggiunti i 7 punti totali)
        if len(punti_selezionati) == 7 and not risultati_fase_1_stampati:
            p1, p2 = punti_selezionati[0], punti_selezionati[1]
            p6, p7 = punti_selezionati[5], punti_selezionati[6]

            # P1 è l'origine (0, 0)
            p1_x_prime, p1_y_prime = (0.0, 0.0)

            # Calcola l'angolo di riferimento (essenziale)
            angle_p1p2 = get_p1p2_angle(p1, p2)

            # Calcolo X'Y' per P6 e P7 (Solidale P1-P2)
            p6_x_prime, p6_y_prime = transform_point(p6, p1, angle_p1p2)
            p7_x_prime, p7_y_prime = transform_point(p7, p1, angle_p1p2)
            
            # La larghezza della trincea è la distanza euclidea tra P6 e P7 (entrambi proiettati)
            # NOTA: p1 e p2 devono essere disponibili qui (vengono definiti nella Fase 0 e devono essere accessibili in locale o globale)
            trincea_unit_val, trincea_px_val = microtrenching_calc(p6, p7, p1, p2, PIXEL_TO_UNIT_FACTOR) 
            
            risultato_immagine.update({
                'trincea_larghezza_um': trincea_unit_val,
                'trincea_larghezza_px': trincea_px_val,
                'p6_x_prime_px': p6_x_prime, 'p6_y_prime_px': p6_y_prime,
                'p7_x_prime_px': p7_x_prime, 'p7_y_prime_px': p7_y_prime,
                
                # COORDINATE X'Y' (MICRON)
                'p6_x_prime_um': p6_x_prime * PIXEL_TO_UNIT_FACTOR, 'p6_y_prime_um': p6_y_prime * PIXEL_TO_UNIT_FACTOR,
                'p7_x_prime_um': p7_x_prime * PIXEL_TO_UNIT_FACTOR, 'p7_y_prime_um': p7_y_prime * PIXEL_TO_UNIT_FACTOR
            })
            if len(punti_selezionati) == 7 and not risultati_fase_1_stampati:
                print(f"Risultato Trincea salvato. Larghezza: {trincea_unit_val:.4f} um.")
                print("Premi 'c' per procedere alla misurazione Micro Trench Sinistro (P8-P9) o s per saltare e andare ai commenti.")
                risultati_fase_1_stampati = True

        # GESTIONE INPUT
        k = cv2.waitKey(50) & 0xFF
        
        if k == ord('c'):
            if len(punti_selezionati) == 7:
                if risultati_fase_1_stampati:
                    fase_corrente = 2 # Passa alla Fase 2
                    print("\nPassaggio alla Fase 2: Selezionare 2 punti (P8-P9) per Micro Trench Sinistro.")
                    redraw_image()
                else:
                    print("Calcoli ancora non completati.")
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
                    risultati_fase_1_stampati = False
                    risultato_immagine['trincea_larghezza_um'] = None 
                redraw_image()
            else:
                # Se siamo sotto P6, torniamo alla fase precedente
                fase_corrente = 0
                print("Punti della Fase 0 modificati. Tornato alla Fase 0.")
                redraw_image()
                break # Esci dal ciclo Fase 1 per tornare a Fase 0
        # --- BLOCCO AGGIUNTO: SPOSTAMENTO DEL PUNTO CORRENTE (Tasti i, k, j, l) ---
        # Si applica solo se almeno un punto è stato selezionato.
        if len(punti_selezionati) > 0:
            
            # Codici tasti di spostamento richiesti
            UP_KEY = ord('i') 
            DOWN_KEY = ord('k')
            LEFT_KEY = ord('j')
            RIGHT_KEY = ord('l')
            
            # Definizione dello step di spostamento (in pixel)
            MOVE_STEP_PX = 1 # Spostamento fine
            
            # Ottieni l'indice dell'ultimo punto selezionato (il punto corrente)
            idx_punto_corrente = len(punti_selezionati) - 1
            current_x, current_y = punti_selezionati[idx_punto_corrente]
            
            # Spostamento
            x_delta, y_delta = 0, 0
            
            # NOTA: Usiamo k (il tasto premuto) direttamente
            if k == UP_KEY:
                y_delta = -MOVE_STEP_PX
            elif k == DOWN_KEY:
                y_delta = MOVE_STEP_PX
            elif k == LEFT_KEY:
                x_delta = -MOVE_STEP_PX
            elif k == RIGHT_KEY:
                x_delta = MOVE_STEP_PX
            
            if x_delta != 0 or y_delta != 0:
                # Applica il nuovo spostamento
                new_x = current_x + x_delta
                new_y = current_y + y_delta
                
                # Sostituisci l'ultimo punto nella lista
                # Questo è cruciale per aggiornare la posizione del punto!
                punti_selezionati[idx_punto_corrente] = (new_x, new_y)
                
                # Gestione speciale per P5 (indice 4) che è sempre proiettato
                # Se il punto spostato è P1 o P2, P5 (se esiste) deve essere ricalcolato e aggiornato.
                if len(punti_selezionati) >= 5 and (idx_punto_corrente == 0 or idx_punto_corrente == 1):
                    # Ricalcola la retta di riferimento (reference_line_params)
                    p1, p2 = punti_selezionati[0], punti_selezionati[1]
                    x1, y1 = p1
                    x2, y2 = p2
                    
                    if abs(x2 - x1) > 1e-6:
                        m = (y2 - y1) / (x2 - x1)
                        q = y1 - m * x1
                        
                        reference_line_params = (m, q)
                    else:
                        
                        reference_line_params = (None, x1)
                        
                    # Ricalcola e aggiorna P5 (l'indice 4) sulla nuova retta
                    p5_originale = punti_selezionati[4] # P5 è memorizzato come punto proiettato
                    # Per spostare P1 o P2, non vogliamo che P5 rimanga lo stesso, ma che sia riproiettato
                    # Sostituisci P5 con la sua nuova proiezione sulla nuova retta P1-P2.
                    p5_proiettato_np = project_point_onto_line(p5_originale, p1, p2)
                    punti_selezionati[4] = tuple(p5_proiettato_np.tolist())
                    
                    print(f"P{idx_punto_corrente+1} spostato. La retta di riferimento e P5 sono stati aggiornati.")

                elif len(punti_selezionati) == 11 and idx_punto_corrente >= 9:
                    # Gestione per P10 e P11
                    print(f"P{idx_punto_corrente+1} spostato.")
                    pass
                else:
                    print(f"P{idx_punto_corrente+1} spostato. Coordinate: ({new_x}, {new_y})")
                    
                # Ridisegna l'immagine per visualizzare lo spostamento
                redraw_image()
        
        # --- FINE BLOCCO AGGIUNTO ---                        
        redraw_image()
        

    # Fase 2: Micro Trench Sinistro (P8, P9)
    risultati_fase_2_stampati = False
    while fase_corrente == 2:
        # LOGICA DI CALCOLO
        if len(punti_selezionati) == 9 and not risultati_fase_2_stampati:
            p1, p2 = punti_selezionati[0], punti_selezionati[1]
            p8, p9 = punti_selezionati[7], punti_selezionati[8]

            # P1 è l'origine (0, 0)
            p1_x_prime, p1_y_prime = (0.0, 0.0)
            
            # Calcola l'angolo di riferimento (essenziale)
            angle_p1p2 = get_p1p2_angle(p1, p2)
            
            # Calcolo X'Y' per P8 e P9 (Solidale P1-P2)
            p8_x_prime, p8_y_prime = transform_point(p8, p1, angle_p1p2)
            p9_x_prime, p9_y_prime = transform_point(p9, p1, angle_p1p2)
            
            # Il micro trenching viene misurato come differenza di gradino (Y')
            micro_left_unit_val, micro_left_px_val = microtrenching_calc(p8, p9, p1, p2, PIXEL_TO_UNIT_FACTOR) 
            
            
            risultato_immagine.update({
                'micro_left_um': micro_left_unit_val,
                'micro_left_px': micro_left_px_val,
                # COORDINATE X'Y' (PIXEL)
                'p8_x_prime_px': p8_x_prime, 'p8_y_prime_px': p8_y_prime,
                'p9_x_prime_px': p9_x_prime, 'p9_y_prime_px': p9_y_prime,
                
                # COORDINATE X'Y' (MICRON)
                'p8_x_prime_um': p8_x_prime * PIXEL_TO_UNIT_FACTOR, 'p8_y_prime_um': p8_y_prime * PIXEL_TO_UNIT_FACTOR,
                'p9_x_prime_um': p9_x_prime * PIXEL_TO_UNIT_FACTOR, 'p9_y_prime_um': p9_y_prime * PIXEL_TO_UNIT_FACTOR
            })
            
            print(f"Risultato Micro Trench Sinistro salvato. Gradino: {micro_left_unit_val:.4f} um.")
            print("Premi 'c' per procedere alla misurazione Micro Trench Destro (P10-P11).")
            risultati_fase_2_stampati = True

        # GESTIONE INPUT
        k = cv2.waitKey(50) & 0xFF
        
        if k == ord('c'):
            if len(punti_selezionati) == 9:
                if risultati_fase_2_stampati:
                    fase_corrente = 3 # Passa alla Fase 3
                    print("\nPassaggio alla Fase 3: Selezionare 2 punti (P10-P11) per Micro Trench Destro.")
                    redraw_image()
                else:
                    print("Calcoli ancora non completati.")
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
                    risultati_fase_2_stampati = False
                    risultato_immagine['micro_left_um'] = None 
                redraw_image()
            else:
                # Se siamo sotto P8, torniamo alla fase precedente
                fase_corrente = 1
                print("Punti della Fase 1 modificati. Tornato alla Fase 1.")
                redraw_image()
                break
        # --- BLOCCO AGGIUNTO: SPOSTAMENTO DEL PUNTO CORRENTE (Tasti i, k, j, l) ---
        # Si applica solo se almeno un punto è stato selezionato.
        if len(punti_selezionati) > 0:
            
            # Codici tasti di spostamento richiesti
            UP_KEY = ord('i') 
            DOWN_KEY = ord('k')
            LEFT_KEY = ord('j')
            RIGHT_KEY = ord('l')
            
            # Definizione dello step di spostamento (in pixel)
            MOVE_STEP_PX = 1 # Spostamento fine
            
            # Ottieni l'indice dell'ultimo punto selezionato (il punto corrente)
            idx_punto_corrente = len(punti_selezionati) - 1
            current_x, current_y = punti_selezionati[idx_punto_corrente]
            
            # Spostamento
            x_delta, y_delta = 0, 0
            
            # NOTA: Usiamo k (il tasto premuto) direttamente
            if k == UP_KEY:
                y_delta = -MOVE_STEP_PX
            elif k == DOWN_KEY:
                y_delta = MOVE_STEP_PX
            elif k == LEFT_KEY:
                x_delta = -MOVE_STEP_PX
            elif k == RIGHT_KEY:
                x_delta = MOVE_STEP_PX
            
            if x_delta != 0 or y_delta != 0:
                # Applica il nuovo spostamento
                new_x = current_x + x_delta
                new_y = current_y + y_delta
                
                # Sostituisci l'ultimo punto nella lista
                # Questo è cruciale per aggiornare la posizione del punto!
                punti_selezionati[idx_punto_corrente] = (new_x, new_y)
                
                # Gestione speciale per P5 (indice 4) che è sempre proiettato
                # Se il punto spostato è P1 o P2, P5 (se esiste) deve essere ricalcolato e aggiornato.
                if len(punti_selezionati) >= 5 and (idx_punto_corrente == 0 or idx_punto_corrente == 1):
                    # Ricalcola la retta di riferimento (reference_line_params)
                    p1, p2 = punti_selezionati[0], punti_selezionati[1]
                    x1, y1 = p1
                    x2, y2 = p2
                    
                    if abs(x2 - x1) > 1e-6:
                        m = (y2 - y1) / (x2 - x1)
                        q = y1 - m * x1
                        
                        reference_line_params = (m, q)
                    else:
                        
                        reference_line_params = (None, x1)
                        
                    # Ricalcola e aggiorna P5 (l'indice 4) sulla nuova retta
                    p5_originale = punti_selezionati[4] # P5 è memorizzato come punto proiettato
                    # Per spostare P1 o P2, non vogliamo che P5 rimanga lo stesso, ma che sia riproiettato
                    # Sostituisci P5 con la sua nuova proiezione sulla nuova retta P1-P2.
                    p5_proiettato_np = project_point_onto_line(p5_originale, p1, p2)
                    punti_selezionati[4] = tuple(p5_proiettato_np.tolist())
                    
                    print(f"P{idx_punto_corrente+1} spostato. La retta di riferimento e P5 sono stati aggiornati.")

                elif len(punti_selezionati) == 11 and idx_punto_corrente >= 9:
                    # Gestione per P10 e P11
                    print(f"P{idx_punto_corrente+1} spostato.")
                    pass
                else:
                    print(f"P{idx_punto_corrente+1} spostato. Coordinate: ({new_x}, {new_y})")
                    
                # Ridisegna l'immagine per visualizzare lo spostamento
                redraw_image()
        
        # --- FINE BLOCCO AGGIUNTO ---                        
        redraw_image()


    # Fase 3: Micro Trench Destro (P10, P11)
    risultati_fase_3_stampati = False
    while fase_corrente == 3:
        # LOGICA DI CALCOLO
        if len(punti_selezionati) == 11 and not risultati_fase_3_stampati:
            p1, p2 = punti_selezionati[0], punti_selezionati[1]
            p10, p11 = punti_selezionati[9], punti_selezionati[10]

            # P1 è l'origine (0, 0)
            p1_x_prime, p1_y_prime = (0.0, 0.0)
                        
            # Calcola l'angolo di riferimento (essenziale)
            angle_p1p2 = get_p1p2_angle(p1, p2)
            
            # Calcolo X'Y' per P10 e P11 (Solidale P1-P2)
            p10_x_prime, p10_y_prime = transform_point(p10, p1, angle_p1p2)
            p11_x_prime, p11_y_prime = transform_point(p11, p1, angle_p1p2)
            
            # Il micro trenching viene misurato come differenza di gradino (Y')
            micro_right_unit_val, micro_right_px_val = microtrenching_calc(p10, p11, p1, p2, PIXEL_TO_UNIT_FACTOR)

            # AGGIORNAMENTO LOGGING P10 e P11
            risultato_immagine.update({
                'micro_right_um': micro_right_unit_val,
                'micro_right_px': micro_right_px_val,
                
                # COORDINATE X'Y' (PIXEL)
                'p10_x_prime_px': p10_x_prime, 'p10_y_prime_px': p10_y_prime,
                'p11_x_prime_px': p11_x_prime, 'p11_y_prime_px': p11_y_prime,
                
                # COORDINATE X'Y' (MICRON)
                'p10_x_prime_um': p10_x_prime * PIXEL_TO_UNIT_FACTOR, 'p10_y_prime_um': p10_y_prime * PIXEL_TO_UNIT_FACTOR,
                'p11_x_prime_um': p11_x_prime * PIXEL_TO_UNIT_FACTOR, 'p11_y_prime_um': p11_y_prime * PIXEL_TO_UNIT_FACTOR,
            })
            
            
            
            print(f"Risultato Micro Trench Destro salvato. Gradino: {micro_right_unit_val:.4f} um.")
            print("Misurazione completata. Premi 's' per salvare o 'q' per saltare.")
            risultati_fase_3_stampati = True

        # GESTIONE INPUT
        k = cv2.waitKey(50) & 0xFF
        
        # Nessun tasto 'c' perché è l'ultima fase
        
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
                    risultati_fase_3_stampati = False
                    risultato_immagine['micro_right_um'] = None 
                redraw_image()
            else:
                # Se siamo sotto P10, torniamo alla fase precedente
                fase_corrente = 2
                print("Punti della Fase 2 modificati. Tornato alla Fase 2.")
                redraw_image()
                break
        # --- BLOCCO AGGIUNTO: SPOSTAMENTO DEL PUNTO CORRENTE (Tasti i, k, j, l) ---
        # Si applica solo se almeno un punto è stato selezionato.
        if len(punti_selezionati) > 0:
            
            # Codici tasti di spostamento richiesti
            UP_KEY = ord('i') 
            DOWN_KEY = ord('k')
            LEFT_KEY = ord('j')
            RIGHT_KEY = ord('l')
            
            # Definizione dello step di spostamento (in pixel)
            MOVE_STEP_PX = 1 # Spostamento fine
            
            # Ottieni l'indice dell'ultimo punto selezionato (il punto corrente)
            idx_punto_corrente = len(punti_selezionati) - 1
            current_x, current_y = punti_selezionati[idx_punto_corrente]
            
            # Spostamento
            x_delta, y_delta = 0, 0
            
            # NOTA: Usiamo k (il tasto premuto) direttamente
            if k == UP_KEY:
                y_delta = -MOVE_STEP_PX
            elif k == DOWN_KEY:
                y_delta = MOVE_STEP_PX
            elif k == LEFT_KEY:
                x_delta = -MOVE_STEP_PX
            elif k == RIGHT_KEY:
                x_delta = MOVE_STEP_PX
            
            if x_delta != 0 or y_delta != 0:
                # Applica il nuovo spostamento
                new_x = current_x + x_delta
                new_y = current_y + y_delta
                
                # Sostituisci l'ultimo punto nella lista
                # Questo è cruciale per aggiornare la posizione del punto!
                punti_selezionati[idx_punto_corrente] = (new_x, new_y)
                
                # Gestione speciale per P5 (indice 4) che è sempre proiettato
                # Se il punto spostato è P1 o P2, P5 (se esiste) deve essere ricalcolato e aggiornato.
                if len(punti_selezionati) >= 5 and (idx_punto_corrente == 0 or idx_punto_corrente == 1):
                    # Ricalcola la retta di riferimento (reference_line_params)
                    p1, p2 = punti_selezionati[0], punti_selezionati[1]
                    x1, y1 = p1
                    x2, y2 = p2
                    
                    if abs(x2 - x1) > 1e-6:
                        m = (y2 - y1) / (x2 - x1)
                        q = y1 - m * x1
                        
                        reference_line_params = (m, q)
                    else:
                        
                        reference_line_params = (None, x1)
                        
                    # Ricalcola e aggiorna P5 (l'indice 4) sulla nuova retta
                    p5_originale = punti_selezionati[4] # P5 è memorizzato come punto proiettato
                    # Per spostare P1 o P2, non vogliamo che P5 rimanga lo stesso, ma che sia riproiettato
                    # Sostituisci P5 con la sua nuova proiezione sulla nuova retta P1-P2.
                    p5_proiettato_np = project_point_onto_line(p5_originale, p1, p2)
                    punti_selezionati[4] = tuple(p5_proiettato_np.tolist())
                    
                    print(f"P{idx_punto_corrente+1} spostato. La retta di riferimento e P5 sono stati aggiornati.")

                elif len(punti_selezionati) == 11 and idx_punto_corrente >= 9:
                    # Gestione per P10 e P11
                    print(f"P{idx_punto_corrente+1} spostato.")
                    pass
                else:
                    print(f"P{idx_punto_corrente+1} spostato. Coordinate: ({new_x}, {new_y})")
                    
                # Ridisegna l'immagine per visualizzare lo spostamento
                redraw_image()
        
        # --- FINE BLOCCO AGGIUNTO ---                        
        redraw_image()
        

    # Ritorna risultati se il ciclo termina
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

    x_int = int(round(x))
    y_int = int(round(y))
    size_int = int(round(size))

    # Linea orizzontale
    cv2.line(img, (x_int - size, y_int), (x_int + size, y_int), color, thickness)
    # Linea verticale
    cv2.line(img, (x_int, y_int - size), (x_int, y_int + size), color, thickness)


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

    # --- NUOVO CALCOLO PER L'ANGOLO DI PENDENZA DEL FONDO (P3-P4 rispetto a X') ---
    angle_p1p2 = get_p1p2_angle(p1, p2) # Angolo di rotazione (già calcolato sotto, ma lo portiamo su)

    # Trasformiamo P3 e P4 nel sistema X'Y' (dove X' è parallelo a P1-P2)
    p3_x_prime, p3_y_prime = transform_point(p3, p1, angle_p1p2)
    p4_x_prime, p4_y_prime = transform_point(p4, p1, angle_p1p2)

    # Vettore del fondo nel sistema X'Y'
    v_fondo_x = p4_x_prime - p3_x_prime
    v_fondo_y = p4_y_prime - p3_y_prime

    # Calcolo dell'angolo del fondo rispetto all'asse X'
    # np.arctan2(dy, dx) dà l'angolo corretto nel quadrante [-pi, pi]
    angolo_rad = np.arctan2(v_fondo_y, v_fondo_x)
    angolo_gradi = np.degrees(angolo_rad)

    # Normalizzazione: assicuriamo che l'angolo sia tra 0 e 180 gradi
    # L'angolo deve essere l'inclinazione 'positiva' rispetto all'orizzontale.
    # Se è -ve (fondo in salita da sx a dx), lo convertiamo (ad es., -10 -> 170).
    # Se è > 180, lo convertiamo (ad es., 190 -> 10).
    # L'angolo è l'inclinazione del segmento (che tipicamente è piccolo, vicino a 0 o 180).
    if angolo_gradi < 0:
        # Se p3 e p4 sono a dx di p1-p2, l'angolo sarà 0-180 (per un segmento orizzontale)
        # Se usiamo arctan2, otterremo un angolo -180 < theta <= 180.
        # Per una pendenza dolce, potremmo ottenere un angolo vicino a 0 o +/- 180.
        # Se il fondo è in discesa da P3 a P4, l'angolo è tra 0 e -180.
        # Se il fondo è in salita da P3 a P4, l'angolo è tra 0 e 180.
        
        # Manteniamo la logica originale se questa è l'inclinazione, 
        # ma se vogliamo l'angolo acuto rispetto a X', usiamo abs().
        # **SCELTA PIU' PROBABILE PER UN GRADINO:**
        angolo_gradi = np.abs(angolo_gradi) # Angolo acuto rispetto all'asse X'
        if angolo_gradi > 90:
             angolo_gradi = 180.0 - angolo_gradi # Angolo acuto rispetto all'orizzontale

    # --- 3. Dati X'Y' per il log (usando il sistema solidale)
    # L'angle_p1p2 è già calcolato sopra.
    # p4_x_prime e p4_y_prime sono già calcolati sopra.

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
        cv2.line(immagine_modificabile, p1_orig, p2_orig, (255, 0, 0), 1) # Blu
        
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
        cv2.line(immagine_modificabile, start_p, end_p, (255, 255, 0), 1)


    # 3. Disegna tutti i punti selezionati (P1-P11)
    point_names = ['P1', 'P2', 'P3', 'P4', 'P5', 'P6', 'P7', 'P8', 'P9', 'P10', 'P11']
    for i, p_orig in enumerate(punti_selezionati):
        p_screen = p_orig # Le coordinate sono le coordinate schermo 1:1
        label = point_names[i]
        
        # Assegnazione colore in base alla fase
        if i < 2:
            color = (0, 255, 0) # Verde (Riferimento P1, P2)
        elif i < 5:
            color = (0, 0, 255) # Rosso (Faceting P3, P4, P5)
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
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
    cv2.putText(immagine_modificabile, f"Punti Totali Selezionati: {len(punti_selezionati)}", (10, 60), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
    
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
    """ Salva o aggiorna i risultati in un file XML. """
    # ----------------------------------------------------
    # 1. LOGICA DI LETTURA E APPEND (Mantenere la radice esistente)
    # ----------------------------------------------------
    if os.path.exists(nome_file):
        try:
            # Usiamo lxml.etree per la lettura e per il parsing futuro
            parser = etree.XMLParser(remove_blank_text=True)
            tree = etree.parse(nome_file, parser)
            root = tree.getroot()
            print(f"File XML '{nome_file}' esistente trovato. Aggiungo i nuovi risultati.")
        except etree.ParseError:
            print(f"Errore nella lettura del file XML esistente. Creo un nuovo file.")
            # Se il parsing fallisce, creiamo una nuova radice
            root = etree.Element("RiepilogoMisure")
    else:
        # Se il file non esiste, creiamo una nuova radice
        root = etree.Element("RiepilogoMisure")
    
    # Aggiorna l'attributo di data/tempo sulla radice, se presente
    root.set("data_aggiornamento", str(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")))

    misure_aggiunte = 0
    
# ==============================================================================
# FUNZIONI DI SALVATAGGIO E LOGICA originali in "Misurazione_micro.py"
# ==============================================================================

# ==============================================================================
# FUNZIONI DI SALVATAGGIO E LOGICA SEQUENZIALE
# ==============================================================================

def salva_risultati_xml(nuovi_dati, nome_file):
    """ 
    Salva o aggiorna i risultati in un file XML. 
    Aggiunge i dati forzando un ordinamento logico dei campi, 
    inclusa la corretta gestione delle CoordinateSolidali.
    """
    
    misure_aggiunte = 0 

    # ----------------------------------------------------
    # 1. LOGICA DI LETTURA E APPEND
    # ----------------------------------------------------
    if os.path.exists(nome_file):
        try:
            parser = etree.XMLParser(remove_blank_text=True)
            tree = etree.parse(nome_file, parser)
            root = tree.getroot()
            print(f"File XML '{nome_file}' esistente trovato. Aggiungo i nuovi risultati.")
        except etree.ParseError:
            print(f"Errore nella lettura del file XML esistente. Creo un nuovo file.")
            root = etree.Element("RiepilogoMisure")
    else:
        root = etree.Element("RiepilogoMisure")
    
    root.set("data_aggiornamento", str(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")))

    # Calcola l'ID di partenza (numero di misure già presenti)
    num_misure_esistenti = len(root.findall("Misura"))
    
    # ----------------------------------------------------
    # 2. LOGICA DI SCRITTURA E ORDINAMENTO
    # ----------------------------------------------------
    for misura_data in nuovi_dati:
        
        if misura_data.get("status") == "Skipped":
            continue
        
        # Genera ID progressivo
        current_id = num_misure_esistenti + misure_aggiunte + 1
        
        # Estraggo e Rimuovo il nome del file e l'ID se presente per usarli negli attributi
        filename = misura_data.pop('file_name', 'N/A')
        misura_data.pop('id_misura', None)
        
        # 1. Crea l'elemento Misura (radice per ogni immagine)
        misura_element = etree.SubElement(root, "Misura", 
                                          filename=filename,
                                          id_misura=str(current_id))
        
        # --- A. MISURE SCALA E TEMPORALI (Metadati) ---
        # DataMisurazione è l'unica che era N/A. La correggo con un valore sensato.
        etree.SubElement(misura_element, "DataMisurazione").text = misura_data.pop('DataMisurazione', datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        etree.SubElement(misura_element, "scale_factor").text = str(misura_data.pop('scale_factor', 'N/A'))
        
        # --- B. MISURE GEOMETRICHE (Risultati finali ordinati) ---
        geo_element = etree.SubElement(misura_element, "MisurazioniGeometriche")
        
        # Definiamo l'ordine desiderato delle misurazioni in micrometri (µm)
        misurazioni_um_order = [
            ('gradino_um', 'Gradino'),
            ('angolo_gradi', 'Angolo'),
            ('faceting_um', 'Faceting'),
            ('trincea_larghezza_um', 'TrinceaLarghezza'),
            ('micro_left_um', 'MicroTrenchSinistro'),
            ('micro_right_um', 'MicroTrenchDestro'),
        ]
        
        # Aggiunge prima le misurazioni in µm
        for key, tag in misurazioni_um_order:
            if key in misura_data:
                etree.SubElement(geo_element, tag).text = str(misura_data.pop(key))
                
        # Aggiunge i delta (P4) subito dopo i risultati principali
        if 'delta_x_prime_um' in misura_data:
             etree.SubElement(geo_element, "DeltaX_P4").text = str(misura_data.pop('delta_x_prime_um'))
        if 'delta_y_prime_um' in misura_data:
             etree.SubElement(geo_element, "DeltaY_P4").text = str(misura_data.pop('delta_y_prime_um'))

        # Aggiunge le misure in Pixel
        if 'trincea_larghezza_px' in misura_data:
             etree.SubElement(geo_element, "TrinceaLarghezzaPX").text = str(misura_data.pop('trincea_larghezza_px'))
        if 'micro_left_px' in misura_data:
             etree.SubElement(geo_element, "MicroTrenchSinistroPX").text = str(misura_data.pop('micro_left_px'))
        if 'micro_right_px' in misura_data:
             etree.SubElement(geo_element, "MicroTrenchDestroPX").text = str(misura_data.pop('micro_right_px'))


        # --- C. COORDINATE SOLIDALI (Raggruppamento e Ordinamento) ---
        coord_element = etree.SubElement(misura_element, "CoordinateSolidali")
        
        # Troviamo tutte le chiavi relative alle coordinate
        # Usiamo list() per creare una copia delle chiavi e iterare in sicurezza
        # Le chiavi hanno il formato: p#_x_prime_um, p#_y_prime_px, ecc.
        coord_keys = [k for k in list(misura_data.keys()) if '_prime_' in k]
        
        # Ordiniamo le chiavi per punto (P1, P2, P3...) e per unità (um prima di px)
        def get_coord_sort_key(key):
            # Esempio: 'p5_y_prime_um'
            parts = key.split('_') 
            punto = parts[0] # p5
            
            # Assicurati che punto[1:] sia un numero
            punto_num = int(punto[1:]) if punto[1:].isdigit() else float('inf')
            
            # parts[-1] è 'um' o 'px'
            is_um = 0 if parts[-1] == 'um' else 1
            # parts[1] è 'x' o 'y'
            is_x = 0 if parts[1] == 'x' else 1
            
            # Ordine: Punto (1..11), Unità (um prima di px), Asse (x prima di y)
            return (punto_num, is_um, is_x)

        chiavi_ordinate = sorted(coord_keys, key=get_coord_sort_key)
        
        # Dizionario temporaneo per raggruppare sotto il tag <P#></P#>
        punti_coord_gruppo = {}

        for key in chiavi_ordinate:
            # Estrai P# e il resto del nome della coordinata (X_UM, Y_PX, ecc.)
            try:
                # Esempio: 'p5_y_prime_um' -> ('P5', 'Y_UM')
                parts = key.split('_')
                punto_tag = parts[0].upper() # P5
                
                # Nome della coordinata nel tag XML
                coord_name = parts[1].upper() + '_' + parts[-1].upper() # Y_UM
                
            except IndexError:
                # Fallback, se la chiave non è nel formato atteso
                print(f"Attenzione: Chiave di coordinata '{key}' non parsabile correttamente.")
                continue

            if punto_tag not in punti_coord_gruppo:
                punti_coord_gruppo[punto_tag] = {}
            
            # Assegna il valore e rimuovi dal dizionario misura_data
            punti_coord_gruppo[punto_tag][coord_name] = str(misura_data.pop(key))
            # Non serve pop di nuovo, ma lo lascio per sicurezza su chiavi restanti non coordinate
            misura_data.pop(key, None) 

        # Aggiungi i nodi dei punti all'XML (ordinati da P1 a P11)
        def get_punto_sort_key(tag):
            return int(tag[1:]) if tag.startswith('P') and tag[1:].isdigit() else float('inf')
            
        for punto_tag in sorted(punti_coord_gruppo.keys(), key=get_punto_sort_key):
            p_node = etree.SubElement(coord_element, punto_tag)
            
            # Ordine forzato all'interno del punto
            coord_order = ['X_UM', 'Y_UM', 'X_PX', 'Y_PX']
            
            for name in coord_order:
                if name in punti_coord_gruppo[punto_tag]:
                    etree.SubElement(p_node, name).text = punti_coord_gruppo[punto_tag][name]


        # --- D. ANNOTAZIONI (Commenti e flag) ---
        annotazioni_element = etree.SubElement(misura_element, "Annotazioni")
        
        # Uso pop() con fallback per sicurezza
        etree.SubElement(annotazioni_element, "PolimeroSuParete").text = misura_data.pop('polimero_su_parete', 'N/A')
        etree.SubElement(annotazioni_element, "RideposizioneFondo").text = misura_data.pop('rideposizione_fondo', 'N/A')
        etree.SubElement(annotazioni_element, "Note").text = misura_data.pop('note', 'Nessuna nota aggiunta.')
        
        misure_aggiunte += 1

    # ----------------------------------------------------
    # 3. SALVATAGGIO FINALE 
    # ----------------------------------------------------
    tree = etree.ElementTree(root)
    tree.write(nome_file, pretty_print=True, xml_declaration=True, encoding='UTF-8')
    
    return misure_aggiunte

# ----------------------------------------------------

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
                        
                        #cv2.destroyAllWindows()
                        redraw_image()
                        
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

                            cv2.destroyAllWindows()
                            
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
