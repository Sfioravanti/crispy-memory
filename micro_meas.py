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
parallel_lines_offsets_px = [20, 40, 60]
active_line_index = 0 # Indice della retta selezionata (0, 1, 2)
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
            
            # Calcolo del Faceting (distanza proiettata P3 - P5)
            p3_proj = project_point_onto_line(p3, p1, p2)
            
            # Trasformazione nel sistema solidale
            angle_p1p2 = get_p1p2_angle(p1, p2)
            p3_x_prime, _ = transform_point(p3_proj, p1, angle_p1p2)
            p5_x_prime, _ = transform_point(p5, p1, angle_p1p2)

            faceting_px = abs(p3_x_prime - p5_x_prime)
            faceting_unit_val = faceting_px * PIXEL_TO_UNIT_FACTOR

            risultato_immagine.update({
                'gradino_um': gradino_unit_val,
                'angolo_gradi': angolo_gradi_val,
                'faceting_um': faceting_unit_val,
                'delta_x_prime_px': delta_x_prime_px,
                'delta_y_prime_px': delta_y_prime_px
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
            p6, p7 = punti_selezionati[5], punti_selezionati[6]
            # La larghezza della trincea è la distanza euclidea tra P6 e P7 (entrambi proiettati)
            # NOTA: p1 e p2 devono essere disponibili qui (vengono definiti nella Fase 0 e devono essere accessibili in locale o globale)
            trincea_unit_val, trincea_px_val = microtrenching_calc(p6, p7, p1, p2, PIXEL_TO_UNIT_FACTOR) 
            
            risultato_immagine.update({
                'trincea_larghezza_um': trincea_unit_val,
                'trincea_larghezza_px': trincea_px_val
            })
            
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
                
        redraw_image()
        

    # Fase 2: Micro Trench Sinistro (P8, P9)
    risultati_fase_2_stampati = False
    while fase_corrente == 2:
        # LOGICA DI CALCOLO
        if len(punti_selezionati) == 9 and not risultati_fase_2_stampati:
            p8, p9 = punti_selezionati[7], punti_selezionati[8]
            # Il micro trenching viene misurato come differenza di gradino (Y')
            micro_left_unit_val, micro_left_px_val = microtrenching_calc(p8, p9, p1, p2, PIXEL_TO_UNIT_FACTOR) 
            
            risultato_immagine.update({
                'micro_left_um': micro_left_unit_val,
                'micro_left_px': micro_left_px_val
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
        redraw_image() 
        

    # Fase 3: Micro Trench Destro (P10, P11)
    risultati_fase_3_stampati = False
    # Recupera P1 e P2 (Assumendo siano stati definiti in Fase 0)
    p1 = punti_selezionati[0]
    p2 = punti_selezionati[1]
    # Calcolo preliminare Quota Y' di P9 (Micro Left Lowest Point)
    # Calcola e salva la quota assoluta del punto P9 (indice 8) PRIMA della Fase 3
    if len(punti_selezionati) > 8 and 'y_prime_p9_um' not in risultato_immagine:
        p9 = punti_selezionati[8]
        angle_p1p2 = get_p1p2_angle(p1, p2)
        _, p9_y_prime = transform_point(p9, p1, angle_p1p2)
        y_prime_p9_um = p9_y_prime * PIXEL_TO_UNIT_FACTOR
        risultato_immagine.update({'y_prime_p9_um': y_prime_p9_um})
    while fase_corrente == 3:
        # LOGICA DI CALCOLO
        if len(punti_selezionati) == 11 and not risultati_fase_3_stampati:
            p10, p11 = punti_selezionati[9], punti_selezionati[10]
            # Il micro trenching viene misurato come differenza di gradino (Y')
            micro_right_unit_val, micro_right_px_val = microtrenching_calc(p10, p11, p1, p2, PIXEL_TO_UNIT_FACTOR) 
            # Calcola la Quota Y' di P11 (Micro Right Lowest Point)
            _, p11_y_prime = transform_point(p11, p1, angle_p1p2)
            y_prime_p11_um = p11_y_prime * PIXEL_TO_UNIT_FACTOR
            risultato_immagine.update({
                'micro_right_um': micro_right_unit_val,
                'micro_right_px': micro_right_px_val,
                'y_prime_p11_um': y_prime_p11_um # NUOVO VALORE SALVATO: Quota Y' del punto più basso Destro
            })
            print(f"Risultato Micro Trench Destro salvato. Gradino: {micro_right_unit_val:.4f} um.")
            print("Premere 'c' per il calcolo della differenza destra-sinistra?") # <--- NUOVO PROMPT
            risultati_fase_3_stampati = True
        
        # GESTIONE INPUT
        k = cv2.waitKey(50) & 0xFF
        
        if k == ord('c'):
            if len(punti_selezionati) == 11:
                if risultati_fase_3_stampati:
                    fase_corrente = 4 # Passa alla NUOVA Fase 4
                    global active_line_index
                    active_line_index = 0 # Inizializza la retta attiva alla prima
                    print("\nPassaggio alla Fase 4: Allineamento Rette e Calcolo Differenza Micro Trenches.")
                    redraw_image()
                else:
                    print("Calcoli ancora non completati.")
            else:
                print("Devi prima selezionare 2 punti (P10-P11).")
        
        # ... (restanti controlli di input 'q', 's', 'x', e logica di spostamento punto)
        # Assicurati che qui sia presente la logica per lo spostamento del punto (i, k, j, l)
        redraw_image() 
        
    # ------------------------------------------------------------------
    # B. NUOVA FASE 4: Differenza e Allineamento
    # ------------------------------------------------------------------
    # Fase 4: Differenza Micro Trenches (Delta) e Allineamento
    risultati_fase_4_stampati = False
    diff_microtrench_calc = None
    LINE_MOVE_STEP_PX = 1
    while fase_corrente == 4:
        # GESTIONE INPUT
        k = cv2.waitKey(50) & 0xFF
        
        # 1. Selezione Retta (Tasti 1, 2, 3)
        if k == ord('1'):
            active_line_index = 0
            print("Retta 1 selezionata.")
            redraw_image()
        elif k == ord('2'):
            active_line_index = 1
            print("Retta 2 selezionata.")
            redraw_image()
        elif k == ord('3'):
            active_line_index = 2
            print("Retta 3 selezionata.")
            redraw_image()
        
        # 2. Spostamento Retta (Tasti 'i' e 'k' come richiesto)
        elif k == ord('i'):
            parallel_lines_offsets_px[active_line_index] -= LINE_MOVE_STEP_PX
            print(f"Retta {active_line_index+1} spostata verso l'alto. Offset: {parallel_lines_offsets_px[active_line_index]} px")
            redraw_image()
        elif k == ord('k'):
            parallel_lines_offsets_px[active_line_index] += LINE_MOVE_STEP_PX
            print(f"Retta {active_line_index+1} spostata verso il basso. Offset: {parallel_lines_offsets_px[active_line_index]} px")
            redraw_image()
            
        # 3. Calcolo e Passaggio Finale ('c')
        elif k == ord('c'):
            if not risultati_fase_4_stampati:
                # Calcola la differenza (Delta) e usa la retta 2 (active_line_index=1)
                
                # A. Calcolo del Delta Micro Trench
                delta_microtrench_um = abs(risultato_immagine['y_prime_p9_um'] - risultato_immagine['y_prime_p11_um'])
                risultato_immagine['delta_microtrench_um'] = delta_microtrench_um
                
                # B. Calcolo Allineamento (Gradino A-B)
                # Assumiamo p1, p2, p9, p11 sono già selezionati
                p9_coord = punti_selezionati[8] 
                p11_coord = punti_selezionati[10]
                
                # Proietta P9 e P11 sulla retta 2 (active_line_index=1)
                offset_px = parallel_lines_offsets_px[1] # Retta 2 (indice 1)
                
                p9_line2 = project_point_onto_parallel_line(p9_coord, p1, p2, offset_px)
                p11_line2 = project_point_onto_parallel_line(p11_coord, p1, p2, offset_px)
                
                # Calcola la distanza verticale tra la proiezione del punto e la retta P1-P2
                # Qui usiamo la distanza verticale (Y) in pixel, poi convertiamo
                
                # Calcolo della Y' sul sistema di riferimento P1-P2
                angle_p1p2 = get_p1p2_angle(p1, p2)
                
                # Trasforma p9 e p11 nel sistema solidale X', Y' (base è P1)
                _, p9_y_prime = transform_point(p9_coord, p1, angle_p1p2)
                _, p11_y_prime = transform_point(p11_coord, p1, angle_p1p2)
                
                # Calcola le coordinate sulla retta parallela
                p9_y_line2_prime = p9_y_prime - offset_px # La retta parallela è spostata di 'offset_px' in Y'
                p11_y_line2_prime = p11_y_prime - offset_px
                
                # La misura dell'allineamento è la distanza (verticale Y') tra i punti proiettati
                # sulla retta parallela. Se sono sulla stessa linea, Y' sarà 0.
                allineamento_px = abs(p9_y_line2_prime - p11_y_line2_prime)
                allineamento_um = allineamento_px * PIXEL_TO_UNIT_FACTOR
                
                risultato_immagine['allineamento_ab_um'] = allineamento_um
                
                print(f"\nRisultato Fase 4 Calcolato.")
                print(f"Delta Micro Trench (P9/P11): {delta_microtrench_um:.4f} um.")
                print(f"Allineamento (Gradino P9-P11 sulla Retta 2): {allineamento_um:.4f} um.")
                
                risultati_fase_4_stampati = True
                print("Premi 's' per salvare e uscire, 'q' per saltare.")
                
            else:
                # Se i risultati sono già stampati, il tasto 'c' fa passare al salvataggio (o si aspetta 's')
                print("Risultati già calcolati. Premi 's' per salvare e uscire.")

        # 4. Salvataggio e Skip ('s', 'q')
        elif k == ord('s'): 
            if risultati_fase_4_stampati:
                risultato_immagine['status'] = 'Completed'
            else:
                risultato_immagine['status'] = 'Incomplete'
            return risultato_immagine
        elif k == ord('q'): 
            risultato_immagine['status'] = 'Skipped'
            return risultato_immagine
        
        # 5. Annulla e Torna Indietro ('x')
        elif k == ord('x') or k == 27: 
            fase_corrente = 3
            risultati_fase_4_stampati = False
            risultato_immagine.pop('delta_microtrench_um', None)
            risultato_immagine.pop('allineamento_ab_um', None)
            print("Tornato alla Fase 3.")
            redraw_image()
            
        redraw_image() # Aggiorna la visualizzazione

    # Rimuovi la finestra OpenCV
    cv2.destroyAllWindows()
    return risultato_immagine

# ==============================================================================
# FUNZIONI DI SUPPORTO (Calcoli Geometrici)
# ==============================================================================

def get_p1p2_angle(p1, p2):
    """Calcola l'angolo (in radianti) della retta P1-P2 rispetto all'asse X positivo."""
    x1, y1 = p1
    x2, y2 = p2
    # Utilizza atan2 per ottenere l'angolo corretto su 360 gradi
    angle = math.atan2(y2 - y1, x2 - x1)
    return angle

def rotate_point(point, angle):
    """Ruota un punto attorno all'origine (0,0)."""
    x, y = point
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)
    x_rot = x * cos_a + y * sin_a
    y_rot = -x * sin_a + y * cos_a
    return (x_rot, y_rot)

def transform_point(point, origin, angle):
    """
    Trasforma un punto nel sistema di riferimento locale (X', Y') definito da:
    - Origine (0,0) spostata a 'origin' (P1).
    - Asse X' allineato con 'angle' (angolo P1-P2).
    
    L'asse Y' è perpendicolare a X'.
    """
    px, py = point
    ox, oy = origin
    
    # 1. Traslazione: sposta l'origine a P1
    x_trans = px - ox
    y_trans = py - oy
    
    # 2. Rotazione: ruota il sistema di assi
    # L'angolo di rotazione è -angle perché ruotiamo i punti per allinearli al sistema
    # L'asse X' corrisponde a P1->P2. Per portare i punti sul sistema X', Y' (dove X' è orizzontale)
    # dobbiamo ruotare del negativo dell'angolo della retta.
    
    # NOTA: Per un sistema X' allineato con P1-P2, l'angolo di rotazione è -angle.
    angle_rot = -angle 
    
    # La funzione rotate_point ha già l'asse Y' invertito (standard OpenCV/Image)
    x_prime, y_prime = rotate_point((x_trans, y_trans), angle_rot)
    
    return x_prime, y_prime

def project_point_onto_line(p, p1, p2):
    """Proietta un punto 'p' sulla retta passante per 'p1' e 'p2'."""
    p = np.array(p)
    p1 = np.array(p1)
    p2 = np.array(p2)

    # Vettore della retta P1->P2
    line_vec = p2 - p1
    
    # Vettore P1->P
    p1_to_p = p - p1
    
    # Proiezione scalare (t)
    # t = (P1->P . P1->P2) / |P1->P2|^2
    if np.dot(line_vec, line_vec) == 0:
        # P1 e P2 coincidono, ritorna P1
        return p1
        
    t = np.dot(p1_to_p, line_vec) / np.dot(line_vec, line_vec)
    
    # Punto proiettato
    p_proj = p1 + t * line_vec
    return p_proj

def project_point_onto_parallel_line(p, p1_ref, p2_ref, offset_px):
    """
    Proietta un punto 'p' sulla retta parallela alla retta P1_ref-P2_ref
    spostata verticalmente (Y') di 'offset_px'.
    """
    p = np.array(p)
    p1_ref = np.array(p1_ref)
    p2_ref = np.array(p2_ref)
    
    # 1. Calcola l'angolo della retta di riferimento P1-P2
    angle = get_p1p2_angle(p1_ref.tolist(), p2_ref.tolist())
    
    # 2. Trasforma il punto 'p' nel sistema di riferimento X', Y' (orig: P1_ref, X' allineato a P1->P2)
    p_x_prime, p_y_prime = transform_point(p.tolist(), p1_ref.tolist(), angle)
    
    # 3. Proietta in X' (cioè ignoriamo lo scostamento in X')
    # Il punto sulla retta parallela, nel sistema X', Y', ha coordinate:
    # X_proj' = p_x_prime (Non proiettiamo in X', assumiamo il punto è proiettato solo su Y')
    # Y_line' = offset_px
    
    # Vogliamo proiettare sulla retta definita da Y' = offset_px
    # Poiché VOGLIAMO MANTENERE la proiezione X' sulla retta, e solo allineare Y'
    # il punto proiettato in X', Y' è (p_x_prime, offset_px).
    
    # Però, per l'allineamento (Fase 4), vogliamo il punto 'p' proiettato in X' sulla retta P1-P2
    # e poi spostato per stare sulla retta parallela.
    
    # Proiettiamo prima su P1-P2.
    p_proj_ref = project_point_onto_line(p, p1_ref, p2_ref)
    
    # 4. Spostiamo la proiezione lungo l'asse Y' (perpendicolare a P1-P2)
    # L'angolo perpendicolare (normale) è angle + pi/2
    angle_norm = angle + math.pi / 2
    
    # Spostamento lungo la normale
    dx = offset_px * math.cos(angle_norm)
    dy = offset_px * math.sin(angle_norm)
    
    # Punto finale sulla retta parallela
    p_proj_parallel = p_proj_ref + np.array([dx, dy])
    
    return p_proj_parallel

# ==============================================================================
# FUNZIONI DI CALCOLO E MISURA (Logica di Business)
# ==============================================================================

def gradino_angolo_faceting(punti, scale_factor):
    """
    Calcola il gradino (Y' di P4), l'angolo della parete (P3-P4), e il faceting (P3'-P5).
    
    Input:
        punti: Lista di 5 tuple di punti [(P1), (P2), (P3), (P4), (P5)]
        scale_factor: Fattore di scala (µm/px)
        
    Output:
        gradino_um, angolo_gradi, faceting_um, delta_x_prime_px, delta_y_prime_px
    """
    if len(punti) < 5:
        return None, None, None, None, None
        
    p1, p2, p3, p4, p5 = punti[0:5]
    
    # 1. Definizione del Sistema di Riferimento
    angle_p1p2 = get_p1p2_angle(p1, p2)
    
    # 2. Calcolo Gradino (Y' di P4)
    # Trasformiamo P4 nel sistema X', Y' (base P1, X' allineato P1-P2)
    p4_x_prime, p4_y_prime = transform_point(p4, p1, angle_p1p2)
    gradino_um = abs(p4_y_prime) * scale_factor # Gradino è la distanza perpendicolare (Y') da P1-P2
    
    # 3. Calcolo Angolo della Parete (P3-P4)
    p3_x_prime, p3_y_prime = transform_point(p3, p1, angle_p1p2)
    
    # La retta di interesse (parete) è P3-P4.
    # Calcola l'angolo della retta P3-P4 rispetto all'asse X' (P1-P2).
    # L'angolo viene calcolato nel sistema X', Y'.
    delta_x_prime = p4_x_prime - p3_x_prime
    delta_y_prime = p4_y_prime - p3_y_prime
    
    if abs(delta_x_prime) < 1e-6:
        # Se la retta è quasi verticale nel sistema X', Y'
        angolo_rad = math.pi / 2 # 90 gradi
    else:
        # Calcola l'angolo assoluto della retta P3-P4 nel sistema X', Y'
        angolo_parete_assoluto_rad = math.atan2(delta_y_prime, delta_x_prime)
        
        # L'angolo rispetto all'orizzontale X' (il Top Surface)
        # Poiché stiamo misurando una parete, vogliamo l'angolo rispetto alla verticale (90 gradi).
        # Calcola l'angolo (acuto) tra il vettore P3->P4 e l'asse X'
        angolo_rispetto_x_prime = abs(angolo_parete_assoluto_rad)
        
        # Angolo rispetto all'asse Y' (la parete)
        angolo_rad = math.pi / 2 - angolo_rispetto_x_prime
        
    angolo_gradi = math.degrees(angolo_rad)
    
    # 4. Calcolo Faceting (P3' - P5)
    # Nota: Il Faceting (P3'-P5) è calcolato nel loop principale `misura_immagine_interattiva`
    # in quanto richiede un punto proiettato P3' e il punto P5. Qui ritorniamo None per semplicità,
    # ma i calcoli nel loop principale si basano sulle coordinate trasformate X'.
    
    return gradino_um, angolo_gradi, None, delta_x_prime, delta_y_prime # L'ultima coppia serve per debug/informazione

def microtrenching_calc(p_a, p_b, p1_ref, p2_ref, scale_factor):
    """
    Calcola la distanza tra due punti A e B, proiettandoli sulla retta di riferimento P1-P2.
    
    Input:
        p_a, p_b: I due punti da misurare (P6/P7, P8/P9, o P10/P11)
        p1_ref, p2_ref: I punti della retta di riferimento (P1, P2)
        scale_factor: Fattore di scala (µm/px)
        
    Output:
        distanza_unit_val, distanza_px_val
    """
    if p1_ref is None or p2_ref is None:
        return 0.0, 0.0
        
    # 1. Calcola l'angolo della retta di riferimento
    angle_p1p2 = get_p1p2_angle(p1_ref, p2_ref)
    
    # 2. Trasforma i punti nel sistema X', Y' (base P1, X' allineato P1-P2)
    p_a_x_prime, p_a_y_prime = transform_point(p_a, p1_ref, angle_p1p2)
    p_b_x_prime, p_b_y_prime = transform_point(p_b, p1_ref, angle_p1p2)
    
    # Per la Larghezza Trincea (P6-P7): Si misura la distanza in X' (longitudinale)
    if (p_a, p_b) == (punti_selezionati[5], punti_selezionati[6]):
        distanza_px_val = abs(p_b_x_prime - p_a_x_prime)
    
    # Per Micro Trench Gradino (P8-P9 e P10-P11): Si misura la distanza in Y' (verticale/gradino)
    else:
        # Questa è la logica per il gradino verticale (Y')
        distanza_px_val = abs(p_b_y_prime - p_a_y_prime)
        
    distanza_unit_val = distanza_px_val * scale_factor
    
    return distanza_unit_val, distanza_px_val


# ==============================================================================
# FUNZIONI DI VISUALIZZAZIONE E INTERAZIONE (OpenCV)
# ==============================================================================

def draw_reference_line_mobile(img, ref_params, p_new, p1, p2, color=(0, 255, 0), thickness=1):
    """
    Disegna la retta mobile che passa per p_new ed è parallela alla retta P1-P2.
    Questa funzione viene utilizzata nelle fasi 1, 2, 3 per aiutare l'utente 
    a selezionare i punti sulla "stessa altezza" o "stessa X" di P_new.
    """
    if ref_params is None:
        return img
        
    m, q = ref_params
    H, W = img.shape[:2]
    
    if m is not None: # Retta inclinata (y = mx + q)
        # 1. Trova l'offset di p_new rispetto a P1-P2 nel sistema X', Y' (la quota Y' di p_new)
        angle_p1p2 = get_p1p2_angle(p1, p2)
        _, p_new_y_prime = transform_point(p_new, p1, angle_p1p2)
        
        # 2. Calcola i parametri (m, q_mobile) della retta parallela passante per p_new
        # La retta mobile è definita da y' = p_new_y_prime (nel sistema X', Y')
        # Troviamo q_mobile in coordinate (x, y)
        
        # Definisci il punto di riferimento sulla retta mobile (P1_mobile)
        p1_mobile_x_prime = 0
        p1_mobile_y_prime = p_new_y_prime
        
        # Trasforma P1_mobile indietro in (x, y)
        p1_mobile = transform_back_point((p1_mobile_x_prime, p1_mobile_y_prime), p1, -angle_p1p2)
        
        # Calcola q_mobile dalla retta mobile: y = m*x + q_mobile
        q_mobile = p1_mobile[1] - m * p1_mobile[0]
        
        # Punti di disegno: intersezioni con i bordi x=0 e x=W
        x_start = 0
        y_start = int(m * x_start + q_mobile)
        
        x_end = W
        y_end = int(m * x_end + q_mobile)
        
        # Disegna
        cv2.line(img, (x_start, y_start), (x_end, y_end), color, thickness)
        
    else: # Retta verticale (x = x1)
        # La retta mobile è verticale e passa per x_mobile (x di p_new)
        x_mobile = p_new[0]
        
        # Disegna
        cv2.line(img, (x_mobile, 0), (x_mobile, H), color, thickness)
        
    return img
    
def draw_parallel_line(img, p1_ref, p2_ref, offset_px, color, thickness=1):
    """
    Disegna una retta parallela alla retta P1_ref-P2_ref, spostata di offset_px.
    """
    H, W = img.shape[:2]
    angle = get_p1p2_angle(p1_ref, p2_ref)
    
    # 1. Trova due punti sulla retta parallela
    # Punto di origine traslato (P1)
    p1 = np.array(p1_ref)
    
    # Vettore perpendicolare (normale)
    angle_norm = angle + math.pi / 2
    dx = offset_px * math.cos(angle_norm)
    dy = offset_px * math.sin(angle_norm)
    offset_vec = np.array([dx, dy])
    
    # Punti sulla retta parallela
    p_start_on_line = p1 + offset_vec
    p_end_on_line = np.array(p2_ref) + offset_vec
    
    # 2. Calcola i parametri (m, q) della retta parallela
    x1, y1 = p_start_on_line.astype(int)
    x2, y2 = p_end_on_line.astype(int)
    
    if abs(x2 - x1) > 1e-6:
        m = (y2 - y1) / (x2 - x1)
        q = y1 - m * x1
        
        # Punti di disegno: intersezioni con i bordi x=0 e x=W
        x_start = 0
        y_start = int(m * x_start + q)
        
        x_end = W
        y_end = int(m * x_end + q)
        
        # Disegna
        cv2.line(img, (x_start, y_start), (x_end, y_end), color, thickness)
        
    else: # Retta verticale
        x_mobile = x1 # o x2
        cv2.line(img, (x_mobile, 0), (x_mobile, H), color, thickness)
        
    return 

def transform_back_point(point_prime, origin, angle_rot):
    """
    Trasforma un punto dal sistema X', Y' (ruotato) al sistema originale (x, y).
    - angle_rot è l'angolo di rotazione applicato in `transform_point` (cioè -angle_p1p2)
    
    Per fare l'inversa:
    1. Ruota indietro con -angle_rot
    2. Trasla indietro aggiungendo l'origine.
    """
    x_prime, y_prime = point_prime
    ox, oy = origin
    
    # 1. Ruota indietro (-angle_rot)
    # L'operazione inversa della rotazione (x', y') = rotate_point((x_trans, y_trans), angle_rot)
    # è (x_trans, y_trans) = rotate_point((x_prime, y_prime), -angle_rot)
    x_trans, y_trans = rotate_point((x_prime, y_prime), -angle_rot)
    
    # 2. Trasla indietro
    x = x_trans + ox
    y = y_trans + oy
    
    return (int(round(x)), int(round(y)))

def redraw_image():
    """
    Funzione principale per ridisegnare l'immagine con tutti i punti, linee e prompt.
    """
    global immagine_originale, immagine_modificabile, punti_selezionati, window_name
    global reference_line_params, fase_corrente
    global fase_corrente, visual_scale_factor, current_pan_offset, active_line_index
    global parallel_lines_offsets_px
    
    # 1. Controllo iniziale: Dobbiamo avere l'immagine originale
    if immagine_originale is None: 
        return

    # L'immagine modificabile è una copia dell'originale
    immagine_modificabile = immagine_originale.copy()
    
    # Calcola H e W all'inizio della funzione
    H, W = immagine_modificabile.shape[:2] 
    
    # -----------------------------------------------------------
    # FIX: Gestione p1 e p2 per evitare UnboundLocalError
    # -----------------------------------------------------------
    p1, p2 = None, None
    if len(punti_selezionati) >= 2:
        p1 = punti_selezionati[0]
        p2 = punti_selezionati[1]
    # -----------------------------------------------------------
    
    # 1. Disegna la retta di riferimento P1-P2 (Top Surface)
    if p1 is not None and p2 is not None:
        # Usiamo direttamente le coordinate originali
        cv2.line(immagine_modificabile, p1 , p2 , (255, 0, 0), 1) # Blu
        
        # Disegna la linea di gradino P4
        if len(punti_selezionati) >= 4:
            p4 = punti_selezionati[3]
            p4_proj = project_point_onto_line(p4 , p1 , p2 )
            cv2.line(immagine_modificabile, p4 , p4_proj .tolist(), (0, 0, 255), 1) # Rosso (Gradino) 
            
        # Disegna la linea di faceting P3-P5 (asse X')
        if len(punti_selezionati) >= 5:
            p3 = punti_selezionati[2]
            p5 = punti_selezionati[4]
            # Disegna P3-P3' (non necessario ma utile per visualizzare)
            p3_proj = project_point_onto_line(p3 , p1 , p2 )
            cv2.line(immagine_modificabile, p3 , p3_proj .tolist(), (0, 165, 255), 1) # Arancione
            
            # Disegna P3'-P5 (la distanza di faceting)
            cv2.line(immagine_modificabile, p3_proj .tolist(), p5 , (0, 165, 255), 1) # Arancione (Faceting)

        # Disegna le linee di Micro Trenching P8-P9 e P10-P11
        if len(punti_selezionati) >= 9:
            p8, p9 = punti_selezionati[7], punti_selezionati[8]
            p8_proj = project_point_onto_line(p8, p1, p2)
            p9_proj = project_point_onto_line(p9, p1, p2)
            # Linea di Micro-Trench Sinistro (P8-P9)
            cv2.line(immagine_modificabile, p8, p9, (255, 255, 0), 1) # Ciano
            # Linee di proiezione (P8-P8' e P9-P9')
            cv2.line(immagine_modificabile, p8, p8_proj.tolist(), (255, 255, 0), 1) 
            cv2.line(immagine_modificabile, p9, p9_proj.tolist(), (255, 255, 0), 1) 

        if len(punti_selezionati) >= 11:
            p10, p11 = punti_selezionati[9], punti_selezionati[10]
            p10_proj = project_point_onto_line(p10, p1, p2)
            p11_proj = project_point_onto_line(p11, p1, p2)
            # Linea di Micro-Trench Destro (P10-P11)
            cv2.line(immagine_modificabile, p10, p11, (255, 0, 255), 1) # Magenta
            # Linee di proiezione (P10-P10' e P11-P11')
            cv2.line(immagine_modificabile, p10, p10_proj.tolist(), (255, 0, 255), 1) 
            cv2.line(immagine_modificabile, p11, p11_proj.tolist(), (255, 0, 255), 1) 

            
    # 2. Disegna la retta della trincea mobile (se in Fase 1, 2 o 3)
    # L'ultimo punto selezionato definisce la retta mobile
    if fase_corrente >= 1 and reference_line_params is not None and len(punti_selezionati) >= 3:
        p_new_index = len(punti_selezionati) - 1
        p_new = punti_selezionati[p_new_index]
        draw_reference_line_mobile(immagine_modificabile, reference_line_params, p_new, p1, p2, (0, 255, 0), 1)
            
    # 3. Logica Disegno FASE 4 (Allineamento Rette)
    if fase_corrente == 4 and p1 is not None and p2 is not None:
        
        # Disegna le 3 rette parallele
        colors = [(0, 255, 255), (0, 255, 0), (255, 0, 255)] # Ciano, Verde, Magenta
        for i, offset in enumerate(parallel_lines_offsets_px):
            color = colors[i]
            thickness = 2 if i == active_line_index else 1
            draw_parallel_line(immagine_modificabile, p1, p2, offset, color, thickness)
            
            # Etichetta la retta attiva
            if i == active_line_index:
                angle_p1p2 = get_p1p2_angle(p1, p2)
                
                # Trova un punto iniziale sulla retta parallela per l'etichetta
                p_start_on_line = np.array(p1) + np.array(transform_back_point((0, offset), (0, 0), -angle_p1p2))
                
                label_pos = (int(p_start_on_line[0] + 10), int(p_start_on_line[1] - 5))
                cv2.putText(immagine_modificabile, f"Retta {i+1} (Offset: {offset}px)", label_pos, 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, thickness)

        # Stampa il prompt per la Fase 4
        cv2.putText(immagine_modificabile, "Fase 4: Rette Parallele. Premi 1/2/3 per selezionare la retta. Tasti 'i' (su), 'k' (giu) per spostare. Premere 'c' per il calcolo finale.", (10, H - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # 4. Disegna tutti i punti selezionati (incluso il nome del punto)
    for i, p in enumerate(punti_selezionati):
        label = f"P{i+1}"
        color = (255, 255, 255)
        
        # Colori specifici per fasi/punti chiave
        if i in [0, 1]: # P1, P2 (Rif. Top Surface)
            color = (255, 0, 0) # Blu
        elif i == 3: # P4 (Gradino/Parete)
            color = (0, 0, 255) # Rosso
        elif i == 4: # P5 (Faceting)
            color = (0, 165, 255) # Arancione
        elif i in [5, 6]: # P6, P7 (Trincea)
            color = (0, 255, 0) # Verde
        elif i in [7, 8]: # P8, P9 (Micro Sinistro)
            color = (255, 255, 0) # Ciano
        elif i in [9, 10]: # P10, P11 (Micro Destro)
            color = (255, 0, 255) # Magenta
        
        cv2.circle(immagine_modificabile, p, 3, color, -1)
        cv2.putText(immagine_modificabile, label, (p[0] + 5, p[1] + 5), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    # 5. Visualizzazione finale
    cv2.imshow(window_name, immagine_modificabile)


def on_mouse(event, x, y, flags, param):
    """
    Gestisce gli eventi del mouse (click e pan).
    """
    global punti_selezionati, window_name, fase_corrente
    global reference_line_params, is_panning, last_mouse_pos
    
    # Gestione del Panning (Spostamento Immagine)
    # Questa logica non è implementata nel codice fornito, ma la lascio come placeholder.
    # Attualmente, solo lo zoom (scroll) è supportato.
    
    # 1. SELEZIONE PUNTI (Click Sinistro)
    if event == cv2.EVENT_LBUTTONDOWN:
        p_new = (x, y)
        
        # Aggiunge il punto solo se la fase lo permette
        if fase_corrente == 0 and len(punti_selezionati) < 5:
            punti_selezionati.append(p_new)
            print(f"P{len(punti_selezionati)} selezionato: ({x}, {y})")
            
            # Se P1 e P2 sono selezionati, calcola i parametri della retta di riferimento
            if len(punti_selezionati) == 2:
                p1, p2 = punti_selezionati[0], punti_selezionati[1]
                x1, y1 = p1
                x2, y2 = p2
                
                if abs(x2 - x1) > 1e-6:
                    m = (y2 - y1) / (x2 - x1)
                    q = y1 - m * x1
                    reference_line_params = (m, q)
                else:
                    # Retta verticale (pendenza infinita)
                    reference_line_params = (None, x1)
                print("Retta di riferimento P1-P2 calcolata.")
            
            # Se P5 è selezionato, lo proietta sulla retta P1-P2
            if len(punti_selezionati) == 5:
                # Sovrascrive P5 con la sua proiezione su P1-P2
                p1, p2 = punti_selezionati[0], punti_selezionati[1]
                p5_originale = p_new
                p5_proiettato_np = project_point_onto_line(p5_originale, p1, p2)
                punti_selezionati[4] = tuple(p5_proiettato_np.tolist())
                print(f"P5 proiettato sulla retta P1-P2: {punti_selezionati[4]}")
                
        elif fase_corrente == 1 and len(punti_selezionati) < 7:
            punti_selezionati.append(p_new)
            print(f"P{len(punti_selezionati)} selezionato: ({x}, {y})")
            
        elif fase_corrente == 2 and len(punti_selezionati) < 9:
            punti_selezionati.append(p_new)
            print(f"P{len(punti_selezionati)} selezionato: ({x}, {y})")
            
        elif fase_corrente == 3 and len(punti_selezionati) < 11:
            punti_selezionati.append(p_new)
            print(f"P{len(punti_selezionati)} selezionato: ({x}, {y})")
            
        redraw_image()
    
# ==============================================================================
# FUNZIONI DI CARICAMENTO E SALVATAGGIO
# ==============================================================================

def carica_immagine_tiff_con_scala(filepath):
    """
    Carica un'immagine TIFF e tenta di estrarre il fattore di scala
    dall'XML (se presente) o lo imposta a 1.0.
    """
    # 1. Carica l'immagine
    try:
        # Uso tifffile per caricare immagini TIFF a 16 bit
        img_array = tifffile.imread(filepath)
    except Exception as e:
        print(f"Errore nel caricamento del file {filepath}: {e}")
        return None, 1.0

    # 2. Conversione e normalizzazione (Se necessario)
    # Se l'immagine è in scala di grigi (2D), converti in BGR per OpenCV
    if len(img_array.shape) == 2:
        # Normalizza a 8 bit se è a 16 bit
        if img_array.dtype == np.uint16:
            img_array = (img_array / 256).astype(np.uint8)
        img = cv2.cvtColor(img_array, cv2.COLOR_GRAY2BGR)
    elif img_array.dtype == np.uint16:
        # Se è BGR/RGB a 16 bit, normalizza a 8 bit
        img = (img_array / 256).astype(np.uint8)
    else:
        img = img_array.astype(np.uint8)
        
    # 3. Estrazione del fattore di scala da XML
    xml_filepath = os.path.splitext(filepath)[0] + '.xml'
    scale_factor = 1.0 # Default a 1.0 µm/px
    
    if os.path.exists(xml_filepath):
        try:
            tree = ET.parse(xml_filepath)
            root = tree.getroot()
            
            # Cerca il tag PixelSize (tipico formato Zeiss/Olympus)
            pixel_size_tag = root.find('.//PixelSizeMicrons')
            if pixel_size_tag is not None:
                scale_factor = float(pixel_size_tag.text)
                print(f"Fattore di scala letto da XML: {scale_factor:.4f} µm/px")
                return img, scale_factor
            
            # Tenta di trovare altri formati (es. "DistancePerPixel")
            for tag in root.iter():
                if 'DistancePerPixel' in tag.tag and tag.text is not None:
                    # Estrai il valore dal tag
                    scale_factor = float(tag.text)
                    print(f"Fattore di scala letto da XML (DistancePerPixel): {scale_factor:.4f} µm/px")
                    return img, scale_factor

        except Exception as e:
            print(f"Errore nell'analisi del file XML {xml_filepath}: {e}")
            pass # Continua con il default

    print("Fattore di scala non trovato in XML. Impostato a 1.0 µm/px (verificare).")
    return img, scale_factor

def salva_risultati_xml(risultati_sessione, nome_file_output):
    """
    Salva i risultati di una sessione di misurazione in un file XML.
    """
    try:
        # Carica il file esistente o crea la radice
        if os.path.exists(nome_file_output):
            parser = etree.XMLParser(remove_blank_text=True)
            tree = etree.parse(nome_file_output, parser)
            root = tree.getroot()
        else:
            root = etree.Element("Misurazioni_MicroTrench")
            root.set("DataCreazione", datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
            tree = etree.ElementTree(root)

        # Aggiungi un nuovo blocco di sessione
        session_element = etree.SubElement(root, "Sessione", 
                                           DataInizio=datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

        for risultato in risultati_sessione:
            file_element = etree.SubElement(session_element, "File", 
                                            Nome=risultato['file_name'],
                                            Status=risultato['status'])

            # Metadati
            metadati = etree.SubElement(file_element, "Metadati")
            etree.SubElement(metadati, "DataMisurazione").text = risultato.get('data_misurazione', 'N/A')
            etree.SubElement(metadati, "ScaleFactor_um_per_px").text = f"{risultato['scale_factor']:.4f}"
            etree.SubElement(metadati, "Note").text = risultato.get('note', 'Nessuna nota aggiunta.')
            
            # Sezioni di Misura
            misure = etree.SubElement(file_element, "Misure")

            # Gradino/Angolo/Faceting
            misura_gradino = etree.SubElement(misure, "Gradino_Angolo_Faceting")
            etree.SubElement(misura_gradino, "Gradino_um").text = f"{risultato['gradino_um']:.4f}" if risultato['gradino_um'] is not None else "N/A"
            etree.SubElement(misura_gradino, "AngoloParete_deg").text = f"{risultato['angolo_gradi']:.2f}" if risultato['angolo_gradi'] is not None else "N/A"
            etree.SubElement(misura_gradino, "Faceting_um").text = f"{risultato['faceting_um']:.4f}" if risultato['faceting_um'] is not None else "N/A"

            # Trincea
            misura_trincea = etree.SubElement(misure, "Trincea_Larghezza")
            etree.SubElement(misura_trincea, "Larghezza_um").text = f"{risultato['trincea_larghezza_um']:.4f}" if risultato.get('trincea_larghezza_um') is not None else "N/A"
            
            # Micro Trench Sinistro
            misura_micro_left = etree.SubElement(misure, "MicroTrench_Sinistro")
            etree.SubElement(misura_micro_left, "Gradino_um").text = f"{risultato['micro_left_um']:.4f}" if risultato.get('micro_left_um') is not None else "N/A"
            etree.SubElement(misura_micro_left, "Quota_Y_prime_P9_um").text = f"{risultato['y_prime_p9_um']:.4f}" if risultato.get('y_prime_p9_um') is not None else "N/A"

            # Micro Trench Destro
            misura_micro_right = etree.SubElement(misure, "MicroTrench_Destro")
            etree.SubElement(misura_micro_right, "Gradino_um").text = f"{risultato['micro_right_um']:.4f}" if risultato.get('micro_right_um') is not None else "N/A"
            etree.SubElement(misura_micro_right, "Quota_Y_prime_P11_um").text = f"{risultato['y_prime_p11_um']:.4f}" if risultato.get('y_prime_p11_um') is not None else "N/A"

            # Fase 4: Delta e Allineamento
            misura_allineamento = etree.SubElement(misure, "Allineamento_Finale")
            etree.SubElement(misura_allineamento, "Delta_MicroTrench_um").text = f"{risultato['delta_microtrench_um']:.4f}" if risultato.get('delta_microtrench_um') is not None else "N/A"
            etree.SubElement(misura_allineamento, "Allineamento_Gradino_um").text = f"{risultato['allineamento_ab_um']:.4f}" if risultato.get('allineamento_ab_um') is not None else "N/A"
            
            # Aggiungi posizione e lato (se presenti)
            misura_info = etree.SubElement(file_element, "Informazioni")
            etree.SubElement(misura_info, "Posizione").text = risultato.get('posizione', 'N/A')
            etree.SubElement(misura_info, "Lato_Fondo").text = risultato.get('lato_fondo', 'N/A')

        # Scrittura del file
        tree.write(nome_file_output, pretty_print=True, xml_declaration=True, encoding="utf-8")
        
        # Conta i risultati salvati totali
        misure_salvate = len(root.xpath('.//File'))
        return misure_salvate

    except Exception as e:
        print(f"Errore nel salvataggio del file XML: {e}")
        return 0

def elabora_immagini_sequential(cartella_base):
    """
    Itera su tutte le sottocartelle e immagini per avviare la misurazione interattiva.
    """
    risultati_sessione = []
    
    try:
        # Itera sulle cartelle (es. 10_01, 10_02, ...)
        for subfolder in sorted(os.listdir(cartella_base)):
            subfolder_path = os.path.join(cartella_base, subfolder)
            
            if os.path.isdir(subfolder_path):
                print(f"\n==================================================")
                print(f"Elaborazione cartella: {subfolder}")
                print(f"==================================================")
                
                # Itera sui file TIFF all'interno della sottocartella
                for filename in sorted(os.listdir(subfolder_path)):
                    if filename.endswith(('.tif', '.tiff', '.TIF', '.TIFF')):
                        filepath = os.path.join(subfolder_path, filename)
                        print(f"\nCaricamento immagine: {filename}")
                        
                        try:
                            # Carica l'immagine e ottieni il fattore di scala
                            img, scale_factor = carica_immagine_tiff_con_scala(filepath)
                            
                            if img is None:
                                continue # Passa al file successivo

                            # Avvia la misurazione interattiva
                            risultati_immagine = misura_immagine_interattiva(img, filepath, scale_factor)
                            risultati_sessione.append(risultati_immagine)
                            
                            # Se l'utente ha completato l'immagine ('s'), richiedi i metadati
                            if risultati_immagine['status'] == 'Completed':
                                print("\n--- Metadati Richiesti ---")
                                
                                # RICHIESTA 1: Posizione
                                posizione_utente = input("Posizione (A, B, C, D, ...): ").upper()
                                risultati_immagine["posizione"] = posizione_utente
                                
                                # RICHIESTA 2: Lato/Fondo
                                posizione_fondo = input("Lato o Fondo (LATO, FONDO): ").upper()
                                risultati_immagine["lato_fondo"] = posizione_fondo.upper()

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
    cv2.waitKey(1)
    cv2.destroyAllWindows()
    cv2.waitKey(1) # Pulizia finale
