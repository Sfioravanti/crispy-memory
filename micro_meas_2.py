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
    elif k == ord('i'): # Sposta su
      parallel_lines_offsets_px[active_line_index] -= LINE_MOVE_STEP_PX
      print(f"Retta {active_line_index + 1} spostata SU (i). Offset: {parallel_lines_offsets_px[active_line_index]} px.")
      redraw_image()
    elif k == ord('k'): # Sposta giù
      parallel_lines_offsets_px[active_line_index] += LINE_MOVE_STEP_PX
      print(f"Retta {active_line_index + 1} spostata GIÙ (k). Offset: {parallel_lines_offsets_px[active_line_index]} px.")
      redraw_image()

    # 3. Calcolo (Tasto 'c')
    elif k == ord('c'):
      y_p9 = risultato_immagine.get('y_prime_p9_um')
      y_p11 = risultato_immagine.get('y_prime_p11_um')

      if y_p9 is not None and y_p11 is not None:
        # Calcolo della differenza assoluta: Quota P11 (Destro) - Quota P9 (Sinistro)
        # Il risultato in um è già disponibile
        diff_microtrench_calc = y_p11 - y_p9
        risultato_immagine['diff_microtrench_abs_um'] = diff_microtrench_calc

        # Stampa del calcolo
        print("\n==============================================")
        print("CALCOLO DIFFERENZA MICRO TRENCHES")
        print(f"Differenza (Quota P11 Destra - Quota P9 Sinistra): {diff_microtrench_calc:.4f} um.")
        print("==============================================")
        risultati_fase_4_stampati = True
      else:
        print("Dati Micro Trenches non disponibili per il calcolo.")

    # 4. Controlli di fine ciclo ('q', 's', 'x')
    elif k == ord('q'):
      risultato_immagine['status'] = 'Skipped'
      return risultato_immagine
    elif k == ord('s'):
      risultato_immagine['status'] = 'Completed' if risultati_fase_4_stampati else 'Incomplete'
      return risultato_immagine
    elif k == ord('x') or k == 27:
      # Undo in questa fase torna alla Fase 3
      fase_corrente = 3
      risultati_fase_4_stampati = False
      diff_microtrench_calc = None
      print("Tornato alla Fase 3 (Micro Trench Destro).")
      redraw_image()
      break

    # Nessuna logica di spostamento punto per 'i','k','j','l' (già gestita sopra per la linea)

    if not risultati_fase_4_stampati:
      print(f"Retta Attiva: {active_line_index + 1}. Premi 1/2/3 per selezionare, i/k per spostare.")

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
# NUOVE UTILITY PER IL DISEGNO
# ==============================================================================

def draw_line_from_points(img, p_start, p_end, color, thickness):
  """
  Funzione helper per estendere la retta definita da due punti oltre i bordi dell'immagine.
  (Basata sulla logica di calcolo retta per estensione)
  """
  x1, y1 = p_start
  x2, y2 = p_end

  H, W = img.shape[:2]

  if abs(x2 - x1) < 1: # Linea quasi verticale
    cv2.line(img, (x1, 0), (x1, H), color, thickness)
    return

  m = (y2 - y1) / (x2 - x1)
  q = y1 - m * x1

  # Punti di intersezione con i bordi dell'immagine
  endpoints = []

  # Intersezioni con bordi verticali (x=0 e x=W)
  y_at_0 = int(q)
  y_at_W = int(m * W + q)
  if 0 <= y_at_0 < H:
    endpoints.append((0, y_at_0))
  if 0 <= y_at_W < H:
    endpoints.append((W, y_at_W))

  # Intersezioni con bordi orizzontali (y=0 e y=H)
  if abs(m) > 1e-6:
    x_at_0 = int(-q / m)
    x_at_H = int((H - q) / m)
    if 0 <= x_at_0 < W:
      endpoints.append((x_at_0, 0))
    if 0 <= x_at_H < W:
      endpoints.append((x_at_H, H))

  # Se ci sono almeno due punti unici, traccia la linea tra i due più distanti
  if len(endpoints) >= 2:
    max_dist = -1
    p_final_1, p_final_2 = endpoints[0], endpoints[1]

    for i in range(len(endpoints)):
      for j in range(i + 1, len(endpoints)):
        dist = math.sqrt((endpoints[i][0] - endpoints[j][0])**2 + (endpoints[i][1] - endpoints[j][1])**2)
        if dist > max_dist:
          max_dist = dist
          p_final_1, p_final_2 = endpoints[i], endpoints[j]

    cv2.line(img, p_final_1, p_final_2, color, thickness)
    return

  # Fallback per linee quasi orizzontali o verticali
  if abs(m) < 1e-6:
    y_avg = int((y1 + y2) / 2)
    cv2.line(img, (0, y_avg), (W, y_avg), color, thickness)
    return


def draw_parallel_line(img, p1, p2, offset_px, color, thickness):
  """
  Disegna una retta parallela a P1-P2, spostata verticalmente (normalmente) di offset_px.
  """
  p1_np = np.array(p1)
  p2_np = np.array(p2)

  v_dir = p2_np - p1_np
  v_dir_norm = np.linalg.norm(v_dir)

  if v_dir_norm < 1e-6:
    return

  # Calcola il vettore normale (ruotato di 90 gradi)
  v_norm = np.array([-v_dir[1], v_dir[0]])
  v_unit_norm = v_norm / v_dir_norm

  # Sposta i punti P1 e P2 lungo il vettore normale di 'offset_px'
  offset_vector = v_unit_norm * offset_px

  p1_par = p1_np + offset_vector
  p2_par = p2_np + offset_vector

  # Disegna la retta estesa
  draw_line_from_points(img, p1_par.astype(int), p2_par.astype(int), color, thickness)

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

  # --- FIX PER ACCESSO SICURO A P1 e P2 ---
  # Definisci p1 e p2 solo se esistono i punti, altrimenti usa None
  p1, p2 = None, None
  if len(punti_selezionati) >= 2:
    p1 = punti_selezionati[0]
    p2 = punti_selezionati[1]
  # ----------------------------------------
"""def redraw_image():
  #
  Funzione per ridisegnare l'immagine sullo schermo. Tutte le coordinate
  sono considerate coordinate immagine (1:1).

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
           p5_orig, (0, 165, 255), 1) # Arancione"""

  # 1. Disegna la retta di riferimento P1-P2 (Top Surface)
  # Ora il codice che segue deve solo controllare se p1 e p2 non sono None
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
      p3_proj = project_point_onto_line(p3 , p1 , p2 )
      cv2.line(immagine_modificabile, p3_proj .tolist(), p5 , (0, 165, 255), 1) # Arancione

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

  if fase_corrente == 4 and len(punti_selezionati) >= 2:
    #p1 = punti_selezionati[0]
    #p2 = punti_selezionati[1]

  H, W = immagine_modificabile.shape[:2]

  # Disegna le 3 rette parallele
  for i, offset_px in enumerate(parallel_lines_offsets_px):
    color = (0, 0, 255) # Rosso
    thickness = 1

    if i == active_line_index:
      color = (0, 255, 255) # Giallo per la retta attiva
      thickness = 2

    draw_parallel_line(immagine_modificabile, p1, p2, offset_px, color, thickness)

    # Aggiungi indicatore "sposta su/giù" per la retta attiva
    if i == active_line_index:
      cv2.putText(immagine_modificabile, f"LINEA {i+1} (i/k) \u25B2\u25BC", (10, H - 40 - (i*20)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, thickness+1)

  # Stampa il prompt per la Fase 4
  cv2.putText(immagine_modificabile, "Fase 4: Rette Parallele. Premi 1/2/3 per selezionare la retta. Premere 'c' per il calcolo.", (10, H - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

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
