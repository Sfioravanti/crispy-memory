import cv2
import numpy as np

def key_tester():
    """
    Programma di diagnostica per identificare i codici dei tasti in ambiente OpenCV.
    """
    window_name = "Keycode Tester - Premi i tasti freccia"
    
    # Crea un'immagine nera (500x500 pixel) su cui disegnare
    img = np.zeros((500, 500, 3), dtype=np.uint8)
    
    # Aggiunge istruzioni all'immagine
    cv2.putText(img, "Premi i tasti freccia (Up, Down, Left, Right)", (50, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(img, "I codici appariranno nella console.", (50, 80), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(img, "Premi 'q' per uscire.", (50, 110), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    
    cv2.imshow(window_name, img)
    
    print("--- INIZIO TEST KEYCODES ---")
    print("Clicca sulla finestra e premi i tasti freccia.")
    
    while True:
        # Cattura l'input da tastiera. Il valore 0xFF è essenziale per Linux/macOS.
        # k è il codice del tasto premuto, o 255 se nessun tasto è stato premuto.
        k = cv2.waitKey(100)
        
        # Filtra 'k' per valori diversi da 255 (ovvero, un tasto è stato premuto)
        if k != -1 and k != 255:
            # Test per uscire
            if k == ord('q') or k == 27: # 'q' o ESC
                print("Uscita...")
                break
            
            # Stampa il codice per la diagnostica
            print(f"Tasto premuto: Codice {k}")
            
    cv2.destroyAllWindows()

if __name__ == "__main__":
    key_tester()
