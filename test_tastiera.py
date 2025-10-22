import cv2
import numpy as np

# Crea una semplice immagine nera
img = np.zeros((200, 400), np.uint8)
cv2.putText(img, "Premi un tasto...", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 255, 2)
cv2.imshow("Test Tastiera", img)

while True:
    # Il comando che aspetta il tasto
    k = cv2.waitKey(1)
    if k != -1: # Se un tasto è stato premuto
        print(f"Valore del tasto premuto: {k}")
        print(f"Carattere corrispondente: {chr(k)}")
    
    # Esci premendo il tasto 'q'
    if k == ord('q'):
        break

cv2.destroyAllWindows()
