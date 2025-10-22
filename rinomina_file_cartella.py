import os

def rinomina_files(percorso_principale, numero_da_sostituire, nuovo_numero):
    """
    Rinomina tutti i file in una directory e nelle sue sottodirectory,
    sostituendo un numero specifico nel nome del file.
    """
    print(f"Avvio la scansione da: {percorso_principale}")
    print(f"Sostituisco '{numero_da_sostituire}' con '{nuovo_numero}'\n")

    # Utilizza os.walk per scansionare tutte le sottocartelle
    for dirpath, dirnames, filenames in os.walk(percorso_principale):
        for filename in filenames:
            # Controlla se il numero da sostituire è nel nome del file
            if numero_da_sostituire in filename:
                try:
                    # Crea il nuovo nome del file
                    nuovo_nome = filename.replace(numero_da_sostituire, nuovo_numero)

                    # Costruisci il percorso completo del file originale e di quello nuovo
                    percorso_originale = os.path.join(dirpath, filename)
                    percorso_nuovo = os.path.join(dirpath, nuovo_nome)

                    # Esegue la ridenominazione
                    os.rename(percorso_originale, percorso_nuovo)
                    print(f"Rinominato: {filename} -> {nuovo_nome}")
                except Exception as e:
                    print(f"Errore durante la ridenominazione del file {filename}: {e}")
    
    print("\nProcesso di ridenominazione completato.")

# --- Dati da modificare ---
# Sostituisci questi valori con il percorso della tua cartella e i numeri da cambiare

cartella_da_processare = r"C:\Users\Lavoro\Desktop\SPCT_FIORAVANTI\U_21" # Mettere il percorso della cartella principale
numero_da_sostituire = "T21" # Il numero che vuoi trovare nei nomi dei file
nuovo_numero = "U21" # Il numero con cui vuoi sostituirlo

# Esegue la funzione
rinomina_files(cartella_da_processare, numero_da_sostituire, nuovo_numero)
