import subprocess
import os
import datetime
import time

def sync_to_github(file_path: str, commit_message: str):
    """
    Esegue i comandi Git per aggiungere il file, fare un commit e fare un push al repository remoto.

    :param file_path: Il percorso del file da aggiungere (es. 'riepilogo_misure.xml').
    :param commit_message: Il messaggio di commit.
    :return: True se il push ha successo, False altrimenti.
    """
    
    # Ottiene la directory in cui risiede lo script per impostarla come CWD
    repo_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 1. git add
    print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] Esecuzione: git add {file_path}")
    add_process = subprocess.run(['git', 'add', file_path], cwd=repo_dir, capture_output=True, text=True)
    if add_process.returncode != 0:
        print("ERRORE DURANTE git add:")
        print(add_process.stderr)
        return False

    # 2. git commit
    print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] Esecuzione: git commit -m \"{commit_message}\"")
    commit_process = subprocess.run(['git', 'commit', '-m', commit_message], cwd=repo_dir, capture_output=True, text=True)
    
    # Controlla se ci sono stati cambiamenti effettivi da committare
    if commit_process.returncode != 0:
        if "nothing to commit" in commit_process.stdout or "nothing added to commit" in commit_process.stderr:
            print("Nessun cambiamento rilevato in riepilogo_misure.xml. Saltando il commit e il push.")
            return True # Considerato un successo, non c'è nulla da fare
        
        print("ERRORE DURANTE git commit:")
        print(commit_process.stderr)
        return False

    # 3. git push
    print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] Esecuzione: git push origin main (o master)")
    
    # Prova prima 'main' (standard moderno)
    push_process = subprocess.run(['git', 'push', 'origin', 'main'], cwd=repo_dir, capture_output=True, text=True)
    
    # Se fallisce, prova 'master' (standard meno recente)
    if push_process.returncode != 0:
        print("Tentativo fallito con 'main'. Riprovo con 'master'.")
        time.sleep(1)
        push_process = subprocess.run(['git', 'push', 'origin', 'master'], cwd=repo_dir, capture_output=True, text=True)
        
        if push_process.returncode != 0:
            print("ERRORE CRITICO DURANTE git push:")
            print(push_process.stderr)
            print("\n** Controlla le credenziali e l'accesso al repository remoto **")
            return False

    # Se fallisce, prova 'volupta' (standard meno recente)
    if push_process.returncode != 0:
        print("Tentativo fallito con 'master'. Riprovo con 'volupta'.")
        time.sleep(1)
        push_process = subprocess.run(['git', 'push', 'origin', 'volupta'], cwd=repo_dir, capture_output=True, text=True)
        
        if push_process.returncode != 0:
            print("ERRORE CRITICO DURANTE git push:")
            print(push_process.stderr)
            print("\n** Controlla le credenziali e l'accesso al repository remoto **")
            return False


    print("==================================================")
    print("Sincronizzazione su GitHub completata con successo!")
    print("==================================================")
    return True

if __name__ == '__main__':
    # Esempio di utilizzo dello script indipendente
    xml_file = 'riepilogo_misure.xml'
    message = f"Misurazioni automatiche aggiornate: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    sync_to_github(xml_file, message)
