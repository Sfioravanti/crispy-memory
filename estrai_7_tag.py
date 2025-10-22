import tifffile
import os
from lxml import etree

def estrai_primi_n_tag(file_path, num_tags=5):
    """
    Estrae i primi N tag da un file TIFF e cerca tag specifici per nome.
    """
    tags = {}
    try:
        with tifffile.TiffFile(file_path) as tif:
            page = tif.pages[0]
            
            # 1. Estrae i primi N tag (come prima)
            for i, tag in enumerate(page.tags.values()):
                if i >= num_tags:
                    break
                tags[tag.name] = str(tag.value)

            # 2. Cerca i tag specifici nel metadato FEI
            fei_tags = tif.fei_metadata
            if fei_tags and 'EBeam' in fei_tags:
                ebeam_tags = fei_tags['EBeam']
                if 'HV' in ebeam_tags:
                    tags['HV'] = ebeam_tags['HV']
                if 'TiltCorrectionAngle' in ebeam_tags:
                    tags['TiltCorrectionAngle'] = ebeam_tags['TiltCorrectionAngle']

            if 'Scan' in fei_tags:  # Secondo livello di indentazione
                scan_tags = fei_tags['Scan']
                if 'PixelWidth' in scan_tags:
                    tags['PixelWidth'] = scan_tags['PixelWidth']

            else:
                print("Metadati FEI non trovati nel file.")
    
    except Exception as e:
        print(f"Errore durante l'estrazione dei tag da {os.path.basename(file_path)}: {e}")
        return None
    return tags

def crea_xml_acquisizione(root_directory, output_filename='acquisizione.xml'):
    """
    Scansiona tutte le sottocartelle per i file .tif, estrae i primi 5 tag
    e li salva in un unico file XML.
    """
    # Crea l'elemento radice del nostro XML
    root = etree.Element("Acquisizioni")
    
    # Scansiona tutte le cartelle e sottocartelle
    for dirpath, _, filenames in os.walk(root_directory):
        for filename in filenames:
            if filename.endswith(('.tif', '.tiff')):
                file_path = os.path.join(dirpath, filename)
                print(f"Elaborazione del file: {file_path}")
                
                # Estrae i tag
                dati_tag = estrai_primi_n_tag(file_path)
                
                if dati_tag:
                    # Crea un sotto-elemento per ogni file
                    file_element = etree.SubElement(root, "File", path=file_path)
                    
                    # Aggiunge i tag estratti come sotto-elementi
                    for tag_name, tag_value in dati_tag.items():
                        etree.SubElement(file_element, tag_name).text = str(tag_value)
    
    # Salva il file XML
    albero = etree.ElementTree(root)
    with open(output_filename, "wb") as f:
        albero.write(f, pretty_print=True, xml_declaration=True, encoding='UTF-8')
    
    print(f"\nProcesso completato. File salvato in: {output_filename}")


# --- Istruzioni per l'uso ---

# Sostituisci questo percorso con il tuo percorso principale
percorso_principale = 'C:\\Users\\Lavoro\\Desktop\\SPCT_FIORAVANTI\\'

# Esegui lo script
crea_xml_acquisizione(percorso_principale)
