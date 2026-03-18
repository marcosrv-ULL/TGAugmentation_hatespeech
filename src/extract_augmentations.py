import json
import os

def expand_jsonl(input_file, output_file):
    # Comprobamos si el archivo de entrada existe
    if not os.path.exists(input_file):
        print(f"Error: El archivo {input_file} no existe.")
        return

    print(f"Procesando: {input_file}...")
    
    count = 0
    
    # Abrimos el archivo de entrada (lectura) y el de salida (escritura)
    with open(input_file, 'r', encoding='utf-8') as f_in, \
         open(output_file, 'w', encoding='utf-8') as f_out:
        
        for line in f_in:
            try:
                # Cargar la fila actual
                data = json.loads(line)
                
                # Extraer la lista de augmentations y eliminarla del diccionario base
                # (usamos pop para sacarla y que no se repita en las filas hijas)
                augmentations = data.pop('augmentations', [])
                
                # Si queremos conservar el texto original como referencia, lo guardamos en otra variable
                # data['original_text_reference'] = data['text'] 
                
                # Iterar sobre cada aumentación
                for aug_text in augmentations:
                    # Crear una copia de los datos base (sin la lista de augmentations)
                    new_row = data.copy()
                    
                    # Asignar la aumentación al campo 'text' 
                    # (OJO: Si prefieres que vaya a 'text_masked', cámbialo aquí)
                    new_row['text'] = aug_text

                    new_row['text_masked'] = aug_text

                    
                    # Opcional: Marcar que esta fila es una aumentación
                    new_row['is_augmentation'] = True
                    
                    # Escribir la nueva fila en el archivo de salida
                    f_out.write(json.dumps(new_row) + '\n')
                    count += 1
                    
            except json.JSONDecodeError:
                print(f"Advertencia: No se pudo decodificar una línea en el JSONL.")
                continue

    print(f"---")
    print(f"Proceso completado.")
    print(f"Se han generado {count} filas nuevas.")
    print(f"Guardado en: {output_file}")

if __name__ == "__main__":
    # Rutas de los archivos
    input_path = 'en_dataset_with_stop_words/en_dataset_with_stop_words_masked_train_augmented_tgarag2.jsonl'
    
    # Generamos un nombre para el archivo de salida en la misma carpeta
    output_path = input_path.replace('.jsonl', '_exploded.jsonl')

    # Ejecutar la función
    expand_jsonl(input_path, output_path)