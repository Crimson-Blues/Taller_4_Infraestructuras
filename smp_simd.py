#Nombre: Híbrido SMP-SIMD
#Autores:
# - Juan Diego Cárdenas Mejía - 2416437
# - Samuel Banguero Ortega - 2418671
#Fecha de Creación: 2025-02-17
#Descripción: Simulación de un sistema híbrido que combina SMP (hilos) y SIMD (NumPy) 
#para procesar una matriz de 10000x10000 y comparar con la versión secuencial.
#Curso: Infraestructuras Paralelas y Distribuidas
#Código: 750023C

import numpy as np
import threading
import time


# Función que divide matriz en chunks de tamaño chunk_row X chunk_col
def split_matrix(matrix, chunk_row, chunk_col):
    rows, cols = matrix.shape # Dimensiones de matriz original 

    # Cálculo de módulo y número de chunks
    q_row, r_row = divmod(rows, chunk_row) 
    q_col, r_col = divmod(cols, chunk_col)

    chunks = [] # Recolectar submatrices resultantes

    if r_row == 0 and r_col == 0:
        for i in range(q_row):
            for j in range(q_col):
                chunks.append(matrix[i*chunk_row : (i+1)*chunk_row, 
                                     j*chunk_col : (j+1)*chunk_col])

    return chunks


# Función ejecutada por cada hilo: realiza la suma vectorizada (SIMD) de su submatriz
def hilo_suma_simd(chunk_matrix, results, index):
    # Operación SIMD: NumPy realiza la suma por columnas a nivel de hardware
    suma_columnas = np.sum(chunk_matrix, axis=0) 
    results[index] = np.sum(suma_columnas) # Consolidación del resultado del bloque


# Función principal para ejecutar la solución híbrida (SMP con hilos + SIMD con NumPy)
def par_sum_matrix_smp_simd(main_matrix, chunk_rows, chunk_cols):
    chunks = split_matrix(main_matrix, chunk_rows, chunk_cols)
    results = [0] * len(chunks) # Lista simple para almacenar resultados de cada hilo
    threads = [] # Almacenar hilos

    # Crear hilos asignando cada bloque a un hilo (SMP)
    for i in range(len(chunks)):
        thread = threading.Thread(target=hilo_suma_simd, args=(chunks[i], results, i))
        threads.append(thread)
        thread.start()

    # Esperar a que todos los hilos terminen
    for thread in threads:
        thread.join()

    # Combinar resultados parciales obteniendo la suma total
    final_result = sum(results)
    
    return final_result


# Suma secuencial de matriz mediante bucles en Python
def sec_sum_matrix(matrix):
    rows, cols = matrix.shape
    total_sum = 0
    for i in range(rows):
        for j in range(cols):
            total_sum += int(matrix[i, j]) # Cast a python integer para no sobrepasar límite de suma de Numpy

    return total_sum


# Bloque principal
if __name__ == '__main__':
    # Configuración de dimensiones para el ejercicio integrador
    num_rows = 10000
    num_cols = 10000

    chunk_rows = 1000
    chunk_cols = 1000

    # Crear matriz de 10000x10000 con números aleatorios entre 1 y 100
    matrix = np.random.randint(1, 101, size=(num_rows, num_cols))

    # ----------- Procesamiento secuencial --------------
    inicio_sec = time.time()
    sec_result = sec_sum_matrix(matrix)
    fin_sec = time.time()
    t_sec = fin_sec - inicio_sec

    # ----------- Procesamiento Híbrido (SMP - SIMD) ------------
    inicio_par = time.time()
    par_result = par_sum_matrix_smp_simd(matrix, chunk_rows, chunk_cols)
    fin_par = time.time()
    t_par = fin_par - inicio_par

    # -------------- Resultados ----------------------
    print(f"Resultados equivalentes: {sec_result == par_result}")
    print(f"Tamaño de la matriz: {num_rows}x{num_cols}")
    print(f"Tamaño de los bloques: {chunk_rows}x{chunk_cols} (Total hilos: {len(split_matrix(matrix, chunk_rows, chunk_cols))})")
    print(f"Tiempo total de procesamiento secuencial: {t_sec:.3f} segundos")
    print(f"Tiempo total de procesamiento híbrido (SMP-SIMD): {t_par:.3f} segundos")
    print(f"Speedup obtenido: {t_sec/t_par:.2f}x")