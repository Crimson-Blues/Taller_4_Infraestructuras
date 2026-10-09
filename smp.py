#Nombre: SMP
#Autores:
# - Juan Diego Cárdenas Mejía - 2416437
# - Samuel Banguero Ortega - 2418671
#Fecha de Creación: 2025-02-17
#Descripción: Funciones que dividen la suma de todos los elementos de una matriz entre hilos
#Curso: Infraestructuras Paralelas y Distribuidas
#Código: 750023C

import numpy as np
import threading
import random
import time
import multiprocessing as mp
import statistics #For analysis of results


# Función para sumar todos los elementos de una matriz
# Almacenar resultados en lista en índice específico
def matrix_add_thread(chunk_matrix, results, index):
    results[index] = sec_sum_matrix(chunk_matrix) #Empleamos suma secuencial

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
                                     j*chunk_col : (j+1)*chunk_col]) # Llenar de chunks con slicing

    return chunks # Retorna lista de submatrices

# Función principal para ejecutar la división de la suma en hilos
def par_sum_matrix_threads(main_matrix, chunk_rows, chunk_cols):
    chunks = split_matrix(main_matrix, chunk_rows, chunk_cols)
    results = [0]*len(chunks) # Lista simple para almacenar resultados
    threads = [] # Almacenar hilos

    # Crear hilos para la fase de mapeo
    for i in range(len(chunks)):
        thread = threading.Thread(target=matrix_add_thread, args=(chunks[i], results, i))
        threads.append(thread)
        thread.start()

    # Esperar a que todos los hilos terminen
    for thread in threads:
        thread.join()

    # Combinar resultados con una única suma final de la suma de los chunks
    final_result = sum(results)
    
    return final_result


# Función principal para ejecutar la división de la suma en procesos de OS
def par_sum_matrix_procs(main_matrix, chunk_rows, chunk_cols):
    chunks = split_matrix(main_matrix, chunk_rows, chunk_cols)

    with mp.Pool() as pool:
        # Creación de los trabajadores: ejecutan suma secuencial de chunk de matriz
        procs = [pool.apply_async(sec_sum_matrix, (chunk,)) for chunk in chunks]
        
        # Recuperar resultados de pool de procesos
        results = [p.get() for p in procs]

    # Combinación final de resultados
    final_result = sum(results)
    
    return final_result


# Suma secuencial de matriz
def sec_sum_matrix(matrix):
    rows, cols = np.shape(matrix)
    sum = 0 # Acumulador simple inicializado en 0
    # Ciclos clásicos for anidados para recorrer matriz
    for i in range(rows):
        for j in range(cols):
            sum += matrix[i, j] # Suma de celdas una por una

    return sum

# Suma de matriz de 1000x1000
if __name__ == "__main__":
    # Párametros de ejecución
    num_rows = 5000
    num_cols = 5000
    chunk_rows = 100
    chunk_cols = 100
    num_runs = 10  # Numero de iteraciones de testeo

    # Almacenar tiempos de ejecución
    times_sec = []
    times_threads = []
    times_procs = []

    # Almacenar speed ups individuales
    speedups_threads = []
    speedups_procs = []

    # Equivalencia de todas las soluciones
    equivs_t = []
    equivs_p = []

    print(f"Ejecutando prueba con {num_runs} iteraciones...")

    for i in range(num_runs):
        # Genera una nueva matriz por cada iteración para aumentar variación
        matrix = np.random.randint(1, 101, size=(num_rows, num_cols))

        # ----------- Procesamiento secuencial --------------
        inicio_sec = time.perf_counter()
        sec_result = sec_sum_matrix(matrix)
        fin_sec = time.perf_counter()
        t_sec = fin_sec - inicio_sec
        times_sec.append(t_sec)

        # ----------- Procesamiento paralelo con hilos ------------
        inicio_par = time.perf_counter()
        par_result = par_sum_matrix_threads(matrix, chunk_rows, chunk_cols)
        fin_par = time.perf_counter()
        t_par_t = fin_par - inicio_par
        times_threads.append(t_par_t)
        equivs_t.append(np.array_equal(sec_result, par_result))

        # ----------- Procesamiento paralelo con procesos ------------
        inicio_par_p = time.perf_counter()
        par_p_result = par_sum_matrix_procs(matrix, chunk_rows, chunk_cols)
        fin_par_p = time.perf_counter()
        t_par_p = fin_par_p - inicio_par_p
        times_procs.append(t_par_p)
        equivs_p.append(np.array_equal(sec_result, par_p_result))

        # ----------- Speedups ------------
        speedups_threads.append(t_sec / t_par_t)
        speedups_procs.append(t_sec / t_par_p)

        print(f"Iteration {i + 1}/{num_runs} completed.")

    # ----------- Resumen de resultados --------------
    print("\n" + "=" * 80)
    print(
        f"{'Implementación':<12} | {'Tiempo promedio (s)':<14} | {'Desviación Estándar (s)':<12} | {'Speedup promedio':<12} |{'Equivalencia':<12}"
    )
    print("=" * 80)

    # Ejecución secuencial de base
    mean_sec = statistics.mean(times_sec)
    std_sec = statistics.stdev(times_sec) if num_runs > 1 else 0.0
    print(f"{'Secuencial':<12} | {mean_sec:<14.5f} | {std_sec:<12.5f} | {'1.00x (Base)':<12} | {'True':<12}")

    # Ejecuciones paralelas
    parallel_data = [
        ("Hilos", times_threads, speedups_threads, equivs_t),
        ("Procesos", times_procs, speedups_procs, equivs_p),
    ]

    for name, times, speedups, equivs in parallel_data:
        mean_time = statistics.mean(times)
        std_dev = statistics.stdev(times) if num_runs > 1 else 0.0
        avg_speedup = statistics.mean(speedups)
        std_speedup = statistics.stdev(speedups) if num_runs > 1 else 0.0

        print(
            f"{name:<12} | {mean_time:<14.5f} | {std_dev:<12.5f} | {avg_speedup:.2f}x (±{std_speedup:.2f}) | {all(equivs)}"
        )

    print("=" * 80)
