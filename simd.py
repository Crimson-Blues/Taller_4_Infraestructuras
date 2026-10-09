#Nombre: SIMD
#Autores:
# - Juan Diego Cárdenas Mejía - 2416437
# - Samuel Banguero Ortega - 2418671
#Fecha de Creación: 2025-02-17
#Descripción: Comparación del rendimiento de la multiplicación de matrices utilizando NumPy y una 
#implementación tradicional mediante bucles en Python.
#Curso: Infraestructuras Paralelas y Distribuidas
#Código: 750023C

import numpy as np 
import time
import statistics

#Funcion que multiplica 2 mastrices mediante fors
def multiplicacion_for(m1, m2):
    num_filas = len(m1)
    num_columnas = len(m2[0])
    dimension_comun = len(m1[0])

    resultado_for = np.zeros((num_filas, num_columnas))

    for i in range(num_filas):
        for j in range(num_columnas):
            for k in range(dimension_comun):
                resultado_for[i][j] += m1[i][k] * m2[k][j]

    return resultado_for


#Funcion que multiplica 2 matrices usando la libreria NumPy
def multiplicacion_numpy(m1,m2):

    resultado_numpy = m1 @ m2

    return resultado_numpy



if __name__ == '__main__':
    # Tamaño de las matrices
    num_filas = 1000
    num_columnas = 1000
    num_runs = 3  # Numero de iteraciones de testeo

    # Almacenar tiempos de ejecución
    times_sec = []
    times_np = []

    # Almacenar speed ups individuales
    speedups = []

    # Equivalencia de todas las soluciones
    equivs = []

    print(f"Ejecutando prueba con {num_runs} iteraciones...")

    for i in range(num_runs):
        # Matrices con números aleatorios
        matriz_1 = np.random.randint(1, 101, size=(num_filas, num_columnas))
        matriz_2 = np.random.randint(1, 101, size=(num_filas, num_columnas))
    
        # -------------- Multiplicación mediante fors ----------------------
        inicio_for = time.time()
        resultado_for = multiplicacion_for(matriz_1, matriz_2)
        fin_for = time.time()
        t_for = fin_for - inicio_for
        times_sec.append(t_for)
    
        # -------------- Multiplicación mediante np ----------------------
        inicio_np = time.time()
        resultado_np = multiplicacion_numpy(matriz_1, matriz_2)
        fin_np = time.time()
        t_np = fin_np - inicio_np
        times_np.append(t_np)

        # -------------- Validación y Speedup ----------------------
        son_iguales = np.array_equal(resultado_for, resultado_np)
        equivs.append(son_iguales)
        speedup = t_for / t_np if t_np > 0 else 0
        speedups.append(speedup)

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
        ("SIMD", times_np, speedups, equivs),
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


