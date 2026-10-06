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
    # Modifica este valor para cambiar el tamaño de la matriz
    num_filas = 1000
    num_columnas = 1000

    # Matrices con números aleatorios
    matriz_1 = np.random.randint(1, 101, size=(num_filas, num_columnas))
    matriz_2 = np.random.randint(1, 101, size=(num_filas, num_columnas))

    # -------------- Multiplicación mediante fors ----------------------
    inicio_for = time.time()
    resultado_for = multiplicacion_for(matriz_1, matriz_2)
    fin_for = time.time()
    t_for = fin_for - inicio_for

    # -------------- Multiplicación mediante np ----------------------
    inicio_np = time.time()
    resultado_np = multiplicacion_numpy(matriz_1, matriz_2)
    fin_np = time.time()
    t_np = fin_np - inicio_np

    # -------------- Validación y Speedup ----------------------
    son_iguales = np.array_equal(resultado_for, resultado_np)
    speedup = t_for / t_np if t_np > 0 else 0

    # -------------- Impresión de la Tabla ----------------------
    print("\n" + "="*75)
    print(" " * 20 + "RESULTADOS SIMD (NumPy vs Bucles)")
    print("="*75)
    print(f"{'Dimensión':<12} | {'Tiempo Fors (s)':<18} | {'Tiempo NumPy (s)':<18} | {'Speedup':<10}")
    print("-" * 75)
    print(f"{f'{num_filas}x{num_columnas}':<12} | {t_for:<18.5f} | {t_np:<18.6f} | {speedup:<10.2f}x")
    print("="*75)
    print(f"Resultados equivalentes: {son_iguales}\n")


