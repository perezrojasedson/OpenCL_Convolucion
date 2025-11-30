#ifndef CONVOLUCION_PARALELO_H
#define CONVOLUCION_PARALELO_H

#include "cl_manager.h"

/**
 * Ejecuta la convolución usando OpenCL en la GPU.
 * * @param mgr       Puntero al gestor de OpenCL (ya inicializado y con kernel cargado).
 * @param input     Datos de la imagen de entrada (Host).
 * @param output    Buffer donde se guardará el resultado (Host).
 * @param width     Ancho de la imagen.
 * @param height    Alto de la imagen.
 * @param filter    Array con los pesos del filtro.
 * @param k_size    Tamaño del kernel (ej. 3).
 */
void convolucion_paralelo(
    CLManager* mgr,
    const unsigned char* input,
    unsigned char* output,
    int width,
    int height,
    const float* filter,
    int k_size,
    double* kernel_time_ms
);

#endif // CONVOLUCION_PARALELO_H