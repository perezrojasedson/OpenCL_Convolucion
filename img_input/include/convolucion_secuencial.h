#ifndef CONVOLUCION_SECUENCIAL_H
#define CONVOLUCION_SECUENCIAL_H

/**
 * Ejecuta la convolución de manera secuencial en la CPU (Single Thread).
 * Recorre la imagen píxel a píxel aplicando la máscara del filtro.
 * * @param input     Puntero a los datos de la imagen de entrada (0-255).
 * @param output    Puntero al buffer donde se guardará la imagen procesada.
 * @param width     Ancho de la imagen en píxeles.
 * @param height    Alto de la imagen en píxeles.
 * @param kernel    Array de floats con los coeficientes del filtro (ej. 3x3 = 9 valores).
 * @param k_size    Dimensión del kernel (ej. 3 para una matriz 3x3).
 */
void convolucion_secuencial(
    const unsigned char* input,
    unsigned char* output,
    int width,
    int height,
    const float* kernel,
    int k_size
);

void progreso (int y, int height);

#endif // CONVOLUCION_SEQ_H