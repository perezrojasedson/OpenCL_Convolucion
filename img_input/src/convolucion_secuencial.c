#include "convolucion_secuencial.h"

#include <stdio.h>

// Variable estática
static int ultimo_porcentaje = -1;


void convolucion_secuencial(const unsigned char* input, unsigned char* output,
                            int width, int height, const float* kernel, int k_size) {

    int half = k_size / 2;
    long long total_ops = 0; // Contador para estadística

    // Iterar sobre cada pixel de la imagen (Filas y Columnas)
    for (int y = 0; y < height; y++) {
        //--- PORCENTAJE DE PROGRESO
        progreso(y, height);
        //------------------------------
        for (int x = 0; x < width; x++) {

            float sum = 0.0f;

            // Convolución: Recorrer la máscara/kernel sobre el pixel actual
            for (int ky = -half; ky <= half; ky++) {
                for (int kx = -half; kx <= half; kx++) {

                    // Calcular coordenada del pixel vecino
                    int ix = x + kx;
                    int iy = y + ky;

                    // Manejo de bordes (Clamp) - Igual que en el Kernel GPU
                    if(ix < 0) ix = 0;
                    if(ix >= width) ix = width - 1;
                    if(iy < 0) iy = 0;
                    if(iy >= height) iy = height - 1;

                    // Obtener valor del píxel (0-255) y peso del kernel
                    // Convertimos a float para operar con precisión
                    float pixel_val = (float)input[iy * width + ix];
                    float weight = kernel[(ky + half) * k_size + (kx + half)];

                    sum += pixel_val * weight;
                    total_ops++;
                }
            }

            // "Clamp" del resultado final para que encaje en un byte (0-255)
            if (sum < 0) sum = 0;
            if (sum > 255) sum = 255;

            output[y * width + x] = (unsigned char)sum;
        }
    }
    // 3. Cierre estético
    // Si el último no fue 100 (por redondeo), lo ponemos para cerrar bien
    if (ultimo_porcentaje != 100) {
        printf("100%%");
    }
    printf("\n"); // Ahora sí, salto de línea final

    printf("[Info] Operaciones Totales: %lld\n", total_ops);
}

// Función auxiliar visual (Estilo Horizontal)
void progreso(int y, int height) {
    // Calcular porcentaje
    int porcentaje = (y + 1) * 100 / height;

    // Imprimir solo si cambia para no llenar la pantalla
    if (porcentaje != ultimo_porcentaje) {
        if (porcentaje % 10 == 0) {
            printf("%d%% ", porcentaje);
            fflush(stdout);
            ultimo_porcentaje = porcentaje;
        }
    }
}