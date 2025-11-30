#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "image_utils.h"
#include "cl_manager.h"
#include "convolucion_secuencial.h"
#include "convolucion_paralelo.h"

// Helper visual para títulos bonitos
void imprimir_titulo(const char* titulo) {
    printf("\n");
    printf("╔════════════════════════════════════════════════════╗\n");
    printf("║  %-50s║\n", titulo);
    printf("╚════════════════════════════════════════════════════╝\n");
}

int main() {
    // --- SETUP ---
    imprimir_titulo("CONFIGURACIÓN  E  INICIALIZACIÓN");

    // 1. Filtro
    const int k_size = 3;
    const float kernel_blur[9] = {
        1.0f/9, 1.0f/9, 1.0f/9,
        1.0f/9, 1.0f/9, 1.0f/9,
        1.0f/9, 1.0f/9, 1.0f/9
    };
    printf("-> Filtro Definido: Box Blur 3x3\n");

    // 2. Imagen
    int width, height, channels;
    unsigned char* img_data = load_image("img_input/input.png", &width, &height, &channels);
    if (!img_data) return 1;
    printf("-> Imagen Cargada: %d x %d pixeles\n", width, height);

    // 3. OpenCL Init
    CLManager mgr;
    if (!CLManager_Init(&mgr)) return 1; // El manager imprime el hardware detectado
    if (!CLManager_LoadKernel(&mgr, "kernels/convolucion.cl", "conv2d")) return 1;


    // --- SEMANA 1: CPU ---
    imprimir_titulo("FASE 1: PROCESAMIENTO SECUENCIAL (CPU)");

    unsigned char* cpu_result = (unsigned char*)malloc(width * height);

    printf("Procesando... (Esto puede tardar)\n");
    clock_t start = clock();

    // La función hace el trabajo sucio en silencio
    convolucion_secuencial(img_data, cpu_result, width, height, kernel_blur, k_size);

    clock_t end = clock();
    double time_cpu = (double)(end - start) / CLOCKS_PER_SEC;

    // ¡AGREGA ESTA LÍNEA! Convertimos a milisegundos para comparar con la GPU
    double time_cpu_ms = time_cpu * 1000.0;

    printf(">> Completado.\n");
    printf(">> Tiempo CPU: %.4f segundos\n", time_cpu);
    save_image("img_output/resultado_cpu.png", width, height, cpu_result);


    // --- SEMANA 2 y 3: GPU ---
    imprimir_titulo("FASE 2: PROCESAMIENTO PARALELO (GPU)");

    unsigned char* gpu_result = (unsigned char*)malloc(width * height);
    double kernel_time_ms = 0.0;

    printf("Lanzando Kernel OpenCL...\n");
    // Medimos tiempo de host (aproximado) para comparar por ahora
    start = clock();

    // La función hace todo el trabajo de OpenCL en silencio
    convolucion_paralelo(&mgr, img_data, gpu_result, width, height, kernel_blur, k_size, &kernel_time_ms);

    end = clock();
    double total_gpu_time_ms = ((double)(end - start) / CLOCKS_PER_SEC)*1000.0;

    printf(">> Completado.\n");

    // ANÁLISIS DE TIEMPOS (SEMANA 3)
    printf("\n[ANÁLISIS DE RENDIMIENTO]\n");
    printf("  1. Tiempo Total (Host + Transferencias): %10.4f ms\n", total_gpu_time_ms);
    printf("  2. Tiempo Puro de Kernel (GPU Compute):  %10.4f ms\n", kernel_time_ms);
    printf("  3. Overhead (Transferencia RAM<->VRAM):  %10.4f ms\n", total_gpu_time_ms - kernel_time_ms);

    // Cálculo preliminar de Speedup (Semana 3 lo haremos preciso)
    if (kernel_time_ms > 0) {
        printf(">> Speedup Estimado: %.2fx mas rapido\n", time_cpu_ms / kernel_time_ms);
    }

    save_image("img_output/resultado_gpu.png", width, height, gpu_result);


    // --- FINALIZAR ---
    imprimir_titulo("LIMPIEZA Y SALIDA");

    CLManager_Cleanup(&mgr);
    free(cpu_result);
    free(gpu_result);
    free_image(img_data);

    printf("Programa finalizado correctamente.\n");
    return 0;
}