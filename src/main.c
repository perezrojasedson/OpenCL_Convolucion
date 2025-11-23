#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <sys/stat.h>  // Para verificar si archivos existen

#include "opencl_utils.h"
#include "stb_image.h"
#include "stb_image_write.h"

/*
 * CORRECCIÓN: Función para verificar si un archivo existe
 */
int fileExists(const char* filename) {
    struct stat buffer;
    return (stat(filename, &buffer) == 0);
}

/*
 * CORRECCIÓN: Función para buscar la imagen en múltiples rutas
 */
const char* findImagePath(const char* filename) {
    // Rutas posibles donde puede estar la imagen
    static const char* paths[] = {
        "img_input/input.png",           // Desde bin/
        "../img_input/input.png",        // Desde cmake-build-debug/
        "../../img_input/input.png",     // Desde cmake-build-debug/bin/
        NULL
    };

    for (int i = 0; paths[i] != NULL; i++) {
        if (fileExists(paths[i])) {
            printf("✓ Imagen encontrada en: %s\n", paths[i]);
            return paths[i];
        }
    }

    return NULL;
}

/*
 * =================================================================
 * CONVOLUCIÓN SECUENCIAL (CPU)
 * =================================================================
 */
void sequentialConvolution(
    const unsigned char* img_in,
    unsigned char* img_out,
    int w, int h,
    const float* kernel_data,
    int ksize
) {
    printf("\n=== Iniciando Convolución Secuencial (CPU) ===\n");

    int half = ksize / 2;
    long long total_ops = 0; // Contador de operaciones

    for (int y = 0; y < h; y++) {
        // Mostrar progreso cada 10%
        if (y % (h / 10) == 0) {
            printf("  Progreso: %d%%\r", (y * 100) / h);
            fflush(stdout);
        }

        for (int x = 0; x < w; x++) {
            float sum = 0.0f;

            for (int ky = -half; ky <= half; ky++) {
                for (int kx = -half; kx <= half; kx++) {
                    // Manejo de bordes (Clamp to edge)
                    int ix = x + kx;
                    int iy = y + ky;

                    if (ix < 0) ix = 0;
                    if (ix >= w) ix = w - 1;
                    if (iy < 0) iy = 0;
                    if (iy >= h) iy = h - 1;

                    float pixel = (float)img_in[iy * w + ix];
                    float kval  = kernel_data[(ky + half) * ksize + (kx + half)];

                    sum += pixel * kval;
                    total_ops++;
                }
            }

            // Clamp al rango [0, 255]
            if (sum > 255.0f) sum = 255.0f;
            if (sum < 0.0f)   sum = 0.0f;

            img_out[y * w + x] = (unsigned char)sum;
        }
    }

    printf("  Progreso: 100%%\n");
    printf("✓ Operaciones totales: %lld\n", total_ops);
}

/*
 * =================================================================
 * PROGRAMA PRINCIPAL
 * =================================================================
 */
int main() {
    printf("╔═══════════════════════════════════════════════════════╗\n");
    printf("║  Proyecto Convolución Paralela con OpenCL - Semana 1 ║\n");
    printf("╚═══════════════════════════════════════════════════════╝\n");

    // --- Definir el kernel de convolución (Suavizado 3x3) ---
    const int ksize = 3;
    const float kernel_blur[9] = {
        1.0f/9.0f, 1.0f/9.0f, 1.0f/9.0f,
        1.0f/9.0f, 1.0f/9.0f, 1.0f/9.0f,
        1.0f/9.0f, 1.0f/9.0f, 1.0f/9.0f
    };

    printf("\n📊 Kernel de Convolución: Suavizado (Blur) 3x3\n");
    for (int i = 0; i < ksize; i++) {
        printf("   [");
        for (int j = 0; j < ksize; j++) {
            printf(" %.4f", kernel_blur[i * ksize + j]);
        }
        printf(" ]\n");
    }

    // --- TAREA 1.3: Inicializar OpenCL ---
    initOpenCL(NULL);

    // --- TAREA 1.1: Cargar imagen ---
    const char* img_path = findImagePath("input.png");
    if (!img_path) {
        fprintf(stderr, "\n❌ Error: No se encontró 'input.png' en ninguna ubicación.\n");
        fprintf(stderr, "Por favor, coloca una imagen en:\n");
        fprintf(stderr, "  - Proyecto_OpenCL_Convolucion/img_input/input.png\n");
        fprintf(stderr, "\nPuedes crearla con:\n");
        fprintf(stderr, "  convert -size 512x512 xc:gray50 -fill white -draw \"circle 256,256 256,100\" img_input/input.png\n");
        cleanupOpenCL();
        return 1;
    }

    int w, h, c;
    unsigned char* img_in_uc = stbi_load(img_path, &w, &h, &c, 1);

    if (!img_in_uc) {
        fprintf(stderr, "❌ Error: No se pudo cargar la imagen desde: %s\n", img_path);
        fprintf(stderr, "Razón: %s\n", stbi_failure_reason());
        cleanupOpenCL();
        return 1;
    }

    printf("\n📷 Imagen cargada:\n");
    printf("   Dimensiones: %d x %d\n", w, h);
    printf("   Canales originales: %d (forzado a 1 - escala de grises)\n", c);
    printf("   Tamaño total: %d píxeles\n", w * h);

    // --- TAREA 1.2: Ejecución Secuencial (CPU) ---
    unsigned char* img_out_cpu = (unsigned char*)malloc(w * h * sizeof(unsigned char));
    if (!img_out_cpu) {
        fprintf(stderr, "❌ Error: Falló malloc para img_out_cpu\n");
        stbi_image_free(img_in_uc);
        cleanupOpenCL();
        return 1;
    }

    clock_t start_cpu = clock();
    sequentialConvolution(img_in_uc, img_out_cpu, w, h, kernel_blur, ksize);
    clock_t end_cpu = clock();

    double time_cpu_ms = ((double)(end_cpu - start_cpu) / CLOCKS_PER_SEC) * 1000.0;

    printf("\n╔════════════════════════════════════════╗\n");
    printf("║     RESULTADO SEMANA 1 (CPU)          ║\n");
    printf("╠════════════════════════════════════════╣\n");
    printf("║  Tiempo de Ejecución: %10.2f ms  ║\n", time_cpu_ms);
    printf("╚════════════════════════════════════════╝\n");

    // Guardar resultado
    const char* output_paths[] = {
        "img_output/result_cpu.png",
        "../img_output/result_cpu.png",
        "../../img_output/result_cpu.png"
    };

    int saved = 0;
    for (int i = 0; i < 3 && !saved; i++) {
        if (stbi_write_png(output_paths[i], w, h, 1, img_out_cpu, w * 1)) {
            printf("✓ Resultado guardado en: %s\n", output_paths[i]);
            saved = 1;
        }
    }

    if (!saved) {
        fprintf(stderr, "⚠ Advertencia: No se pudo guardar la imagen de salida.\n");
    }

    // --- TAREA 2.1: Compilar Kernel (GPU) ---
    printf("\n=== Compilando Kernel de GPU ===\n");

    char build_log[4096];
    memset(build_log, 0, 4096);

    const char* kernel_paths[] = {
        "kernels/convolucion.cl",
        "../kernels/convolucion.cl",
        "../../kernels/convolucion.cl"
    };

    cl_program program = NULL;
    for (int i = 0; i < 3 && !program; i++) {
        if (fileExists(kernel_paths[i])) {
            printf("✓ Kernel encontrado en: %s\n", kernel_paths[i]);
            program = createProgramFromFile(kernel_paths[i], build_log, 4096);
            break;
        }
    }

    if (strlen(build_log) > 1) {
        printf("\n=== LOG DE COMPILACIÓN ===\n%s\n", build_log);
    }

    if (!program) {
        fprintf(stderr, "❌ Error: El kernel no pudo ser compilado.\n");
        free(img_out_cpu);
        stbi_image_free(img_in_uc);
        cleanupOpenCL();
        return 1;
    }

    cl_int err;
    cl_kernel kernel_cl = clCreateKernel(program, "conv2d", &err);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "❌ Error: clCreateKernel falló con código %d\n", err);
        clReleaseProgram(program);
        free(img_out_cpu);
        stbi_image_free(img_in_uc);
        cleanupOpenCL();
        return 1;
    }

    printf("✓ Kernel 'conv2d' creado exitosamente.\n");
    printf("\n╔════════════════════════════════════════╗\n");
    printf("║    FIN DE SEMANA 2 - ENTREGABLE       ║\n");
    printf("╠════════════════════════════════════════╣\n");
    printf("║  ✓ Convolución CPU implementada       ║\n");
    printf("║  ✓ OpenCL inicializado                ║\n");
    printf("║  ✓ Kernel GPU compilado                ║\n");
    printf("╚════════════════════════════════════════╝\n");

    // --- LIMPIEZA ---
    clReleaseKernel(kernel_cl);
    clReleaseProgram(program);
    cleanupOpenCL();
    free(img_out_cpu);
    stbi_image_free(img_in_uc);

    printf("\n✓ Programa terminado limpiamente.\n");
    return 0;
}