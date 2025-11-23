#include "opencl_utils.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/*
 * Definición de las variables globales
 */
cl_context       context = NULL;
cl_command_queue command_queue = NULL;
cl_device_id     device_id = NULL;
cl_platform_id   platform_id = NULL; // Agregamos esto para referencia

/*
 * Función auxiliar para imprimir info de plataforma
 */
void printPlatformInfo(cl_platform_id platform) {
    char name[128];
    char vendor[128];
    char version[128];

    clGetPlatformInfo(platform, CL_PLATFORM_NAME, sizeof(name), name, NULL);
    clGetPlatformInfo(platform, CL_PLATFORM_VENDOR, sizeof(vendor), vendor, NULL);
    clGetPlatformInfo(platform, CL_PLATFORM_VERSION, sizeof(version), version, NULL);

    printf("  Platform: %s\n", name);
    printf("  Vendor:   %s\n", vendor);
    printf("  Version:  %s\n", version);
}

/*
 * Función auxiliar para imprimir info de dispositivo
 */
void printDeviceInfo(cl_device_id device) {
    char name[128];
    cl_device_type type;
    cl_uint compute_units;

    clGetDeviceInfo(device, CL_DEVICE_NAME, sizeof(name), name, NULL);
    clGetDeviceInfo(device, CL_DEVICE_TYPE, sizeof(type), &type, NULL);
    clGetDeviceInfo(device, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(compute_units), &compute_units, NULL);

    printf("  Device:   %s\n", name);
    printf("  Type:     %s\n", (type == CL_DEVICE_TYPE_GPU) ? "GPU" :
                               (type == CL_DEVICE_TYPE_CPU) ? "CPU" : "Other");
    printf("  Compute Units: %u\n", compute_units);
}

/*
 * CORRECCIÓN 1: Inicialización mejorada que evita Clover
 */
void initOpenCL(const char* prefer_platform) {
    cl_uint num_platforms = 0;
    cl_int err;

    // 1. Obtener TODAS las plataformas disponibles
    err = clGetPlatformIDs(0, NULL, &num_platforms);
    if (err != CL_SUCCESS || num_platforms == 0) {
        fprintf(stderr, "Error: No se encontraron plataformas OpenCL.\n");
        exit(EXIT_FAILURE);
    }

    printf("\n=== Plataformas OpenCL Detectadas: %u ===\n", num_platforms);

    cl_platform_id* platforms = (cl_platform_id*)malloc(sizeof(cl_platform_id) * num_platforms);
    clGetPlatformIDs(num_platforms, platforms, NULL);

    // 2. Buscar la mejor plataforma (EVITAR Clover)
    cl_platform_id best_platform = NULL;
    char platform_name[128];

    for (cl_uint i = 0; i < num_platforms; i++) {
        clGetPlatformInfo(platforms[i], CL_PLATFORM_NAME, sizeof(platform_name), platform_name, NULL);

        printf("\nPlataforma %u:\n", i);
        printPlatformInfo(platforms[i]);

        // ESTRATEGIA: Preferir PoCL sobre Clover
        if (strstr(platform_name, "Portable Computing Language") != NULL) {
            best_platform = platforms[i];
            printf("  --> SELECCIONADA (PoCL - Más estable)\n");
            break;
        }
        // Si no hay PoCL, usar cualquier otra que NO sea Clover
        else if (strstr(platform_name, "Clover") == NULL && best_platform == NULL) {
            best_platform = platforms[i];
        }
    }

    if (best_platform == NULL) {
        // Si solo hay Clover, usarla (con advertencia)
        best_platform = platforms[0];
        printf("\nADVERTENCIA: Solo Clover disponible, puede tener problemas.\n");
    }

    platform_id = best_platform;

    // 3. Buscar dispositivos en la plataforma seleccionada
    cl_uint num_devices = 0;

    // Primero intentar GPU
    err = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_GPU, 1, &device_id, &num_devices);
    if (err != CL_SUCCESS || num_devices == 0) {
        printf("\nNo se encontró GPU, buscando CPU...\n");
        err = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_CPU, 1, &device_id, &num_devices);
        if (err != CL_SUCCESS) {
            fprintf(stderr, "Error: No se encontró ningún dispositivo OpenCL.\n");
            free(platforms);
            exit(EXIT_FAILURE);
        }
    }

    printf("\n=== Dispositivo Seleccionado ===\n");
    printDeviceInfo(device_id);

    // 4. Crear Contexto
    context = clCreateContext(NULL, 1, &device_id, NULL, NULL, &err);
    if (err != CL_SUCCESS || context == NULL) {
        fprintf(stderr, "Error: No se pudo crear el contexto OpenCL (código %d).\n", err);
        free(platforms);
        exit(EXIT_FAILURE);
    }

    // 5. Crear Cola de Comandos (compatible con OpenCL 1.x y 2.x)
    #ifdef CL_VERSION_2_0
        cl_command_queue_properties props[] = {CL_QUEUE_PROPERTIES, 0, 0};
        command_queue = clCreateCommandQueueWithProperties(context, device_id, props, &err);
    #else
        command_queue = clCreateCommandQueue(context, device_id, 0, &err);
    #endif

    if (err != CL_SUCCESS || command_queue == NULL) {
        fprintf(stderr, "Error: No se pudo crear la cola de comandos (código %d).\n", err);
        clReleaseContext(context);
        free(platforms);
        exit(EXIT_FAILURE);
    }

    free(platforms);
    printf("\n✓ OpenCL inicializado exitosamente.\n");
}

/*
 * Implementación de cleanupOpenCL
 */
void cleanupOpenCL() {
    if (command_queue) {
        clReleaseCommandQueue(command_queue);
        command_queue = NULL;
    }
    if (context) {
        clReleaseContext(context);
        context = NULL;
    }
    printf("Recursos de OpenCL liberados.\n");
}

/*
 * CORRECCIÓN 2: Manejo mejorado de errores en compilación de kernel
 */

cl_program createProgramFromFile(const char* filename, char* build_log, size_t log_size) {
    // 1. Lectura de archivo
    FILE* fp = fopen(filename, "rb"); // abrir en modo binario para no tocar bytes
    if (!fp) {
        fprintf(stderr, "Error: No se pudo abrir el archivo de kernel: %s\n", filename);
        return NULL;
    }

    if (fseek(fp, 0, SEEK_END) != 0) {
        fclose(fp);
        fprintf(stderr, "Error: fseek falló al medir el archivo: %s\n", filename);
        return NULL;
    }
    long ftell_size = ftell(fp);
    if (ftell_size < 0) {
        fclose(fp);
        fprintf(stderr, "Error: ftell falló en: %s\n", filename);
        return NULL;
    }
    size_t file_size = (size_t)ftell_size;
    rewind(fp);

    char* source_str = (char*)malloc(file_size + 1);
    if (!source_str) {
        fclose(fp);
        fprintf(stderr, "Error: Falló malloc para el código fuente del kernel\n");
        return NULL;
    }

    size_t bytes_read = fread(source_str, 1, file_size, fp);
    fclose(fp);
    source_str[bytes_read] = '\0'; // null-terminate para seguridad

    printf("\n=== Compilando Kernel ===\n");
    printf("Archivo: %s (%zu bytes)\n", filename, bytes_read);

    // 1.b Detectar y eliminar BOM UTF-8 si existe
    size_t start = 0;
    if (bytes_read >= 3 &&
        (unsigned char)source_str[0] == 0xEF &&
        (unsigned char)source_str[1] == 0xBB &&
        (unsigned char)source_str[2] == 0xBF) {
        start = 3;
        printf("NOTICE: Se detectó BOM UTF-8 en el kernel; será ignorado.\n");
    }
    const char* source_ptr = source_str + start;
    size_t bytes_no_bom = bytes_read - start;

    // DEBUG: Guardar el código fuente para inspección (antes de free)
    FILE* debug_fp = fopen("/tmp/kernel_source_debug.cl", "wb");
    if (debug_fp) {
        fwrite(source_ptr, 1, bytes_no_bom, debug_fp);
        fclose(debug_fp);
        printf("DEBUG: Código fuente guardado en /tmp/kernel_source_debug.cl\n");
    } else {
        printf("WARNING: No se pudo abrir /tmp/kernel_source_debug.cl para debug.\n");
    }

    // 2. Creación del programa (usar la longitud correcta)
    cl_int err;
    const char* srcs[1] = { source_ptr };
    const size_t src_lens[1] = { bytes_no_bom };

    cl_program program = clCreateProgramWithSource(context, 1, srcs, src_lens, &err);
    free(source_str); // ahora sí liberamos

    if (err != CL_SUCCESS || program == NULL) {
        fprintf(stderr, "Error: clCreateProgramWithSource falló (código %d).\n", err);
        return NULL;
    }

    // 3. Compilación del programa
    // Quitar -Werror: no queremos que advertencias rompan la compilación.
    const char* build_options = "-cl-std=CL1.2";
    err = clBuildProgram(program, 1, &device_id, build_options, NULL, NULL);

    // 4. Obtener el Build Log SIEMPRE
    if (build_log && log_size > 0) {
        size_t log_len = 0;
        clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_len);

        if (log_len > 1 && log_len < log_size) {
            clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, log_len, build_log, NULL);
            build_log[log_len] = '\0';
        } else if (log_len > 1) {
            // si el log es más grande que el buffer, truncarlo limpiamente
            clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, log_size - 1, build_log, NULL);
            build_log[log_size - 1] = '\0';
        } else {
            build_log[0] = '\0';
        }
    }

    if (err != CL_SUCCESS) {
        fprintf(stderr, "\n❌ Error: Compilación del kernel falló (código %d).\n", err);
        fprintf(stderr, "Revisa el log de compilación arriba.\n");
        clReleaseProgram(program);
        return NULL;
    }

    printf("✓ Kernel compilado exitosamente.\n");
    return program;
}
