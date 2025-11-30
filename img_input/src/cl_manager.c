#include "cl_manager.h"
#include <stdlib.h>
#include <string.h>

// Función auxiliar de lectura
char* read_file(const char* filename, size_t* size) {
    FILE* fp = fopen(filename, "rb");
    if (!fp) return NULL;
    fseek(fp, 0, SEEK_END);
    *size = ftell(fp);
    rewind(fp);
    char* buffer = (char*)malloc(*size + 1);
    fread(buffer, 1, *size, fp);
    buffer[*size] = '\0';
    fclose(fp);
    return buffer;
}

// Función auxiliar para imprimir info (Privada)
void printPlatformInfo(cl_platform_id platform) {
    char info[128];
    clGetPlatformInfo(platform, CL_PLATFORM_NAME, sizeof(info), info, NULL);
    printf("  Platform: %s\n", info);
    clGetPlatformInfo(platform, CL_PLATFORM_VENDOR, sizeof(info), info, NULL);
    printf("  Vendor:   %s\n", info);
    clGetPlatformInfo(platform, CL_PLATFORM_VERSION, sizeof(info), info, NULL);
    printf("  Version:  %s\n", info);
}

int CLManager_Init(CLManager* mgr) {
    cl_int err;
    cl_uint num_platforms;

    // 1. Detectar Plataformas
    err = clGetPlatformIDs(0, NULL, &num_platforms);
    if (err != CL_SUCCESS || num_platforms == 0) {
        printf("Error: No se encontraron plataformas OpenCL.\n");
        return 0;
    }

    printf("\n=== Plataformas OpenCL Detectadas: %u ===\n", num_platforms);

    cl_platform_id* platforms = (cl_platform_id*)malloc(sizeof(cl_platform_id) * num_platforms);
    clGetPlatformIDs(num_platforms, platforms, NULL);

    // 2. SELECCIÓN INTELIGENTE DE PLATAFORMA
    // Buscamos priorizar GPU real (AMD, NVIDIA, Intel) sobre emuladores (PoCL, Clover)
    int selected_idx = -1;
    char buffer[128];

    for(unsigned int i=0; i<num_platforms; i++) {
        clGetPlatformInfo(platforms[i], CL_PLATFORM_NAME, sizeof(buffer), buffer, NULL);
        printf("Plataforma %u: %s\n", i, buffer);

        // Criterio de selección: Si contiene AMD, NVIDIA o Intel, es prioritaria
        if (selected_idx == -1 && (strstr(buffer, "AMD") || strstr(buffer, "NVIDIA") || strstr(buffer, "Intel"))) {
            selected_idx = i;
        }
    }

    // Si no encontramos ninguna marca conocida, usamos la primera (0) por defecto
    if (selected_idx == -1) selected_idx = 0;

    mgr->platform_id = platforms[selected_idx];

    // Imprimir cuál elegimos
    clGetPlatformInfo(mgr->platform_id, CL_PLATFORM_NAME, sizeof(buffer), buffer, NULL);
    printf("--> SELECCIONADA: %s\n", buffer);

    free(platforms);

    // 3. Obtener Dispositivo (Intentar GPU primero)
    err = clGetDeviceIDs(mgr->platform_id, CL_DEVICE_TYPE_GPU, 1, &mgr->device_id, NULL);
    if (err != CL_SUCCESS) {
        printf("Aviso: La plataforma seleccionada no tiene GPU disponible. Usando CPU...\n");
        err = clGetDeviceIDs(mgr->platform_id, CL_DEVICE_TYPE_CPU, 1, &mgr->device_id, NULL);
    }

    if (err != CL_SUCCESS) {
        printf("Error: No se encontró ningún dispositivo válido en la plataforma.\n");
        return 0;
    }

    // Mostrar info del dispositivo final
    char name[128];
    cl_uint units;
    clGetDeviceInfo(mgr->device_id, CL_DEVICE_NAME, sizeof(name), name, NULL);
    clGetDeviceInfo(mgr->device_id, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(units), &units, NULL);

    printf("\n=== Dispositivo Seleccionado ===\n");
    printf("  Device:   %s\n", name);
    printf("  Compute Units: %u\n", units);

    // 4. Contexto y Cola
    mgr->context = clCreateContext(NULL, 1, &mgr->device_id, NULL, NULL, &err);

    //Agregamos CL_QUEUE_PROFILING_ENABLE
    // Esto le dice a la GPU: "Guarda los tiempos de inicio y fin de cada comando"
    cl_command_queue_properties properties = CL_QUEUE_PROFILING_ENABLE;

    mgr->queue = clCreateCommandQueue(mgr->context, mgr->device_id,properties, &err);

    if (err == CL_SUCCESS) printf("✓ OpenCL inicializado exitosamente.\n");
    return (err == CL_SUCCESS);
}

int CLManager_LoadKernel(CLManager* mgr, const char* filename, const char* kernel_name) {
    cl_int err;
    size_t src_size;

    printf("\n=== Compilando Kernel ===\n");
    char* source_str = read_file(filename, &src_size);
    if (!source_str) return 0;

    printf("Archivo: %s (%zu bytes)\n", filename, src_size);

    mgr->program = clCreateProgramWithSource(mgr->context, 1, (const char**)&source_str, &src_size, &err);
    free(source_str);

    err = clBuildProgram(mgr->program, 1, &mgr->device_id, NULL, NULL, NULL);

    if (err != CL_SUCCESS) {
        // Log de error
        char log[4096];
        clGetProgramBuildInfo(mgr->program, mgr->device_id, CL_PROGRAM_BUILD_LOG, sizeof(log), log, NULL);
        printf("Error Build:\n%s\n", log);
        return 0;
    }

    printf("✓ Kernel compilado exitosamente.\n");

    mgr->kernel = clCreateKernel(mgr->program, kernel_name, &err);
    return (err == CL_SUCCESS);
}

void CLManager_Cleanup(CLManager* mgr) {
    if(mgr->kernel) clReleaseKernel(mgr->kernel);
    if(mgr->program) clReleaseProgram(mgr->program);
    if(mgr->queue) clReleaseCommandQueue(mgr->queue);
    if(mgr->context) clReleaseContext(mgr->context);
    printf("Recursos de OpenCL liberados.\n");
}