#ifndef CL_MANAGER_H
#define CL_MANAGER_H

#include <CL/cl.h>
#include <stdio.h>

// Estructura para mantener organizado el entorno OpenCL
typedef struct {
    cl_platform_id platform_id;
    cl_device_id device_id;
    cl_context context;
    cl_command_queue queue;
    cl_program program;
    cl_kernel kernel;
} CLManager;

// Inicializa Plataforma, Dispositivo, Contexto y Cola
int CLManager_Init(CLManager* mgr);

// Lee el código fuente .cl, lo compila y extrae el kernel
int CLManager_LoadKernel(CLManager* mgr, const char* filename, const char* kernel_name);

// Libera memoria al terminar
void CLManager_Cleanup(CLManager* mgr);

void printPlatformInfo(cl_platform_id platform);

#endif // CL_MANAGER_H