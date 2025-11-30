#include "convolucion_paralelo.h"
#include <stdio.h>
#include <stdlib.h>

void convolucion_paralelo(CLManager* mgr, const unsigned char* input, unsigned char* output,
                              int width, int height, const float* filter, int k_size, double* kernel_time_ms) {

    cl_int err;
    cl_event prof_event; //

    // 1. Convertir entrada a float (Mejor precisión en GPU)
    // ----------------------------------------------------
    size_t num_pixels = width * height;
    size_t img_size_bytes = num_pixels * sizeof(float);

    float* host_input_float = (float*)malloc(img_size_bytes);
    float* host_output_float = (float*)malloc(img_size_bytes);

    if (!host_input_float || !host_output_float) {
        printf("Error: Fallo de memoria en conversion float.\n");
        return;
    }

    for(size_t i = 0; i < num_pixels; i++) {
        host_input_float[i] = (float)input[i];
    }

    // 2. Crear Buffers en la GPU
    // ----------------------------------------------------
    // Buffer Entrada (Copiamos desde el host inmediatamente)
    cl_mem d_input = clCreateBuffer(mgr->context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                    img_size_bytes, host_input_float, &err);

    // Buffer Salida (Solo escritura)
    cl_mem d_output = clCreateBuffer(mgr->context, CL_MEM_WRITE_ONLY,
                                     img_size_bytes, NULL, &err);

    // Buffer Filtro/Kernel (Copiamos desde el host)
    cl_mem d_filter = clCreateBuffer(mgr->context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                     sizeof(float) * k_size * k_size, (void*)filter, &err);

    if (err != CL_SUCCESS) {
        printf("Error creando buffers OpenCL (Code %d)\n", err);
        goto cleanup; // Salto a limpieza
    }

    // 3. Configurar Argumentos del Kernel
    // ----------------------------------------------------
    // El orden debe coincidir con __kernel void conv2d en convolucion.cl
    err  = clSetKernelArg(mgr->kernel, 0, sizeof(cl_mem), &d_input);
    err |= clSetKernelArg(mgr->kernel, 1, sizeof(cl_mem), &d_output);
    err |= clSetKernelArg(mgr->kernel, 2, sizeof(cl_mem), &d_filter);
    err |= clSetKernelArg(mgr->kernel, 3, sizeof(int), &width);
    err |= clSetKernelArg(mgr->kernel, 4, sizeof(int), &height);
    err |= clSetKernelArg(mgr->kernel, 5, sizeof(int), &k_size);

    if (err != CL_SUCCESS) {
        printf("Error configurando argumentos del kernel.\n");
        goto cleanup_mem;
    }

    // 4. Ejecutar Kernel
    // Definimos cuántos hilos queremos lanzar.
    // "Para toda la imagen" significa: Un hilo por cada píxel (ancho x alto).
    size_t global_work_size[2] = { (size_t)width, (size_t)height };

    // Esta función es la que menciona tu requisito:
    err = clEnqueueNDRangeKernel(
        mgr->queue,       // La cola de comandos
        mgr->kernel,      // El kernel configurado
        2,                // Dimensiones (2D: X e Y)
        NULL,             // Offset global (no usado)
        global_work_size, // <--- AQUÍ definimos el tamaño total (Width x Height)
        NULL,             // Local work size (NULL = deja que OpenCL decida)
        0, NULL,
        &prof_event     // Eventos (para la semana 3)
    );

    if (err != CL_SUCCESS) {
        printf("Error al encolar el kernel (Code %d)\n", err);
        goto cleanup_mem;
    }

    // Esperar a que termine para poder leer los tiempos
    clWaitForEvents(1, &prof_event);

    // --- PROFILING (NUEVO) ---
    cl_ulong time_start, time_end;
    clGetEventProfilingInfo(prof_event, CL_PROFILING_COMMAND_START, sizeof(time_start), &time_start, NULL);
    clGetEventProfilingInfo(prof_event, CL_PROFILING_COMMAND_END, sizeof(time_end), &time_end, NULL);

    // Calcular tiempo en nanosegundos y convertir a milisegundos
    double nanoSeconds = (double)(time_end - time_start);
    *kernel_time_ms = nanoSeconds / 1000000.0;
    // -------------------------
    // Esperar a que termine
    clFinish(mgr->queue);

    // 5. Leer Resultados (GPU -> CPU)
    // ----------------------------------------------------
    err = clEnqueueReadBuffer(mgr->queue, d_output, CL_TRUE, 0, img_size_bytes, host_output_float, 0, NULL, NULL);

    if (err != CL_SUCCESS) {
        printf("Error leyendo resultados de la GPU.\n");
    }

    // 6. Convertir float de vuelta a unsigned char
    // ----------------------------------------------------
    for(size_t i = 0; i < num_pixels; i++) {
        float val = host_output_float[i];
        if(val < 0) val = 0;
        if(val > 255) val = 255;
        output[i] = (unsigned char)val;
    }

    // --- Limpieza de recursos locales de esta función ---
cleanup_mem:
    clReleaseEvent(prof_event);
    if(d_input) clReleaseMemObject(d_input);
    if(d_output) clReleaseMemObject(d_output);
    if(d_filter) clReleaseMemObject(d_filter);

cleanup:
    if(host_input_float) free(host_input_float);
    if(host_output_float) free(host_output_float);
}