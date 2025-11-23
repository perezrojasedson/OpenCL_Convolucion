#ifndef OPENCL_UTILS_H
#define OPENCL_UTILS_H

// Usamos el header de C de OpenCL (que ya tienes en tu SDK)
#include <CL/cl.h>
#include <stdio.h> // Para size_t

/*
 * NOTA: Este es un patrón de diseño común en C.
 * En lugar de pasar 'context', 'queue' y 'device' a cada función,
 * los definimos en opencl_utils.c y los declaramos aquí como 'extern'.
 * 'extern' le dice a main.c: "estas variables existen en otra parte,
 * puedes usarlas".
*/
extern cl_context       context;
extern cl_command_queue command_queue;
extern cl_device_id     device_id;

/**
 * @brief TAREA 1: Inicializa OpenCL.
 * Busca una plataforma y un dispositivo (priorizando GPU),
 * crea un contexto y una cola de comandos.
 *
 * Rellena las variables globales (context, command_queue, device_id).
 *
 * @param prefer_platform Un substring para preferir una plataforma (ej. "NVIDIA", "AMD").
 * Si es NULL o "", elige la primera GPU que encuentre.
 */
void initOpenCL(const char* prefer_platform);

/**
 * @brief TAREA 1: Libera los recursos de OpenCL (cola, contexto).
 */
void cleanupOpenCL();


/**
 * @brief TAREA 2: Carga y compila un archivo .cl.
 *
 * @param filename Ruta al kernel (ej. "kernels/convolucion.cl")
 * @param build_log Un buffer de char para guardar el log de compilación.
 * @param log_size El tamaño de ese buffer de log.
 *
 * @return El cl_program compilado. El llamador debe liberarlo con clReleaseProgram.
 */
cl_program createProgramFromFile(const char* filename, char* build_log, size_t log_size);

#endif // OPENCL_UTILS_H