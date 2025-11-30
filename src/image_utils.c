#include "image_utils.h"
#include <stdio.h>

// Definimos la implementación de STB solo aquí para evitar conflictos
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

unsigned char* load_image(const char* filename, int* width, int* height, int* channels) {
    // Forzamos la carga a 1 canal (escala de grises) pasando el último parámetro '1'
    // Esto simplifica la convolución para empezar.
    unsigned char* data = stbi_load(filename, width, height, channels, 1);

    if (data == NULL) {
        printf("Error: No se pudo cargar la imagen %s\n", filename);
        printf("Razon: %s\n", stbi_failure_reason());
    }
    return data;
}

void save_image(const char* filename, int width, int height, unsigned char* data) {
    // Guardamos en formato PNG, 1 canal (Grises)
    // El último parámetro es el "stride" (ancho en bytes), que para 1 byte/pixel es el ancho.
    if (stbi_write_png(filename, width, height, 1, data, width) == 0) {
        printf("Error: No se pudo guardar la imagen en %s\n", filename);
    } else {
        printf("Imagen guardada: %s\n", filename);
    }
}

void free_image(unsigned char* data) {
    stbi_image_free(data);
}