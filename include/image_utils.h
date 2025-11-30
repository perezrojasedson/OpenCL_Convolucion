#ifndef IMAGE_UTILS_H
#define IMAGE_UTILS_H

// Carga una imagen y devuelve un puntero a los datos (unsigned char)
// Rellena width, height y channels con los datos de la imagen cargada.
unsigned char* load_image(const char* filename, int* width, int* height, int* channels);

// Guarda una imagen en formato PNG
void save_image(const char* filename, int width, int height, unsigned char* data);

// Libera la memoria de la imagen cargada
void free_image(unsigned char* data);

#endif // IMAGE_UTILS_H