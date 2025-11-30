// kernels/convolucion.cl

__kernel void conv2d(
    __global const float* input,    // Imagen de entrada (linealizada)
    __global float* output,         // Imagen de salida (linealizada)
    __constant float* kdata,        // La matriz del filtro (Kernel 3x3, 5x5, etc)
    int width,                      // Ancho de la imagen
    int height,                     // Alto de la imagen
    int ksize                       // Tamaño del filtro (ej. 3)
)
{
    // 1. Obtener las coordenadas del pixel que este hilo va a procesar
    int gx = (int)get_global_id(0); // Columna
    int gy = (int)get_global_id(1); // Fila

    // 2. Protección: Si el hilo cae fuera de la imagen, no hace nada
    if (gx >= width || gy >= height) return;

    // 3. Inicializar acumulador
    int khalf = ksize / 2;
    float sum = 0.0f; //

    // 4. Convolución: Iterar sobre la ventana del filtro
    for (int ky = -khalf; ky <= khalf; ky++) {
        for (int kx = -khalf; kx <= khalf; kx++) {

            // Coordenadas del vecino
            int ix = gx + kx;
            int iy = gy + ky; //

            // Manejo de bordes (Clamp to Edge):
            // Si la coordenada sale de la imagen, usamos el pixel del borde más cercano
            if (ix < 0) ix = 0;
            if (ix >= width) ix = width - 1; //
            if (iy < 0) iy = 0;
            if (iy >= height) iy = height - 1; //

            // Leer valor del pixel y peso del filtro
            float pixel = input[iy * width + ix];
            float weight = kdata[(ky + khalf) * ksize + (kx + khalf)]; //

            sum += pixel * weight;
        }
    }

    // 5. Escribir el resultado final en la posición global
    output[gy * width + gx] = sum; //
}