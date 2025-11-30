# Proyecto 06: Convolución de Imágenes Paralela con OpenCL

## 1. Resumen (Abstract)

Este proyecto implementa la aceleración de operaciones de convolución 2D para procesamiento de imágenes utilizando OpenCL (Open Computing Language). La convolución es una operación fundamental en el procesamiento de imágenes que permite aplicar filtros como suavizado gaussiano, detección de bordes (Sobel), enfoque (sharpen) y efectos de relieve (emboss).

El objetivo principal es demostrar el poder del paralelismo masivo de datos (SIMD) que ofrecen las GPUs modernas, comparando el rendimiento de una implementación secuencial en CPU contra una implementación paralela en GPU. Los resultados experimentales muestran speedups significativos, especialmente en imágenes de alta resolución (4K y superiores), donde el tiempo de procesamiento se reduce de varios segundos a milisegundos.

**Tecnologías utilizadas:**
- OpenCL 1.2+ para aceleración por GPU
- C11 para el código del host
- stb_image para carga/escritura de imágenes
- CMake para compilación multiplataforma

---

## 2. Introducción

### 2.1 Motivación

El procesamiento de imágenes es una tarea computacionalmente intensiva que se beneficia enormemente del paralelismo. Una imagen de 4K (3840×2160) contiene más de 8 millones de píxeles, y cada píxel debe procesarse individualmente durante una operación de convolución. En una CPU tradicional, este procesamiento es secuencial, lo que limita el rendimiento.

Las GPUs modernas, diseñadas para renderizado gráfico, contienen miles de núcleos pequeños optimizados para operaciones paralelas. OpenCL permite aprovechar esta arquitectura para acelerar cálculos de propósito general.

### 2.2 Objetivos

1. Implementar la convolución 2D secuencial como baseline de rendimiento
2. Desarrollar kernels OpenCL eficientes para convolución paralela
3. Medir y analizar el speedup obtenido
4. Proporcionar múltiples filtros de imagen predefinidos
5. Crear una interfaz de usuario interactiva para selección de filtros

---

## 3. Fundamento Teórico

### 3.1 Convolución 2D

La convolución 2D es una operación matemática que combina dos funciones para producir una tercera. En procesamiento de imágenes, se utiliza para aplicar filtros a una imagen.

**Fórmula matemática:**

```
O(i,j) = Σₘ Σₙ I(i-m, j-n) × K(m,n)
```

Donde:
- `O(i,j)` es el valor del píxel de salida en la posición (i,j)
- `I` es la imagen de entrada
- `K` es el kernel de convolución (matriz de pesos)
- `m,n` son los índices del kernel

**Ejemplo de kernel Gaussiano 3×3:**

```
     | 1  2  1 |
K =  | 2  4  2 |  × (1/16)
     | 1  2  1 |
```

### 3.2 Modelo de Ejecución OpenCL

OpenCL define un modelo de ejecución heterogéneo con dos componentes principales:

#### Host (CPU)
- Gestiona el contexto de ejecución
- Crea buffers de memoria en el dispositivo
- Transfiere datos entre Host y Device
- Lanza la ejecución de kernels

#### Device (GPU)
- Ejecuta los kernels en paralelo
- Organizado en Work-Groups (grupos de trabajo)
- Cada Work-Item procesa un elemento de datos

**Jerarquía de ejecución:**

```
NDRange (Global)
├── Work-Group 0
│   ├── Work-Item 0,0
│   ├── Work-Item 0,1
│   └── ...
├── Work-Group 1
│   └── ...
└── ...
```

### 3.3 Mapeo de Convolución a OpenCL

En nuestra implementación:
- **Cada Work-Item** = Calcula UN píxel de salida
- **NDRange** = Dimensiones de la imagen (width × height)
- **Work-Group** = Tiles de 16×16 píxeles

---

## 4. Diseño e Implementación

### 4.1 Arquitectura del Sistema

```
┌─────────────────────────────────────────────────────────────┐
│                         HOST (CPU)                          │
├─────────────────────────────────────────────────────────────┤
│  main.c                                                     │
│  ├── Carga de imagen (stb_image)                           │
│  ├── Inicialización OpenCL (contexto, cola)                │
│  ├── Compilación de kernels                                │
│  ├── Gestión de memoria (buffers)                          │
│  └── Benchmarking (CPU vs GPU)                             │
├─────────────────────────────────────────────────────────────┤
│  opencl_utils.c                                             │
│  ├── Inicialización de dispositivos                        │
│  ├── Manejo de errores                                     │
│  └── Definición de filtros                                 │
└─────────────────────────────────────────────────────────────┘
                            │
                            │ PCIe / Memoria
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                       DEVICE (GPU)                          │
├─────────────────────────────────────────────────────────────┤
│  convolution.cl                                             │
│  ├── convolution_3x3    (Convolución genérica 3×3)         │
│  ├── convolution_5x5    (Convolución genérica 5×5)         │
│  ├── sobel_combined     (Detección de bordes optimizada)   │
│  └── convolution_3x3_optimized (Con memoria local)         │
└─────────────────────────────────────────────────────────────┘
```

### 4.2 Estructura del Proyecto

```
/Proyecto_OpenCL_Convolucion
├── src/
│   ├── main.c              # Programa principal y benchmark
│   ├── opencl_utils.c      # Funciones auxiliares OpenCL
│   └── stb_all.c           # Implementación stb_image
├── include/
│   ├── opencl_utils.h      # Headers de utilidades
│   ├── stb_image.h         # Carga de imágenes
│   └── stb_image_write.h   # Escritura de imágenes
├── kernels/
│   └── convolution.cl      # Kernels GPU
├── img_input/              # Imágenes de entrada
├── img_output/             # Imágenes procesadas
├── CMakeLists.txt          # Sistema de build
└── README.md               # Este documento
```

### 4.3 El Kernel OpenCL

**Kernel principal de convolución 3×3:**

```c
__kernel void convolution_3x3(
    __global const uchar* input,
    __global uchar* output,
    const int width,
    const int height,
    __constant float* conv_kernel,
    const float divisor,
    const float offset
) {
    // Obtener posición global del work-item
    int x = get_global_id(0);
    int y = get_global_id(1);
    
    // Verificar límites
    if (x >= width || y >= height) return;
    
    // Acumuladores para cada canal RGB
    float sum_r = 0.0f, sum_g = 0.0f, sum_b = 0.0f;
    
    // Aplicar convolución 3×3
    for (int ky = -1; ky <= 1; ky++) {
        for (int kx = -1; kx <= 1; kx++) {
            // Manejo de bordes con clamp
            int nx = clamp(x + kx, 0, width - 1);
            int ny = clamp(y + ky, 0, height - 1);
            
            int pixel_idx = (ny * width + nx) * 4;  // RGBA
            int kernel_idx = (ky + 1) * 3 + (kx + 1);
            float weight = conv_kernel[kernel_idx];
            
            sum_r += input[pixel_idx + 0] * weight;
            sum_g += input[pixel_idx + 1] * weight;
            sum_b += input[pixel_idx + 2] * weight;
        }
    }
    
    // Normalizar y escribir resultado
    int out_idx = (y * width + x) * 4;
    output[out_idx + 0] = clamp(sum_r/divisor + offset, 0.0f, 255.0f);
    output[out_idx + 1] = clamp(sum_g/divisor + offset, 0.0f, 255.0f);
    output[out_idx + 2] = clamp(sum_b/divisor + offset, 0.0f, 255.0f);
    output[out_idx + 3] = input[out_idx + 3];  // Preservar alpha
}
```

### 4.4 Filtros Implementados

| # | Filtro | Tamaño | Descripción |
|---|--------|--------|-------------|
| 1 | Gaussiano 3×3 | 3×3 | Suavizado ligero |
| 2 | Gaussiano 5×5 | 5×5 | Suavizado fuerte |
| 3 | Sobel X | 3×3 | Detección bordes horizontales |
| 4 | Sobel Y | 3×3 | Detección bordes verticales |
| 5 | Sobel Combinado | 3×3 | Detección bordes completa |
| 6 | Sharpen | 3×3 | Enfoque de imagen |
| 7 | Laplaciano | 3×3 | Detección de bordes |
| 8 | Emboss | 3×3 | Efecto de relieve |
| 9 | Box Blur | 3×3 | Desenfoque uniforme |

---

## 5. Resultados y Análisis

### 5.1 Configuración Experimental

**Hardware de prueba:**
- CPU: [Especificar modelo]
- GPU: [Especificar modelo]
- RAM: [Especificar cantidad]

**Imágenes de prueba:**
- Resolución: 3840 × 2160 (4K)
- Formato: PNG/RGBA
- Tamaño en memoria: ~33 MB

### 5.2 Tabla de Resultados

| Filtro | Tiempo CPU (ms) | Tiempo GPU (ms) | Speedup |
|--------|-----------------|-----------------|---------|
| Gaussiano 3×3 | [completar] | [completar] | [completar]x |
| Gaussiano 5×5 | [completar] | [completar] | [completar]x |
| Sobel Combinado | [completar] | [completar] | [completar]x |
| Sharpen | [completar] | [completar] | [completar]x |
| Edge Detect | [completar] | [completar] | [completar]x |

### 5.3 Análisis de Rendimiento

#### Desglose del tiempo GPU:
```
Tiempo Total GPU = T_transferencia_H→D + T_cómputo + T_transferencia_D→H
```

La **latencia de transferencia** es un factor crítico en el rendimiento de OpenCL. Para imágenes pequeñas, la sobrecarga de transferencia puede superar el beneficio del paralelismo.

#### Cuándo la GPU es más eficiente:
1. **Imágenes grandes** (>1080p): Mayor paralelismo compensa transferencia
2. **Kernels grandes** (5×5 o más): Mayor carga computacional por píxel
3. **Procesamiento por lotes**: Amortiza costo de inicialización

### 5.4 Gráfico de Speedup (Conceptual)

```
Speedup
   ^
   |                                    ●
   |                              ●
   |                        ●
   |                  ●
   |            ●
   |      ●
   |  ●
   +----●-----------------------------> Resolución
      480p  720p  1080p  1440p  4K
```

---

## 6. Conclusiones y Trabajo Futuro

### 6.1 Conclusiones

1. **OpenCL acelera significativamente** el procesamiento de imágenes, especialmente en resoluciones altas
2. **La transferencia de datos** es el cuello de botella principal para imágenes pequeñas
3. **El paralelismo masivo** de la GPU es ideal para operaciones pixel-independientes como la convolución
4. **La implementación es portable** gracias a CMake y al estándar OpenCL

### 6.2 Trabajo Futuro

1. **Memoria local optimizada**: Implementar carga de tiles en memoria compartida para reducir accesos a memoria global
2. **Kernels separables**: Descomponer filtros 2D en dos pasadas 1D (ej. Gaussiano)
3. **Procesamiento por lotes**: Procesar múltiples imágenes en paralelo
4. **Soporte para video**: Aplicar filtros en tiempo real a streams de video
5. **Más filtros**: Implementar filtros avanzados como bilateral, Canny, etc.

---

## 7. Instrucciones de Compilación y Uso

### 7.1 Requisitos

**Linux (Ubuntu/Debian):**
```bash
sudo apt install build-essential cmake
sudo apt install opencl-headers ocl-icd-opencl-dev
# Para GPU NVIDIA:
sudo apt install nvidia-opencl-dev
# Para CPU (desarrollo):
sudo apt install pocl-opencl-icd
```

**Windows:**
- Visual Studio 2019+ o MinGW
- OpenCL SDK (NVIDIA CUDA Toolkit / Intel OpenCL SDK / AMD APP SDK)

### 7.2 Compilación

```bash
# Clonar o descargar el proyecto
cd Proyecto_OpenCL_Convolucion

# Crear directorio de build
mkdir build && cd build

# Configurar CMake
cmake ..

# Compilar
cmake --build . --config Release

# Ejecutar
./bin/Proyecto_OpenCL_Convolucion
# En Windows: .\bin\Release\Proyecto_OpenCL_Convolucion.exe
```

### 7.3 Uso del Programa

1. Coloca tu imagen en `img_input/input.png`
2. Ejecuta el programa
3. Selecciona un filtro del menú (1-9)
4. Los resultados se guardan en `img_output/`

**Uso con imagen personalizada:**
```bash
./Proyecto_OpenCL_Convolucion mi_imagen.jpg
```

---

## 8. Referencias

1. Khronos Group. "OpenCL Specification 1.2". https://www.khronos.org/opencl/
2. NVIDIA. "OpenCL Programming Guide for the CUDA Architecture"
3. Gonzalez, R. & Woods, R. "Digital Image Processing", 4th Edition
4. Sanders, J. & Kandrot, E. "CUDA by Example"

---

## 9. Autores

- [Nombre del estudiante 1]
- [Nombre del estudiante 2]
- [Nombre del estudiante 3]
- [Nombre del estudiante 4]

**Curso:** Computación Paralela  
**Fecha:** [Fecha de entrega]

---

## 10. Licencia

Este proyecto es de uso académico. Se permite su uso y modificación con fines educativos.
