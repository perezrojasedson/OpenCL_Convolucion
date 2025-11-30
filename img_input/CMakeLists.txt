cmake_minimum_required(VERSION 3.10)
project(Proyecto_OpenCL_Convolucion C) # <--- Lenguaje C

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -Wextra")

# ============================================
# 1. Directorios de Cabeceras (Headers)
# ============================================
# Aquí buscará tus archivos .h (cl_manager.h, image_utils.h, etc.)
include_directories(${PROJECT_SOURCE_DIR}/include)

# ============================================
# 2. Configuración de OpenCL (Windows/Linux)
# ============================================
if (WIN32)
    message(STATUS "Configurando para Windows...")

    # TU RUTA ESPECÍFICA DEL SDK
    set(MI_OPENCL_SDK_RUTA "F:/OpenCl/OpenCL-SDK-v2025.07.23-Win-x64")

    # Añadir includes y librerías del SDK
    include_directories("${MI_OPENCL_SDK_RUTA}/include")
    link_directories("${MI_OPENCL_SDK_RUTA}/lib")

    # Variable para enlazar más abajo
    set(OpenCL_LIBRARY "${MI_OPENCL_SDK_RUTA}/lib/OpenCL.lib")
else()
    # Configuración automática para Linux
    message(STATUS "Configurando para Linux/Unix...")
    find_package(OpenCL REQUIRED)
endif()

# ============================================
# 3. Archivos Fuente
# ============================================
# Busca automáticamente todos los archivos .c en la carpeta src
file(GLOB SOURCES "src/*.c")

# ============================================
# 4. Crear Ejecutable
# ============================================
add_executable(${PROJECT_NAME} ${SOURCES})

# ============================================
# 5. Vincular Librerías (Linker)
# ============================================
if (WIN32)
    target_link_libraries(${PROJECT_NAME} PRIVATE ${OpenCL_LIBRARY})
else()
    target_link_libraries(${PROJECT_NAME} PRIVATE OpenCL::OpenCL m)
endif()

# ============================================
# 6. Configurar Salida y Recursos
# ============================================
# Ponemos el ejecutable en una carpeta 'bin' para mantener orden
set_target_properties(${PROJECT_NAME} PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)

# COPIA AUTOMÁTICA: Esto es vital.
# Copia las carpetas 'kernels' e 'img_input' junto al ejecutable (.exe)
# cada vez que compilas. Así el programa encuentra los archivos.
add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_directory
        ${CMAKE_SOURCE_DIR}/kernels
        ${CMAKE_BINARY_DIR}/bin/kernels
        COMMAND ${CMAKE_COMMAND} -E copy_directory
        ${CMAKE_SOURCE_DIR}/img_input
        ${CMAKE_BINARY_DIR}/bin/img_input
        COMMENT "Copiando kernels e imagenes al directorio de salida..."
)