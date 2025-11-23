cmake_minimum_required(VERSION 3.10)
project(Proyecto_OpenCL_Convolucion C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -Wextra")

# ============================================
# Detectar plataforma
# ============================================
message(STATUS "Sistema detectado: ${CMAKE_SYSTEM_NAME}")

# ============================================
# Archivos fuente
# ============================================
set(SOURCES
        src/main.c
        src/opencl_utils.c
        src/stb_all.c
)

set(HEADERS
        include/opencl_utils.h
        include/stb_image.h
        include/stb_image_write.h
)

# ============================================
# Configuración por sistema operativo
# ============================================
if (WIN32)
    # ============================
    # WINDOWS
    # ============================
    message(STATUS "Configurando para Windows...")

    # Ruta al SDK de OpenCL en Windows
    set(MI_OPENCL_SDK_RUTA "F:/OpenCl/OpenCL-SDK-v2025.07.23-Win-x64")
    set(OpenCL_INCLUDE_DIR "${MI_OPENCL_SDK_RUTA}/include")
    set(OpenCL_LIBRARY "${MI_OPENCL_SDK_RUTA}/lib/OpenCL.lib")

    include_directories(${OpenCL_INCLUDE_DIR})
    include_directories(${PROJECT_SOURCE_DIR}/include)
    add_executable(${PROJECT_NAME} ${SOURCES} ${HEADERS})

    target_link_libraries(${PROJECT_NAME} PRIVATE ${OpenCL_LIBRARY})

else()
    # ============================
    # LINUX / UNIX
    # ============================
    message(STATUS "Configurando para Linux/Unix...")

    find_package(OpenCL REQUIRED)

    if(NOT OpenCL_FOUND)
        message(FATAL_ERROR "OpenCL no encontrado. Instala:\n sudo apt install opencl-headers ocl-icd-opencl-dev")
    endif()

    include_directories(
            ${PROJECT_SOURCE_DIR}/include
            ${OpenCL_INCLUDE_DIRS}
    )

    add_executable(${PROJECT_NAME} ${SOURCES} ${HEADERS})

    target_link_libraries(${PROJECT_NAME} PRIVATE
            ${OpenCL_LIBRARIES}
            m   # Linux math library
    )

endif()

# ============================================
# Configurar directorio de salida
# ============================================
set_target_properties(${PROJECT_NAME} PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)

# ============================================
# Copiar kernels automáticamente
# ============================================
add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_directory
        ${CMAKE_SOURCE_DIR}/kernels
        ${CMAKE_BINARY_DIR}/bin/kernels
        COMMENT "Copiando kernels al directorio de salida..."
)

# ============================================
# Mensaje final
# ============================================
message(STATUS "===========================================")
message(STATUS "Proyecto: ${PROJECT_NAME}")
message(STATUS "Sistema detectado: ${CMAKE_SYSTEM_NAME}")
message(STATUS "Compilador C: ${CMAKE_C_COMPILER}")
message(STATUS "Ejecutable: ${CMAKE_BINARY_DIR}/bin/${PROJECT_NAME}")
message(STATUS "===========================================")
