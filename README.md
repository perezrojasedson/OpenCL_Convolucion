# Resumen

## Problema de la convolución 2D
La convolución 2D tiene un alto costo computacional, especialmente cuando se aplica a imágenes grandes o con muchos filtros.

## ¿Cómo ayuda OpenCL?
OpenCL permite aprovechar el paralelismo masivo de GPUs, CPUs multinúcleo para acelerar la convolución 2D.

OpenCL divide el cálculo en muchos hilos que se ejecutan en paralelo, de modo que cada hilo calcula el valor de un píxel (o parte de él). En vez de procesar los píxeles de uno en uno (como en CPU secuencial), se procesan miles simultáneamente.

## Definición
La convolución 2D consiste en aplicar una máscara o filtro(Kernel) sobre una imagen, ombinando valores de píxeles vecinos para obtener un nuevo valor en cada posición.

## Elementos involucrados
- **Imagen**: Una matriz 2D de tamaño M x N.
- **Kernel o filtro**: Una matriz de tamaño k x k que define la operación a aplicar.
- **Imagen resultante**: Otra matriz, más pequeña, con nuevos valores.

## Cálculo
La operación de convolución 2D entre una imagen I y unm kernel K se define matematicamente como: 


## Host
El Host es el programa principal que se ejecuta en el CPU.

## Device
Un Device es el hardware que ejecuta el kernel. Puede ser una GPU, CPU o acelerador especializado.

## Kernel
Un Kernel en OpenCL es el programa que se ejecuta en paralelo en el device.

## Work-Item
Un Work-Item es la unidad mínima de ejecución en OpenCL; es decir, un "hilo" o "thread" que ejecuta una instancia del kernel.


