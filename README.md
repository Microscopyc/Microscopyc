# Proyecto Microscopyc

## Diseño de Software para Cómputo científico

## Facultad de Matemática, Astronomía y Física, Universidad Nacional de Córdoba (FaMAF–UNC) Bvd. Medina Allende s/n, Ciudad Universitaria, X5000HUA, Córdoba, Argentina

**Docente:**

Dr. Juan Bautista Cabral

**Estudiantes:**

* Carolina Salinas Domjan

* Augusto Romero

* Horacio José Brizuela

## Motivación
Microscopyc aborda la necesidad de realizar la lectura y el análisis fiable, preciso, reproducible y veloz de imágenes microscópicas en Python. 
El proyecto busca proporcionar herramientas para analizar y clasificar estructuras celulares dentro de imágenes microscópicas y realizar la comparación 
morfológica entre distintas muestras para células con un mismo tipo de clasificación, tales como tamaño, forma, agrupación, distribución. 
Microscopyc se enfoca en ofrecer métodos robustos para la identificación, clasificación y comparación de células y/o microorganismos, facilitando y automatizando
el análisis microscópico aplicable a múltiples áreas de investigación.

## Funcionalidades a lograr con el proyecto

1. Lectura automatizada a partir de la importación o carga local de imágenes.
2. Aprendizaje automático para identificar y categorizar el contenido de imágenes mediante redes neuronales convolucionales. 
3. Manejo de unidades y cantidades físicas.
4. Procesamiento de imágenes con detección y medición de morfología, tamaño, agrupamientos y dimensiones celulares.
5. Registro de datos de salida de procesamientos de imágenes.
6. Cálculo comparativo de parámetros entre registros.
7. Reporte final automático visual según criterios determinados por el usuario, con componentes lingüísticos descriptivos, listas de cantidades y dimensiones observadas. 

## API

``` Python
import microscopyc as myc
import numpy as np
import matplotlib.pyplot as plt

# Leer imágenes del usuario de la carpeta que las contiene
src = 'Imagenes/' # Carpeta con imágenes del usuario
images = myc_image_read()# Lee las imágenes y las enumera

# Identificar las células de interés de las imágenes 
cells = myc.CellIdentiffier(images, class = 'bacteria',type = 'Staphylococcus')

# Visualizar selección de células
fig = plt.figure(figsize=(12, 4))
myc.plot_cells(cells, fig=fig)
plt.show()

# Ejemplos de parámetros posibles a medir una vez finalizada la herramienta:
# Medir y registrar tamaño de células
size_data = myc.load_size('cells')

# Medir la agrupación 
group_data = myc.load_group('cells')

# Y otros

# Realizar la comparación y devolver los resultados
compare = myc.load_comparation(size_data, group_data)
print (compare)
```

## Plan de trabajo

### Escenario 1

Fase | Tarea | Duración total (meses) | Desde | Hasta | Entregables |
-----|-------|------------------------|-------|-------|-------------|
I | Requerimientos | 1 | Oct 2024 | Nov 2024 | Doc. de requerimientos (texto) |
II | Diseño | 2 | Nov 2024 | Ene 2025 | Diagramas y/o scripts |
III | Implementación del código | 3 | Nov 2024 | Feb 2025 | Scripts |
IV |Testing | 2 | Ene 2025 | Mar 2025 | Backlogs, reportes?|
V | Documentación | 4 | Nov 2024 | Mar 2025 | Github completo |
VI | Entrega final | - | | Mar 2025 | GitHub, informe y defensa |

### Escenario 2

Fase | Tarea | Duración total (meses) | Desde | Hasta | Entregables |
-----|-------|------------------------|-------|-------|-------------|
I | Requerimientos | 1 | Oct 2024 | Nov 2024 | Doc. de requerimientos (texto) |
II | Diseño | 5 | Nov 2024 | Abr 2025 | Diagramas y/o scripts |
III | Implementación del código | 6 | Nov 2024 | May 2025 | Scripts |
IV |Testing | 5 | Ene 2025 | Jun 2025 | Backlogs, reportes?|
V | Documentación | 7 | Nov 2024 | Jun 2025 | Github completo |
VI | Entrega final | - | | Jul 2025 | GitHub, informe y defensa |

