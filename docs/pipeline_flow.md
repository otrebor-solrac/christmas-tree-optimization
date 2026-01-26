# Flujo del Pipeline de Optimización

## Comando de Entrada

```bash
python3 run_rust.py --target N [--strategy S] [--optimizer O] [--gen G] [--fine-tune F]
```

### Parámetros

| Parámetro | Default | Opciones | Descripción |
|-----------|---------|----------|-------------|
| `--target` | requerido | 1-200 | Número de árboles objetivo |
| `--strategy` | `all` | `all`, `mosaic`, `zipper`, `pruning`, `incremental`, `none` | Estrategia geométrica |
| `--optimizer` | `ga` | `ga`, `cmaes`, `sa` | Optimizador (si estrategias fallan) |
| `--gen` | 10 | entero | Generaciones del optimizador |
| `--fine-tune` | 0 | entero | Iteraciones de Fine-Tuning |
| `--steps` | 50 | entero | Pasos de gravedad |
| `--pop` | 100 | entero | Tamaño de población (solo GA) |

---

## Diagrama de Flujo

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         ENTRADA                                          │
├─────────────────────────────────────────────────────────────────────────┤
│  python3 run_rust.py --target N [--strategy S] [--optimizer O]          │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                    FASE 1: ESTRATEGIAS GEOMÉTRICAS                       │
│                    (Si --strategy != "none")                             │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌─ Mosaic ──────────────────────────┐                                   │
│  │ • Solo si N % 4 == 0              │                                   │
│  │ • Usa T{N/4}.csv como base        │                                   │
│  │ • Crea mosaico 2x2                │                                   │
│  └───────────────────────────────────┘                                   │
│                                                                          │
│  ┌─ Grid Zipper ─────────────────────┐                                   │
│  │ • Genera desde cero               │                                   │
│  │ • Patrón zigzag con stride_x,     │                                   │
│  │   row_height configurables        │                                   │
│  └───────────────────────────────────┘                                   │
│                                                                          │
│  ┌─ Pruning ─────────────────────────┐                                   │
│  │ • Lee T{N+1}.csv                  │                                   │
│  │ • Elimina 1 árbol (el peor)       │                                   │
│  └───────────────────────────────────┘                                   │
│                                                                          │
│  ┌─ Incremental ─────────────────────┐                                   │
│  │ • Lee T{N-1}.csv                  │                                   │
│  │ • Agrega 1 árbol en la periferia  │                                   │
│  └───────────────────────────────────┘                                   │
│                                                                          │
│  → Resultado: Mejor estrategia con score finito                          │
│               O ninguna si todas fallan                                  │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                    ┌───────────────┴───────────────┐
                    │                               │
            ¿Alguna funcionó?                ¿Ninguna funcionó?
                    │                               │
                    ▼                               ▼
┌──────────────────────────┐     ┌─────────────────────────────────────────┐
│ Usa la mejor estrategia  │     │        FASE 2: OPTIMIZADOR              │
│ como candidato           │     │        (Solo si fase 1 falló)           │
└──────────────────────────┘     ├─────────────────────────────────────────┤
            │                    │                                         │
            │                    │  Carga T{N}.csv existente o genera      │
            │                    │  semilla con Grid Zipper                │
            │                    │                                         │
            │                    │  Según --optimizer:                     │
            │                    │  ┌─ ga ──────────────────────┐          │
            │                    │  │ Algoritmo Genético        │          │
            │                    │  │ • Población + Selección   │          │
            │                    │  │ • Crossover + Mutación    │          │
            │                    │  └───────────────────────────┘          │
            │                    │                                         │
            │                    │  ┌─ cmaes ───────────────────┐          │
            │                    │  │ CMA-ES Adaptativo         │          │
            │                    │  │ • 4 restarts con sigma    │          │
            │                    │  │   creciente (5%→50%)      │          │
            │                    │  └───────────────────────────┘          │
            │                    │                                         │
            │                    │  ┌─ sa ──────────────────────┐          │
            │                    │  │ Simulated Annealing       │          │
            │                    │  │ • Acepta soluciones peores│          │
            │                    │  │ • Temperatura decreciente │          │
            │                    │  └───────────────────────────┘          │
            │                    └─────────────────────────────────────────┘
            │                               │
            └───────────┬───────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                    FASE 3: REFINAMIENTO                                  │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  1. save_if_better() ─────────────────────────────────────────────────── │
│     • Verifica si candidato tiene colisiones                             │
│     • Verifica si archivo existente tiene colisiones                     │
│     • Guarda si: nuevo válido Y (viejo inválido O nuevo mejor score)     │
│                                                                          │
│  2. Gravedad (--steps pasos) ─────────────────────────────────────────── │
│     • Mueve árboles hacia el centro                                      │
│     • Compacta el layout                                                 │
│     • save_if_better()                                                   │
│                                                                          │
│  3. Fine-Tuning (--fine-tune iters) ──────────────────────────────────── │
│     • Perturbaciones aleatorias pequeñas                                 │
│     • Acepta solo mejoras                                                │
│     • save_if_better()                                                   │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                         SALIDA                                           │
├─────────────────────────────────────────────────────────────────────────┤
│  solutions/T{N}.csv                                                      │
│  - Mejor solución encontrada (o la existente si no hubo mejora)          │
│  - Formato: id, x, y, deg, score                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Dependencias entre Archivos

```
T1.csv  ←───────────────────────────────────────────────────────────────┐
T2.csv  ← Incremental(T1)                                               │
T3.csv  ← Incremental(T2)                                               │
T4.csv  ← Mosaic(T1) o Incremental(T3)                                  │
...                                                                     │
T50.csv ← Zipper, Pruning(T51), Incremental(T49)                        │
...                                                                     │
T200.csv ← Zipper, Pruning(T201), Mosaic(T50)                           │
                                                                        │
                    Pruning lee archivo de N+1 ─────────────────────────┘
```

---

## Detalle de Estrategias

### Mosaic
- **Condición**: N es múltiplo de 4
- **Input**: `solutions/T{N/4}.csv`
- **Proceso**: Duplica la solución base en un mosaico 2x2, escalando posiciones

### Grid Zipper
- **Condición**: Siempre disponible
- **Input**: Ninguno (genera desde cero)
- **Proceso**: Crea patrón zigzag con árboles alternando orientación (0° y 180°)
- **Parámetros**: `stride_x` (separación horizontal), `row_height` (separación vertical)

### Pruning
- **Condición**: Existe `T{N+1}.csv`
- **Input**: `solutions/T{N+1}.csv`
- **Proceso**: Elimina el árbol que menos afecta el bounding box

### Incremental
- **Condición**: Existe `T{N-1}.csv`
- **Input**: `solutions/T{N-1}.csv`
- **Proceso**: Agrega un árbol en la periferia del layout existente

---

## Detalle de Optimizadores

### GA (Algoritmo Genético)
- **Uso**: Default cuando estrategias geométricas fallan
- **Proceso**: 
  - Crea población de soluciones
  - Selección por torneo
  - Crossover entre mejores individuos
  - Mutación aleatoria

### CMA-ES (Covariance Matrix Adaptation)
- **Uso**: `--optimizer cmaes`
- **Proceso**:
  - Optimización continua basada en distribución gaussiana
  - 4 restarts con sigma creciente (5%, 15%, 30%, 50%)
  - Mejor para escapar mínimos locales con exploración adaptativa

### SA (Simulated Annealing)
- **Uso**: `--optimizer sa`
- **Proceso**:
  - Acepta soluciones peores con probabilidad decreciente
  - Temperatura inicial alta → baja
  - Puede explorar configuraciones muy diferentes

---

## Detalle de Refinamiento

### save_if_better()
```
SI nuevo_tiene_colisiones:
    NO guardar
    
SI archivo_existe:
    SI archivo_tiene_colisiones:
        GUARDAR (reemplaza inválido)
    ELSE SI nuevo_score < viejo_score:
        GUARDAR (mejora)
ELSE:
    GUARDAR (archivo no existe)
```

### Gravedad
- Mueve cada árbol hacia el centroide del layout
- Paso pequeño (0.01 por iteración)
- Verifica colisiones después de cada movimiento

### Fine-Tuning
- Perturbaciones aleatorias pequeñas en x, y, ángulo
- Solo acepta si reduce el score
- Rápido, enfocado en micro-optimizaciones

---

## Ejemplos de Uso

```bash
# Ejecutar todas las estrategias para T50
python3 run_rust.py --target 50 --fine-tune 100

# Solo Pruning para T50
python3 run_rust.py --target 50 --strategy pruning --fine-tune 100

# Forzar CMA-ES sin estrategias geométricas
python3 run_rust.py --target 50 --strategy none --optimizer cmaes --gen 100

# Simulated Annealing con muchas iteraciones
python3 run_rust.py --target 50 --strategy none --optimizer sa --gen 50

# Batch de 200 a 1 (para Pruning)
for n in {200..1}; do python3 run_rust.py --target $n --strategy pruning --fine-tune 100; done
```
