# 🎄 Guía de Uso del Optimizador (Rust)

Esta guía explica cómo utilizar el comando `./target/release/christmas_tree_optimizer` para mejorar las soluciones del problema Christmas Tree Packing.

---

## 🚀 Conceptos Clave

1.  **I/O Automático**: 
    - Si no especificas `-i` (input), intentará cargar del archivo `-o` (output) para resumir trabajo.
    - Si no especificas `-o` (output), guardará por defecto en `../solutions/T[N].csv`.
2.  **Kaggle Score**: El programa calcula internamente el score oficial (Bounding Square Area / N). **Solo guardará el resultado si es mejor que el que ya existe en el archivo de salida.**
3.  **Ajuste de N**: Si el archivo de entrada tiene menos árboles de los indicados en `--target-n`, el programa añade los faltantes automáticamente.

---

## 🛠️ Comandos por Estrategia

### 1. Algoritmo Genético (GA)
Ideal para exploración global y encontrar nuevas estructuras.
```bash
# Evolucionar una solución existente (500 generaciones)
./target/release/christmas_tree_optimizer -i ../solutions/T197.csv --strategy ga --generations 500

# Exploración desde cero (Pure Random - ignora el input)
./target/release/christmas_tree_optimizer --target-n 6 --strategy ga --pure-random --generations 1000
```

### 2. Fine-Tuning (Ajuste Fino)
Mueve y rota árboles ligeramente para ganar decimales. Muy efectivo para el tramo final.
```bash
./target/release/christmas_tree_optimizer -i ../solutions/T197.csv --strategy finetune --fine-tune-iters 5000
```

### 3. Gravedad (Compresión)
Empuja todos los árboles hacia el centro (0,0) mientras sea válido.
```bash
./target/release/christmas_tree_optimizer -i ../solutions/T197.csv --strategy gravity --gravity-steps 500
```

### 4. Recocido Simulado (Simulated Annealing)
Bueno para salir de mínimos locales donde Fine-Tuning se queda atrapado.
```bash
./target/release/christmas_tree_optimizer -i ../solutions/T6.csv --strategy sa --generations 200
```

### 5. ZipSkew (Geométrico)
Estrategia de empaquetado en rejilla optimizada con CMA-ES.
```bash
./target/release/christmas_tree_optimizer --target-n 197 --strategy zipskew --generations 100
```

### 6. Pipeline Automático (All)
Ejecuta una secuencia predefinida (ZipSkew -> Repair -> Gravity -> FineTune).
```bash
./target/release/christmas_tree_optimizer --target-n 197 --strategy all
```

### 7. Poda (Pruning)
Elimina árboles de forma inteligente para alcanzar un N menor.
```bash
# Podar de T197 a T196
./target/release/christmas_tree_optimizer -i ../solutions/T197.csv -o ../solutions/T196.csv --target-n 196 --strategy pruning

# Buscar T{N+1} automáticamente para podar si no especificas input diferente
./target/release/christmas_tree_optimizer --target-n 6 --strategy pruning
```

### 8. Incremento (Incremental)
Añade árboles en la periferia de forma inteligente.
```bash
# Subir de T5 a T6
./target/release/christmas_tree_optimizer -i ../solutions/T5.csv -o ../solutions/T6.csv --target-n 6 --strategy incremental

# Buscar T{N-1} automáticamente para incrementar
./target/release/christmas_tree_optimizer --target-n 197 --strategy incremental
```

---

## 📈 Ejemplos de Flujos de Trabajo

### De N=5 a N=6 (Incremento)
Si tienes un récord en T5 y quieres usarlo como base para T6:
```bash
./target/release/christmas_tree_optimizer -i ../solutions/T5.csv -o ../solutions/T6.csv --target-n 6 --strategy ga
```

### Optimización en Cascada
Puedes encadenar comandos para ir puliendo una solución:
1. **Paso 1 (Base)**: `./target/release/christmas_tree_optimizer --target-n 20 --strategy zipper`
2. **Paso 2 (Compactar)**: `./target/release/christmas_tree_optimizer -i ../solutions/T20.csv --strategy gravity --gravity-steps 1000`
3. **Paso 3 (Pulir)**: `./target/release/christmas_tree_optimizer -i ../solutions/T20.csv --strategy finetune --fine-tune-iters 10000`

---

## 📋 Parámetros Comunes

| Parámetro | Descripción | Defecto |
|-----------|-------------|---------|
| `-i`, `--input` | Archivo de entrada (.csv) | (Opcional) |
| `-o`, `--output` | Archivo donde guardar si hay mejora | `../solutions/T[N].csv` |
| `--target-n` | Número de árboles objetivo | Inferred from filename |
| `--generations` | Iteraciones para GA, SA, CMA-ES | 20 |
| `--pop-size` | Tamaño de población (solo GA) | 50 |
| `--gravity-steps`| Pasos de compresión por gravedad | 20 |
| `--fine-tune-iters`| Pasos de ajuste fino | 1000 |
| `--pure-random` | Ignorar input y empezar de cero | false |
