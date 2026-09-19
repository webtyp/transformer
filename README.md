# transformer

Grafo del encoder transformer y kernels CPU/WASM: ids de tokens entran, un vector sale.

## Kernels y Benchmark de la Fase 3

`webtyp/transformer` implementa los kernels numéricos en Go plano (construidos sobre `vector.Dot`) para ejecutar la inferencia del encoder en CPU/WASM.

### Resultados del Benchmark (`BenchmarkEncode_20x12x384`)

El arnés sintético mide el costo del paso forward completo de un encoder para 20 tokens, 12 capas, 384 dimensiones, 6 cabezales de atención y dimensión FFN de 1536:

- **Tiempo medido:** ~261.7 ms / op
- **Piso teórico derivado (P1):** ~259 ms
- **Evaluación:** El tiempo medido (< 300 ms) confirma la viabilidad del transformer en CPU/WASM para la fase 3.

### Ejecución de tests y benchmarks

```bash
# Tests unitarios
go test -v ./...

# Benchmark sintético
go test -bench=BenchmarkEncode_20x12x384 -benchtime=2s .
```
