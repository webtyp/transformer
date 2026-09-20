# transformer
<img src="docs/img/badges.svg">

Grafo del encoder transformer y kernels CPU/WASM: ids de tokens entran, un vector sale.

## Kernels y Benchmark de la Fase 3

`webtyp/transformer` implementa los kernels numéricos en Go plano (construidos sobre `vector.Dot`) para ejecutar la inferencia del encoder en CPU/WASM.

### Resultados del Benchmark (`BenchmarkEncode_20x12x384`)

El arnés sintético mide el costo del paso forward completo de un encoder para 20 tokens, 12 capas, 384 dimensiones, 6 cabezales de atención y dimensión FFN de 1536, corrido bajo `tinygo test -target wasm` — el target real; `go test -bench` da un número nativo ~2,5× más rápido y no sirve para esta decisión.

- **Tiempo medido:** ~300–340 ms/op (6 corridas de `-benchtime=2s`/`3s`, promedio ~313 ms; hay
  varianza corrida a corrida, y la mayoría cae apenas sobre 300 ms)
- **Piso teórico derivado (P1):** ~259 ms
- **Evaluación:** el número cae en el rango medio de la tabla de desenlaces del plan
  (300 ms – 1 s): **viable con reservas, requiere una decisión humana** sobre si esa latencia
  es aceptable en un cuadro de búsqueda. No es el desenlace "< 300 ms, etapa 2 como está
  escrita" — está documentado así en vez de redondeado hacia abajo.

### Ejecución de tests y benchmarks

```bash
# Tests unitarios
go test -v ./...

# Benchmark sintético — SOLO bajo TinyGo/wasm, ver arriba
tinygo test -target wasm -bench=BenchmarkEncode_20x12x384 -benchtime=2s .
```
