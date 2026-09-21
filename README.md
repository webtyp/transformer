# transformer
<img src="docs/img/badges.svg">

Grafo del encoder transformer y kernels CPU/WASM: ids de tokens entran, un vector sale.

Implementa el forward pass real de `ibm-granite/granite-embedding-97m-multilingual-r2`
(ModernBERT: 12 capas, 384 dims, 12 cabezales, FFN con compuerta SiLU, RoPE NeoX con
theta 150000/160000 según capa global/local, pooling CLS en la posición 0).
Ver `docs/PLAN.md` (etapa 2) para la arquitectura verificada contra
`modeling_modernbert.py` de `transformers` 4.56.2.

## Desvío documentado respecto al plan

El plan (Stage 1) pedía no tocar la lógica de `RoPE` más allá de la firma. Al
verificar contra `modeling_modernbert.py` (`rotate_half` + `emb = cat(freqs, freqs)`,
`interleaved=False` también en el path de flash-attention) resultó que el modelo usa
rotación estilo GPT-NeoX (mitades), mientras el kernel de la etapa 1 rotaba pares
intercalados (estilo GPT-J). Medición con el fixture real: pares → coseno 0.96–0.98
contra el modelo (bajo el umbral 0.99 del plan); mitades → coseno 1.00000000 en las
3 oraciones. Se corrigió el kernel a mitades; el fixture es el árbitro y así lo exige.

## Fixture de referencia (`testdata/reference_vectors.json`)

Tres oraciones fijas → vectores de 384 floats obtenidos con
`AutoModel.from_pretrained("ibm-granite/granite-embedding-97m-multilingual-r2")`
(`transformers` + torch CPU, `attn_implementation="eager"`), tomando
`last_hidden_state[:, 0]` (pooling CLS, sin normalizar L2 — eso queda en `embed`).
`Encode` con los pesos reales reproduce estos vectores con coseno 1.00000000
(verificado fuera del repo: el checkpoint, ~109 MB en float32, no se vende en
`testdata`; el test embebe solo los vectores).

## Benchmark (`BenchmarkEncode_20x12x384`)

Llama al `Encode` real (12 capas, 12 cabezales, `GatedFFN` real) sobre pesos
sintéticos fijos (seed 42), 20 tokens. El arnés sintético de la etapa 1 cumplió su
propósito (piso de la fase 3) y se eliminó.

| Comando | Resultado (i7-11800H) |
|---|---|
| `go test -bench=BenchmarkEncode -benchtime=2s .` (nativo) | ~331 ms/op |
| `gotest -tinygo` | pasa (vet ✅, race ✅, tests ✅, cobertura 74.2%) |
| `tinygo test -target wasm -bench=BenchmarkEncode -benchtime=2s .` (número real) | ~526 ms/op |

Lectura: el piso P1 de la etapa 1 (~259 ms) describía el arnés viejo (6 cabezales,
FFN simple con GELU). El grafo real cuesta más — la proyección de subida con
compuerta duplica el ancho (3072 vs 1536) y hay 12 cabezales — así que 526 ms sobre
el piso es un número creíble, no un benchmark roto. Sigue en la banda "viable con
reservas" de la etapa 1 y requiere la misma decisión humana sobre latencia.

Nota: `tinygo build -target wasm -o /dev/null .` no aplica a este repo porque es
un paquete biblioteca (`package transformer`, sin `main`); falla con
`expected main package to have name "main", not "transformer"` tanto antes como
después de esta etapa. La compilación wasm real se verifica con
`GOOS=js GOARCH=wasm go build ./...` y `tinygo test -target wasm .`, que pasan.

### Ejecución de tests y benchmarks

```bash
# Tests unitarios
go test -v ./...

# Suite completa (vet, race, cover, wasm)
gotest
gotest -tinygo

# Benchmark real — SOLO bajo TinyGo/wasm vale para la decisión (ver arriba)
tinygo test -target wasm -bench=BenchmarkEncode -benchtime=2s .
```
