---
PLAN: "feat: webtyp/transformer — etapa 2, el grafo real de granite-embedding-97m-multilingual-r2"
TAG: v0.2.0
EXECUTOR: jules
REVIEWER: none
---

> This plan is dispatched via the CodeJob workflow. See skill: agents-workflow.
> Índice maestro: https://github.com/webtyp/agent/blob/main/docs/MASTER_PLAN.md
>
> **Etapa 2 de 2.** La etapa 1 (`v0.1.0`, ya mergeada) entregó los kernels numéricos
> genéricos y un arnés sintético de benchmark. Este plan construye el grafo **real** —
> específico de la arquitectura de `granite-embedding-97m-multilingual-r2` (`ModernBertModel`),
> no una forma genérica — y corre inferencia real contra el artifact real que producen
> [`webtyp/weightsc`](https://github.com/webtyp/weightsc/blob/main/docs/PLAN.md) y
> [`webtyp/tokenizer`](https://github.com/webtyp/tokenizer/blob/main/docs/PLAN.md).
>
> **Todos los números de arquitectura de abajo están verificados contra el `config.json` y el
> header real de `model.safetensors` del modelo** (HuggingFace,
> `ibm-granite/granite-embedding-97m-multilingual-r2`), no de memoria ni de un paper genérico
> de ModernBERT. Donde un detalle no está fijado acá (ver "Verificación obligatoria contra la
> fuente" más abajo), la fuente de verdad es el código real de HuggingFace `transformers`
> (`modeling_modernbert.py`, versión `4.56.2` — la que exportó este `config.json`), no una
> suposición.
>
> **Nota de idioma:** la prosa va en español; los bloques de código mantienen sus
> comentarios en inglés.

# Plan — `webtyp/transformer`, etapa 2

## Responsabilidad única

Ids de tokens y pesos entran; **un vector sale**. Este plan implementa el forward pass real
de `ModernBertModel` para embeddings — nada de entrenamiento, nada de cabeza de MLM, nada de
generación.

## Design gate

Esta etapa cambia una firma ya publicada (`RoPE`, v0.1.0) y agrega un kernel nuevo — pasa por
el gate.

**1. Prior art.** `modeling_modernbert.py` de HuggingFace `transformers` (la implementación de
referencia de este modelo exacto); `llama.cpp`'s `ggml` para el patrón de "grafo compuesto de
kernels reusables" en C; `onnxruntime`'s graph de operadores. Este plan porta el primero
literalmente — no reinterpreta la arquitectura, la copia.

**2. Novice-name test.** `Encode(cfg, weights, tokenEmbeds, seqLen)` — mismo verbo que ya usa
la etapa 1 en sus comentarios y que usa toda la literatura de embeddings ("encode a sentence").
`GatedFFN` es el nombre que ya usa la arquitectura (SwiGLU/GeGLU son casos particulares de
"FFN con compuerta"; acá no hace falta el nombre exacto de la variante, el comportamiento es
lo que importa).

**3. Complexity ledger.**
```
Kernels nuevos                      +1 (GatedFFN — no existía en la etapa 1)
Firmas que cambian                  1  (RoPE gana un parámetro theta — ver abajo)
Formas de hacer un forward pass     1  (Encode; nada más lo re-implementa)
```

**4. Dónde vive.** Mismo repo que la etapa 1 — es su continuación directa, no una capa nueva.
Sigue sin depender de `weights` ni de `tokenizer` (la regla de la etapa 1 no cambia: recibe
tensores ya decodificados e ids ya tokenizados, ver "Qué NO entra a este paquete" abajo).

**5. Qué borra / qué cambia.** `RoPE(q, k []float32, pos, dim, heads int) error` (v0.1.0) tenía
`10000.0` fijo adentro — un valor que **ningún candidato real usa** (`granite` usa 150000 y
160000, ver abajo). Era una firma incompleta descubierta en su primer uso real, no una
decisión estable — se corrige acá, en el primer PR que la necesita, no se acarrea el defecto
con un segundo parámetro opcional ni un wrapper. Nueva firma:

```go
func RoPE(q, k []float32, pos int, theta float64, dim, heads int) error
```

Todo call site de la etapa 1 (`bench_test.go`, tests) se actualiza para pasar `10000.0`
explícito donde antes era implícito — el comportamiento por defecto no cambia para quien ya
lo usaba, solo deja de estar escondido.

## Arquitectura real — verificada, no de memoria

```
model_type: modernbert          hidden_size: 384        num_hidden_layers: 12
num_attention_heads: 12         intermediate_size: 1536  vocab_size: 180000
layer_norm_eps: 1e-5            attention_bias: false    mlp_bias: false
norm_bias: false                global_attn_every_n_layers: 3
global_rope_theta: 150000.0     local_rope_theta: 160000.0     local_attention: 128
classifier_pooling: "cls"
```

`num_attention_heads` es **12**, no 6 — la etapa 1 benchmarkeó con 6 porque el modelo todavía
no estaba elegido. `head_dim = 384 / 12 = 32`.

**Capas globales vs. locales:** cada 3 capas hay una de atención **global** (completa); el
resto son **locales** (ventana deslizante). Con `global_attn_every_n_layers = 3`, las capas
0, 3, 6, 9 son globales (verificalo contra `modeling_modernbert.py`: la convención exacta de
qué índice cuenta como "cada N" — 0-indexed vs 1-indexed — es exactamente el tipo de detalle
que hay que leer del código real, no adivinar). Las capas globales usan `global_rope_theta`
(150000); las locales, `local_rope_theta` (160000) con una máscara de ventana de
`local_attention` (128) tokens.

**Para una consulta del navegador (~20-30 tokens) esto no cambia nada observable:** la
ventana local (128) es más ancha que la secuencia entera, así que atención local y global dan
el mismo resultado. Para un documento del backend (hasta 8K tokens) sí importa — implementá la
ventana correctamente igual, pero no le dediques el mismo esfuerzo de testing que al resto:
el test de aceptación de este plan usa una secuencia corta (ver Tests).

**Capa 0 no tiene `attn_norm`.** Verificado en el header real de `model.safetensors`: las
capas 1-11 tienen `layers.N.attn_norm.weight`, la capa 0 no — usa `Identity` ahí porque
`embeddings.norm` ya normalizó justo antes. Si tu grafo aplica un norm antes de la atención de
la capa 0, está mal.

**El MLP es con compuerta (`GatedFFN`), no MLP simple.** `layers.N.mlp.Wi.weight` tiene shape
`[3072, 384]` — el doble de `intermediate_size` (1536), porque son **dos proyecciones
fusionadas en una sola matriz**: una que se activa y otra que actúa de compuerta
multiplicativa. `hidden_activation: "silu"` en `config.json`. **El orden exacto de qué mitad
es cuál y en qué orden se multiplican es del código real de `modeling_modernbert.py`
(`ModernBertMLP.forward`) — leelo, no lo derives de esta prosa.** El punto de verdad final es
el test de fixture contra la referencia (ver Tests).

**Pooling: CLS**, no mean-pooling. `classifier_pooling: "cls"` — el vector de salida es el
hidden state final en la posición 0 (el token `<|startoftext|>` que `webtyp/tokenizer`
antepone). No promedies sobre toda la secuencia.

**Sin biases en ningún lado** (`attention_bias`, `mlp_bias`, `norm_bias` todos `false`) — los
kernels de la etapa 1 ya soportan `gamma`/`beta` opcionales (`nil`) en `LayerNorm`; usalos con
`beta = nil` en todo este plan.

## Verificación obligatoria contra la fuente

No inventes estos tres detalles a partir de esta prosa — leé el código real
(`huggingface/transformers`, tag/versión `4.56.2`, archivo `modeling_modernbert.py`) antes de
escribir el grafo:

1. Orden exacto del split de `mlp.Wi` (`input, gate = Wi(x).chunk(2, dim=-1)` — ¿cuál mitad
   lleva la activación, cuál es la compuerta cruda?).
2. Convención exacta de qué índice de capa es "global" con `global_attn_every_n_layers = 3`
   (0-indexed módulo N, o algo distinto).
3. Ancho exacto de la ventana de atención local (¿128 tokens total, o ±64 a cada lado de la
   posición actual? ¿Cómo se aplica en los bordes de la secuencia?).

## API

```go
// Config is the architecture shape — see "Arquitectura real" above for granite's values.
type Config struct {
	NumLayers           int
	Heads               int
	Dim                 int
	FFNDim              int     // 1536 — before the ×2 fusion in Wi
	GlobalEveryNLayers  int
	LocalWindow         int
	GlobalRopeTheta     float64
	LocalRopeTheta      float64
	Eps                 float32
}

// LayerWeights holds one ModernBERT block's tensors, ALREADY DEQUANTIZED to float32 and
// ALREADY TRANSPOSED for MatmulT (row-major, output-dim-major — same convention the stage 1
// kernels already use). AttnNormGamma is nil for layer 0 (see "Capa 0" above).
type LayerWeights struct {
	AttnNormGamma []float32 // [Dim], nil for layer 0
	WqkvT         []float32 // [3*Dim, Dim]
	WoT           []float32 // [Dim, Dim]
	MlpNormGamma  []float32 // [Dim]
	WiT           []float32 // [2*FFNDim, Dim]
	MlpWoT        []float32 // [Dim, FFNDim]
}

type Weights struct {
	EmbedNormGamma []float32 // [Dim]
	Layers         []LayerWeights // len == Config.NumLayers
	FinalNormGamma []float32 // [Dim]
}

// Encode runs one forward pass. tokenEmbeds is seqLen*Dim float32 values — the embedding
// table ROWS FOR THIS SEQUENCE'S TOKENS ONLY, already gathered and dequantized by the
// caller (see "Qué NO entra a este paquete"). Returns the pooled (CLS) vector, Dim elements long.
// Does NOT normalize the output — that stays embed's job (unchanged from stage 1).
func Encode(cfg Config, w Weights, tokenEmbeds []float32, seqLen int) ([]float32, error)
```

**`GatedFFN`, el kernel nuevo:**

```go
// GatedFFN computes the gated feed-forward block: Wi projects to 2*ffnDim, splits into
// two halves, applies the activation to one and multiplies elementwise by the other, then
// Wo projects back down. See "Verificación obligatoria" #1 for which half gets the
// activation — get this from modeling_modernbert.py, not from guessing.
func GatedFFN(dst, src, wiT, woT []float32, seqLen, dim, ffnDim int) error
```

Construite sobre `MatmulT` y `SiLU` de la etapa 1 — no reimplementes esas partes.

## Qué NO entra a este paquete

Igual que en la etapa 1: sin `weights`, sin `tokenizer`, sin `syscall/js`. La tabla de
embeddings completa (180 000 filas) **nunca** se decodifica entera a `float32` — eso costaría
~276 MB para usar ~20-30 filas por consulta (`MASTER_PLAN.md` D5). Gatherear y dequantizar las
filas de esta secuencia es responsabilidad del **futuro adaptador `webtyp/embed`**, no de este
paquete — `Encode` recibe `tokenEmbeds` ya resuelto. No agregues ese código acá aunque
`embed` todavía no exista: es la responsabilidad de otro repo, y dispatcharlo antes de tiempo
es trabajo especulativo.

## Tests

**Fixture de referencia real, no inventada.** Igual que `webtyp/tokenizer`: corré el modelo
real una vez (Python, `sentence-transformers` o `transformers` +
`AutoModel.from_pretrained("ibm-granite/granite-embedding-97m-multilingual-r2")`) sobre 2-3
oraciones cortas fijas, y checkeá el vector de salida (384 floats) como
`testdata/reference_vectors.json`. Esta es la única forma real de confirmar los tres puntos de
"Verificación obligatoria" — si los tenés mal, el test lo detecta; si el test pasa y están
mal, es porque el fixture también está mal, así que generalo con cuidado.

| Test | Verifica |
|---|---|
| `TestEncode_MatchesReference` | el vector pooled para cada oración del fixture cae dentro de tolerancia (similitud coseno ≥ 0.99 — hay ruido de cuantización int8 en el camino real, no bit-exacto) del vector real de HuggingFace |
| `TestGatedFFN_MatchesNaive` | contra una implementación ingenua de dos matmuls + split + activación + multiplicación, 1e-5 relativo |
| `TestEncode_Layer0SkipsAttnNorm` | pasar `AttnNormGamma: nil` en la capa 0 no pánica y produce el mismo resultado que aplicar directamente sin norm |
| `TestRoPE_DifferentThetaPerLayerType` | con `theta` distinto, la rotación da resultados distintos para la misma posición (confirma que el parámetro realmente se usa, no quedó hardcodeado en otro lado) |
| `TestEncode_PoolsPositionZero` | cambiar el hidden state en cualquier posición != 0 no cambia el vector pooled; cambiarlo en la posición 0 sí |

## Benchmark real — la comparación que esta tanda de planes existe para producir

Reemplazá (o agregá junto a) `BenchmarkEncode_20x12x384` de la etapa 1 con un benchmark que
llame a `Encode` de verdad, con la forma real (12 capas, 12 cabezales, `GatedFFN` real) sobre
pesos sintéticos del tamaño correcto. Corré los tres targets y registrá los tres números en el
README, lado a lado — esto es lo que el índice maestro pidió de esta tanda:

```bash
go test -bench=BenchmarkEncode -benchtime=2s .              # nativo
gotest                                                        # Go stdlib, target js/wasm
gotest -tinygo                                                 # TinyGo
tinygo test -target wasm -bench=BenchmarkEncode -benchtime=2s . # el número real, igual que etapa 1
```

Si alguno de los tres falla en compilar o corre pero da un resultado que no tiene sentido
(por ejemplo, más rápido que el piso derivado), documentalo explícitamente en el README con el
mensaje de error o el número real — **no lo omitas ni lo redondees**. Es exactamente el dato
que decide si TinyGo alcanza para este proyecto o si hace falta reconsiderar el toolchain.

## Checklist de aceptación

```bash
go vet ./...
gotest
gotest -tinygo
GOOS=js GOARCH=wasm go build ./...
tinygo build -target wasm -o /dev/null .
grep -rn "syscall/js\|webtyp.com/weights\|webtyp.com/tokenizer" .           # → vacío
grep -rn "map\[" --include="*.go" . | grep -v _test.go                      # → vacío
grep -rn '"errors"\|"fmt"' --include="*.go" . | grep -v _test.go            # → vacío
```
