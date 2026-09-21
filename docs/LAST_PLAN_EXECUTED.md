---
PLAN: "feat: webtyp/transformer — etapa 2, el grafo real de granite-embedding-97m-multilingual-r2"
TAG: v0.2.0
---

> This plan is dispatched via the CodeJob workflow. See skill: agents-workflow.
> Índice maestro: https://github.com/webtyp/agent/blob/main/docs/MASTER_PLAN.md
>
> **Nota de idioma:** la prosa va en español; los bloques de código mantienen sus
> comentarios en inglés.

# Plan — `webtyp/transformer`, etapa 2

## Leé esta sección PRIMERO. No escribas una línea de código antes de terminarla.

Este repo **ya tiene código de la etapa 1, publicado y correcto**. Esta etapa lo **extiende**,
no lo reemplaza y no lo reescribe. Si en algún momento sentís que "sería más simple empezar de
nuevo" o "reorganizar todo el paquete", pará: es la señal de que no leíste esta sección con
cuidado. Dos intentos anteriores de este mismo plan se abortaron por reescribir de más — esta
sección existe específicamente para que no pase una tercera vez.

**Archivos que YA EXISTEN en este repo, ahora mismo, en `main`:**

| Archivo | Qué tiene | Qué hacés con él en esta etapa |
|---|---|---|
| `kernels.go` | Los 7 kernels de la etapa 1 (lista exacta abajo) | **Editalo una sola vez**, para el cambio de firma de `RoPE` (Stage 1). Nada más ahí cambia. |
| `kernels_test.go` | Tests de esos 7 kernels | Actualizá las llamadas a `RoPE` a la firma nueva (Stage 1). No toques nada más de este archivo. |
| `bench_test.go` | `BenchmarkEncode_20x12x384`, el arnés sintético de la etapa 1 | Se reemplaza por un benchmark que llama al `Encode` real (Stage 5). El arnés sintético cumplió su propósito (midió el piso de la fase 3) y ya no hace falta mantenerlo. |
| `transformer.go` | Un stub vacío del scaffold inicial (`gonew`): `type Transformer struct{}`, `func New() *Transformer` | **Borralo.** No es parte de ningún diseño — es lo que `gonew` deja por defecto en todo repo nuevo. No lo uses como base ni le agregues campos: la API real de esta etapa va en un archivo nuevo, `encode.go` (Stage 2). |
| `README.md`, `AGENTS.md` | Docs de la etapa 1 | `README.md` se actualiza en Stage 5 con el benchmark nuevo. `AGENTS.md` no cambia. |

**Los 7 kernels de `kernels.go`, firmas EXACTAS tal como están hoy — no las reimplementes, no
las "mejores", llamalas tal cual:**

```go
func MatmulT(dst, a, bT []float32, m, k, n int) error
func LayerNorm(dst, src, gamma, beta []float32, dim int, eps float32) error
func RMSNorm(dst, src, gamma []float32, dim int, eps float32) error
func Softmax(x []float32) error
func GELU(x []float32) error
func SiLU(x []float32) error
func Add(dst, src []float32) error
func RoPE(q, k []float32, pos, dim, heads int) error  // esta SÍ cambia — es la única. Ver Stage 1.
```

`MatmulT` ya espera `bT` **transpuesta** (fila = dimensión de salida). Todos los pesos que uses
en esta etapa vienen ya transpuestos así — no transpongas nada vos, y no le pases una matriz en
el layout que no sea ese.

**Dependencias del repo, ya en `go.mod`, no agregues ninguna otra:** `webtyp.com/vector` (usás
`vector.Dot` indirectamente, a través de `MatmulT` — no lo llames directo) y `webtyp.com/fmt`
(para errores: `fmt.Err("transformer: mensaje")`, nunca el `errors`/`fmt` de stdlib).

**Esta etapa NO depende de `webtyp/weights` ni de `webtyp/tokenizer`.** Aunque esos dos repos
existen y tienen sus propios planes en curso, este paquete sigue recibiendo tensores ya
decodificados en `[]float32` y una secuencia ya tokenizada — nunca un `weights.Artifact` ni
un `tokenizer.BPE`. Si te encontrás importando cualquiera de los dos, pará: te saliste del
alcance de este plan.

---

## Etapas, en orden. Hacé cada una completa antes de pasar a la siguiente.

### Stage 1 — Cambiar la firma de `RoPE`, y solo eso, en los archivos existentes

`RoPE` (v0.1.0, en `kernels.go`) tiene `10000.0` fijo adentro de la función — ver el código
actual. Ningún modelo real de este proyecto usa ese valor (`granite-embedding-97m-multilingual-r2`
usa 150000 o 160000 según la capa, ver Stage 2). Cambiá la firma a:

```go
func RoPE(q, k []float32, pos int, theta float64, dim, heads int) error
```

Adentro de la función, reemplazá el `10000.0` hardcodeado por el parámetro `theta`. Nada más
cambia en la lógica de `RoPE`.

**Archivos a tocar en este stage, y nada más:**
- `kernels.go`: la firma y el cuerpo de `RoPE`.
- `kernels_test.go`: cada test que llama a `RoPE` le pasa `10000.0` explícito donde antes no
  hacía falta (`TestRoPE_RotationPreservesNorm`, `TestRoPE_PositionZeroIsIdentity`).
- `bench_test.go`: la llamada a `RoPE` dentro de `BenchmarkEncode_20x12x384` — aunque ese
  benchmark se reemplaza en Stage 5, tiene que seguir compilando mientras tanto, así que
  actualizale la llamada acá también (pasale `10000.0`).

**Verificación de este stage:** `go build ./... && go test ./...` pasa, sin tocar ningún otro
archivo.

### Stage 2 — Arquitectura real de `granite-embedding-97m-multilingual-r2`, verificada

Estos números salen de leer el `config.json` real y el header real de `model.safetensors` del
modelo (HuggingFace, `ibm-granite/granite-embedding-97m-multilingual-r2`) — no son de un paper
genérico de ModernBERT ni de memoria:

```
num_hidden_layers: 12       num_attention_heads: 12      hidden_size: 384
intermediate_size: 1536     layer_norm_eps: 1e-5          vocab_size: 180000
attention_bias: false       mlp_bias: false                norm_bias: false
global_attn_every_n_layers: 3
global_rope_theta: 150000.0    local_rope_theta: 160000.0    local_attention: 128
classifier_pooling: "cls"
```

`head_dim = 384 / 12 = 32` (12 cabezales, no 6 — la etapa 1 benchmarkeó con 6 porque el modelo
todavía no estaba elegido; esta etapa usa el número real).

**Capas globales vs. locales:** cada 3 capas hay una de atención global (completa); el resto
son locales (ventana deslizante de 128). Con `global_attn_every_n_layers = 3`, las capas 0, 3,
6, 9 son globales — **confirmá la convención exacta (0-indexed módulo 3) contra
`modeling_modernbert.py` de `huggingface/transformers` versión `4.56.2`** (la que exportó este
`config.json`) antes de escribirlo; no la adivines de esta prosa. Las capas globales usan
`global_rope_theta` (150000); las locales, `local_rope_theta` (160000) con la ventana de 128.

**Para una consulta de navegador (~20-30 tokens) esto no cambia nada observable** — la ventana
local (128) es más ancha que la secuencia entera, así que atención local y global dan el mismo
resultado. Implementá la ventana correcta igual (para cuando el backend embeba documentos de
hasta 8K tokens), pero el test de aceptación de este plan usa secuencias cortas — no le
dediques a esto el mismo esfuerzo de testing que al resto.

**La capa 0 no tiene norm antes de la atención.** Usa `Identity` ahí porque `embeddings.norm`
ya normalizó justo antes. En la `Config`/`Weights` de Stage 3, esto se representa con
`AttnNormGamma: nil` para la capa 0 — `LayerNorm` de `kernels.go` ya acepta `gamma == nil`
(mirá su firma arriba), así que no hace falta una rama especial en tu grafo, solo pasar `nil`
para esa capa.

**El MLP es con compuerta, no un MLP simple.** El peso de proyección de subida es una sola
matriz que fusiona dos proyecciones (activación de entrada + compuerta multiplicativa),
`hidden_activation: "silu"`. **El orden exacto de qué mitad lleva la activación y en qué orden
se multiplican sale de leer `ModernBertMLP.forward` en `modeling_modernbert.py` — no lo
derives de esta prosa, y no lo dejes "a criterio": el fixture de Stage 4 es lo único que
confirma si lo tenés bien.**

**Pooling: la posición 0 del último hidden state** (`classifier_pooling: "cls"`), no un
promedio sobre la secuencia. La posición 0 es el token `<|startoftext|>` que
`webtyp/tokenizer` antepone.

**Sin biases en ningún lado** — todo `LayerNorm`/`RMSNorm` de esta etapa se llama con
`beta = nil`.

### Stage 3 — Archivo nuevo: `encode.go`

Un archivo, no varios — si supera 500 líneas, dividilo por dominio y avisá en el PR por qué.

```go
package transformer

// Config is the architecture shape. See docs/PLAN.md Stage 2 for granite's real values.
type Config struct {
	NumLayers          int
	Heads              int
	Dim                int
	FFNDim             int // 1536 — before the ×2 fusion in Wi
	GlobalEveryNLayers int
	LocalWindow        int
	GlobalRopeTheta    float64
	LocalRopeTheta     float64
	Eps                float32
}

// LayerWeights holds one block's tensors: already dequantized to float32, already
// transposed for MatmulT (see "Leé esta sección PRIMERO" above — same row=output-dim
// convention kernels.go already uses). AttnNormGamma is nil for layer 0 (Stage 2).
type LayerWeights struct {
	AttnNormGamma []float32 // [Dim], nil for layer 0
	WqkvT         []float32 // [3*Dim, Dim]
	WoT           []float32 // [Dim, Dim]
	MlpNormGamma  []float32 // [Dim]
	WiT           []float32 // [2*FFNDim, Dim]
	MlpWoT        []float32 // [Dim, FFNDim]
}

type Weights struct {
	EmbedNormGamma []float32      // [Dim]
	Layers         []LayerWeights // len == Config.NumLayers
	FinalNormGamma []float32      // [Dim]
}

// GatedFFN computes the gated feed-forward block: wiT projects to 2*ffnDim, splits into
// two halves, applies the activation to one and multiplies elementwise by the other, then
// woT projects back down. Which half gets the activation: Stage 2's note on the MLP order —
// verify against modeling_modernbert.py, don't guess. Built on MatmulT and SiLU from
// kernels.go — do not reimplement either.
func GatedFFN(dst, src, wiT, woT []float32, seqLen, dim, ffnDim int) error

// Encode runs one forward pass. tokenEmbeds is seqLen*Dim float32 values — the embedding
// rows FOR THIS SEQUENCE'S TOKENS ONLY, already gathered and dequantized by the caller
// (webtyp/embed's future adapter — NOT this package; do not add embedding-table gather
// code here, and do not import webtyp/weights to do it "properly"). Returns the pooled
// (CLS) vector, Dim elements long. Does NOT L2-normalize — that stays embed's job, same
// boundary as stage 1.
func Encode(cfg Config, w Weights, tokenEmbeds []float32, seqLen int) ([]float32, error)
```

Dentro de `Encode`: aplicá `EmbedNormGamma` sobre `tokenEmbeds` (esto es `embeddings.norm`,
ANTES de la capa 0), después iterá las `NumLayers` capas (cada una: pre-norm de atención si
`AttnNormGamma != nil` → QKV vía `MatmulT` → `RoPE` con el `theta` que corresponda según si la
capa es global o local (Stage 2) → atención (global o con ventana local) → proyección de
salida vía `MatmulT` → residual con `Add` → pre-norm de MLP → `GatedFFN` → residual con `Add`),
después `FinalNormGamma`, después devolvé la fila 0 del resultado (pooling CLS).

### Stage 4 — Fixture de referencia real

**No inventes los vectores esperados.** Corré el modelo real una vez (Python, `transformers` +
`AutoModel.from_pretrained("ibm-granite/granite-embedding-97m-multilingual-r2")`) sobre 2-3
oraciones cortas fijas, y guardá los vectores de salida (384 floats cada uno) en
`testdata/reference_vectors.json`, formato `[{"text": "...", "vector": [...]}]`. Esta es la
única forma real de confirmar que el orden del split de `GatedFFN` y la convención de capas
globales/locales (Stage 2) están bien — si están mal, este es el test que lo detecta.

### Stage 5 — Tests, en `encode_test.go`

| Test | Verifica |
|---|---|
| `TestEncode_MatchesReference` | por cada entrada de `testdata/reference_vectors.json`, similitud coseno ≥ 0.99 contra el vector real (no bit-exacto — hay ruido de cuantización int8 en el camino real) |
| `TestGatedFFN_MatchesNaive` | contra una implementación ingenua (dos matmuls + split + activación + multiplicación a mano), 1e-5 relativo |
| `TestEncode_Layer0SkipsAttnNorm` | `AttnNormGamma: nil` en la capa 0 no pánica |
| `TestRoPE_DifferentThetaPerLayerType` | dos `theta` distintos dan resultados distintos para la misma posición (confirma que el parámetro se usa de verdad) |
| `TestEncode_PoolsPositionZero` | cambiar el hidden state en una posición != 0 no cambia el vector pooled; cambiarlo en la posición 0 sí |

### Stage 6 — Reemplazar el benchmark, y medir en los tres targets

Reemplazá `BenchmarkEncode_20x12x384` en `bench_test.go` por un benchmark que llame a `Encode`
de verdad (12 capas, 12 cabezales, `GatedFFN` real) sobre pesos sintéticos del tamaño correcto
— mismo nombre de función está bien, el arnés sintético ya cumplió su propósito.

Corré los tres comandos y registrá los tres números en `README.md`, lado a lado — esto es lo
que esta etapa existe para producir, no un detalle opcional:

```bash
go test -bench=BenchmarkEncode -benchtime=2s .                   # nativo
gotest -tinygo                                                    # TinyGo
tinygo test -target wasm -bench=BenchmarkEncode -benchtime=2s .   # el número real, igual que etapa 1
```

Si alguno falla en compilar o da un número que no tiene sentido (por ejemplo, más rápido que
el piso derivado en `PENDING_ITEMS.md` P1), documentalo en el README con el mensaje de error o
el número real tal cual salió — no lo omitas ni lo redondees.

## Checklist de aceptación final

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

## Tabla de etapas

| Etapa | Archivos que toca | Archivos que crea |
|---|---|---|
| 1 | `kernels.go`, `kernels_test.go`, `bench_test.go` | — |
| 2 | (solo lectura — arquitectura, no código) | — |
| 3 | — (borra `transformer.go`) | `encode.go` |
| 4 | — | `testdata/reference_vectors.json` |
| 5 | — | `encode_test.go` |
| 6 | `bench_test.go`, `README.md` | — |
