---
PLAN: "feat: webtyp/transformer — etapa 3, soporte para bekko-embedding-v1-a8m"
TAG: v0.1.3
---

> This plan is dispatched via the CodeJob workflow. See skill: agents-workflow.
> Índice maestro: https://github.com/webtyp/agent/blob/main/docs/MASTER_PLAN.md
>
> **Nota de idioma:** la prosa va en español; los bloques de código mantienen sus
> comentarios en inglés.

# Plan — `webtyp/transformer`, etapa 3: `bekko-embedding-v1-a8m`

## Por qué existe esta etapa

`P1b` se reabrió: el forward pass real de granite (526 ms, `docs/LAST_PLAN_EXECUTED.md` de la
etapa 2, ahora en git history) cae en la banda "viable con reservas", y
`SMALL_MODEL_FOR_EMBEDING.md` §6 ya nombraba `bekko-embedding-v1-a8m` como el plan B real —
4 capas contra las 12 de granite. Decisión del usuario: cambiar a `bekko-embedding-v1-a8m`.

## Verificado contra el modelo real, no de memoria

`config.json` y el header real de `model.safetensors`
(`hotchpotch/bekko-embedding-v1-a8m`, HuggingFace, MIT) — leídos directo, no inferidos:

```
num_hidden_layers: 4        num_attention_heads: 6      hidden_size: 384
intermediate_size: 1152     global_attn_every_n_layers: 3   local_attention: 128
global_rope_theta: 160000   local_rope_theta: 160000    layer_norm_eps: 1e-5
classifier_pooling: mean    vocab_size: 256000           dtype: bfloat16

model.safetensors: 26 tensores.
  embeddings.tok_embeddings.weight  [256000, 384]  BF16
  embeddings.norm.weight            [384]          BF16
  final_norm.weight                 [384]          BF16
  por capa i en 0..3:
    layers.{i}.attn.Wqkv.weight     [1152, 384]    BF16   (3×384, igual que granite)
    layers.{i}.attn.Wo.weight       [384, 384]     BF16
    layers.{i}.attn_norm.weight     [384]          BF16   AUSENTE en la capa 0 (mismo Identity que granite)
    layers.{i}.mlp.Wi.weight        [2304, 384]    BF16   (2×1152, gate+up fusionado, igual patrón que granite)
    layers.{i}.mlp.Wo.weight        [384, 1152]    BF16
    layers.{i}.mlp_norm.weight      [384]          BF16
```

Conteo: 3 + (4×6 − 1) = 26. ✓ coincide con el header real.

**Conclusión clave: es el mismo `ModernBertModel` que granite, con menos capas y FFN más
angosto.** Ningún kernel de `kernels.go` cambia. `Encode` en `encode.go` tampoco cambia de
estructura — solo necesitaba una cosa que granite no ejercitaba: `classifier_pooling: mean`
en vez de CLS.

## Cambio 1 — `Config` gana un campo `Pooling`

`encode.go`: nuevo tipo `Pooling int` con dos valores, `PoolingCLS` (cero, compatible con
código existente — granite sigue sin tocar) y `PoolingMean`. `Config` gana el campo
`Pooling Pooling`.

## Cambio 2 — `Encode` pondera por `cfg.Pooling` al final

Antes de esta etapa, la última línea de `Encode` siempre copiaba `h[:dim]` (posición 0). Ahora
rama por `cfg.Pooling`: `PoolingMean` promedia las `dim` columnas sobre las `seqLen` filas de
`h`; cualquier otro valor (incluido el cero, `PoolingCLS`) mantiene el `copy` de siempre.

## Cambio 3 — `bekkoA8mConfig()` en los tests, y dos tests de pooling

`encode_test.go` gana un helper `bekkoA8mConfig()` (los valores reales de arriba, con
`Pooling: PoolingMean`) al lado de `graniteConfig()`, y dos tests:

- `TestEncode_MeanPooling_SingleTokenMatchesCLS`: con `seqLen=1`, mean y CLS tienen que dar el
  mismo vector — invariante trivial, no depende de la implementación de atención.
- `TestEncode_MeanPooling_DiffersFromCLS`: con `seqLen=5`, mean y CLS tienen que diferir — un
  smoke test para agarrar una regresión donde `PoolingMean` quede sin cablear y se comporte
  como CLS en silencio.

## Qué NO cambia

`GatedFFN`, `RoPE`, `Softmax`, `LayerNorm`, `MatmulT`, `Add`, el algoritmo de atención
global/local, el orden de las capas — todo lo que ya sirvió para granite sirve tal cual acá.
`webtyp/weightsc` tampoco cambia: es agnóstico al modelo, ya lo verificó `Fix 3` de su propia
ronda de correcciones. `webtyp/weights` tampoco: el artifact es agnóstico a la dimensión y al
número de capas.

## Build que define "done"

```bash
go vet ./...
gotest
GOOS=js GOARCH=wasm go build ./...
tinygo test -target wasm .
```

Los cuatro en verde. `tinygo build -target wasm -o /dev/null .` sigue sin aplicar (paquete
biblioteca, sin `main`), igual que en la etapa 2.
