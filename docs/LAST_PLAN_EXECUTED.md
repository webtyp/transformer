---
PLAN: "fix: webtyp/transformer — Config.Activation, GatedFFN hardcoded SiLU regardless of model"
TAG: v0.1.4
---

> This plan is dispatched via the CodeJob workflow. See skill: agents-workflow.
> Índice maestro: https://github.com/webtyp/agent/blob/main/docs/MASTER_PLAN.md

# Plan — `webtyp/transformer`, `Config.Activation` (bug real, encontrado comparando contra el modelo real)

## El bug, y cómo se encontró

`GatedFFN` llamaba `SiLU(first)` sin condición — asumido correcto para
`granite-embedding-97m-multilingual-r2` en la etapa 2/3 (su `TestEncode_MatchesReference`,
coseno 1.0 contra el modelo real, lo confirma en los hechos), **pero incorrecto para
`bekko-embedding-v1-a8m`**, cuyo `config.json` real dice `"hidden_activation": "gelu"` —
verificado leyendo el archivo ahora, no de memoria.

**Cómo se encontró:** `webtyp/embed`'s `TestStaticEmbedder_MatchesReference` (docs/PLAN.md
Cambio 5 de ese repo) comparó la salida real de `StaticEmbedder` contra vectores de
referencia generados corriendo el modelo real
(`AutoModel.from_pretrained("hotchpotch/bekko-embedding-v1-a8m")`, `transformers`+`torch`
CPU) — coseno ~0.91–0.92 en las tres oraciones de prueba, consistente pero claramente mal (no
es ruido de cuantización int8, que da >0.999 típicamente). La lista de sospechosos que el
plan de `embed` ya tenía escrita (transposición de pesos, indexado de capa 0, tokenizer)
todos descartados por inspección — la config real de Bekko señaló la activación como la
causa antes de necesitar más diagnóstico.

## El fix

`Config` gana un campo `Activation Activation`, con `ActivationSiLU` como valor cero
(preserva el comportamiento de Granite sin tocar ningún caller existente) y
`ActivationGELU` (exacta, basada en `erf` — la que ya existía en `kernels.go` como `GELU`,
sin usar hasta ahora). `GatedFFN` gana un parámetro `act Activation` y rama entre
`SiLU`/`GELU` según ese valor; `Encode` pasa `cfg.Activation` en su única llamada a
`GatedFFN`.

**Test nuevo, `TestGatedFFN_ActivationSelectsRealFunction`**: corre `GatedFFN` con
`ActivationSiLU` y con `ActivationGELU` sobre el mismo input y falla si el resultado es
idéntico — exactamente la regresión que este bug fue (un parámetro que existe pero no se usa).

## Qué NO cambia

`granite-embedding-97m-multilingual-r2` sigue sin especificar `Activation` en su `Config` —
zero value, `ActivationSiLU`, mismo comportamiento de siempre. `TestEncode_MatchesReference`
(el fixture real de Granite) sigue en coseno 1.0 después de este cambio — verificado, no
asumido.

## Build que define "done"

```bash
go vet ./...
gotest
GOOS=js GOARCH=wasm go build ./...
tinygo test -target wasm .
```

Los cuatro en verde. Cadena completa verificada además end-to-end en `webtyp/embed`: con este
fix, `TestStaticEmbedder_MatchesReference` pasa contra el modelo real (ver ese repo).
