# Agent Guide — `webtyp/transformer`

Constraints for agents working on this library. **Read this before any change.**
The current work order is [docs/PLAN.md](docs/PLAN.md); the master index is
[`agent/docs/MASTER_PLAN.md`](https://github.com/webtyp/agent/blob/main/docs/MASTER_PLAN.md).

---

## What this library is

Token ids and weights in; **one vector out**. It owns the encoder graph and the numeric kernels that
run it — nothing else. Not tokenization (`webtyp/tokenizer`), not the weight format
(`webtyp/weights`), not the `Embedder` port (`webtyp/embed`, which composes all three), and not
output normalization (that is `embed`'s promise to keep).

Its **primary runtime is a browser tab compiled with TinyGo**. The host is a development
convenience. A change that is green on the host and red under TinyGo is **not done**.

---

## Dependencies: `webtyp.com/vector` and `webtyp.com/fmt`. Nothing else.

Not `weights`, not `tokenizer`, not `syscall/js`. This package receives decoded tensors and
tokenized ids. That is what keeps the whole package testable with plain `go test` — no browser, no
200 MB artifact — and it is the reason it is a separate repository instead of part of `embed`.

Adding a dependency here is a design change, not an implementation detail. It needs a plan.

---

## The builds that define "done"

```bash
go vet ./...
gotest
GOOS=js GOARCH=wasm go build ./...
tinygo test -target wasm .
tinygo test -target wasm -bench=. -benchtime=2s .
```

**`gotest -bench` does not work for benchmarks here.** Passing any argument to `gotest` disables
its wasm stage and hands you a *native* number — roughly 2.5× faster than the target and therefore
a lie for every decision this repo exists to make. Always benchmark with `tinygo test -target wasm`
directly.

---

## Never import these

| Never | Use instead |
|---|---|
| `fmt`, `errors`, `strconv`, `strings` | `webtyp.com/fmt` |
| `math` beyond what TinyGo supports cheaply | plain Go arithmetic; check the wasm size cost |
| `encoding/json`, `net/http`, `context` (stdlib), `os`, `log` | nothing here needs them |
| `syscall/js` | this package never touches the DOM |
| `map[K]V` | a slice scanned linearly |

Tests use the standard library only, with no external assertion packages.

---

## Numeric rules that cost hours when learned late

- **`matmul` is built on `vector.Dot`, never reimplemented.** `Dot` is already unrolled, already
  tested against a naive reference, and is *already the number P1 measured* (~3.3 GFLOPS scalar
  under TinyGo WASM). A matmul built on top inherits that measurement. Unrolling the **outer** loop
  is allowed if a benchmark demands it; reimplementing the inner product is not.
- **`Softmax` must subtract the maximum.** With real attention logits, `exp(x)` reaches `+Inf` and
  the result is a silent `NaN`.
- **`LayerNorm` accumulates mean and variance in `float64`**, even though the data is `float32`.
  Accumulating in `float32` over 384 elements drifts enough to move recall without failing a single
  kernel test.
- **Write every kernel flat first.** Optimize only behind a benchmark that justifies it, exactly as
  `vector` did with `Dot`.
- **A benchmark faster than the derived floor is a broken benchmark**, not a win — the compiler
  most likely eliminated work whose result is unused.

---

## Layout & tests

- Flat hierarchy: Go files in the repo root. No subdirectories for library code.
- Max 500 lines per file; split by domain and rename when exceeded.
- More than 5 test files in the root → move **all** of them to `tests/`, package `tests`.
- Benchmark harnesses live in `*_test.go`, never in the package. If you find yourself exporting a
  type so a benchmark can reach it, stop — chain the kernel functions instead.
- Publish with `gopush 'message'` — never `git commit`/`git push` directly.

---

## Common mistakes to avoid

- Writing the encoder graph before its stage is dispatched. Stage 1 is kernels plus the cost-model
  benchmark; finishing those means you are finished.
- Benchmarking with `gotest -bench` and recording a native number.
- Reaching for SIMD or build tags. Plain Go, one implementation, all targets.
