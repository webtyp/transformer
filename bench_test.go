package transformer

import (
	"math/rand"
	"testing"
)

// benchSink keeps the benchmark honest: Encode's result is folded here so the
// compiler cannot eliminate the forward pass as dead work.
var benchSink float32

// BenchmarkEncode_20x12x384 runs the real Encode forward pass at granite's shape —
// 20 tokens, 12 layers, 384 dims, 12 heads, FFN 1536 with the gated SiLU block —
// over fixed synthetic weights. This replaces the stage-1 synthetic harness, whose
// cost-model purpose (measuring the stage-3 floor) is fulfilled.
func BenchmarkEncode_20x12x384(b *testing.B) {
	const (
		seqLen = 20
		dim    = 384
	)

	cfg := Config{
		NumLayers:          12,
		Heads:              12,
		Dim:                dim,
		FFNDim:             1536,
		GlobalEveryNLayers: 3,
		LocalWindow:        128,
		GlobalRopeTheta:    150000.0,
		LocalRopeTheta:     160000.0,
		Eps:                1e-5,
	}

	rnd := rand.New(rand.NewSource(42))
	news := func(n int) []float32 {
		buf := make([]float32, n)
		for i := range buf {
			buf[i] = float32(rnd.NormFloat64() * 0.1)
		}
		return buf
	}
	ones := func(n int) []float32 {
		buf := make([]float32, n)
		for i := range buf {
			buf[i] = 1.0
		}
		return buf
	}

	w := Weights{EmbedNormGamma: ones(dim), FinalNormGamma: ones(dim)}
	for li := 0; li < cfg.NumLayers; li++ {
		lw := LayerWeights{
			WqkvT:        news(3 * dim * dim),
			WoT:          news(dim * dim),
			MlpNormGamma: ones(dim),
			WiT:          news(2 * cfg.FFNDim * dim),
			MlpWoT:       news(dim * cfg.FFNDim),
		}
		if li > 0 {
			lw.AttnNormGamma = ones(dim)
		}
		w.Layers = append(w.Layers, lw)
	}
	embeds := news(seqLen * dim)

	b.ResetTimer()
	for n := 0; n < b.N; n++ {
		got, err := Encode(cfg, w, embeds, seqLen)
		if err != nil {
			b.Fatalf("Encode: %v", err)
		}
		for _, v := range got {
			benchSink += v
		}
	}
}
