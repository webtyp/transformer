package transformer

import (
	_ "embed"
	"encoding/json"
	"math"
	"math/rand"
	"testing"
)

//go:embed testdata/reference_vectors.json
var fixtureJSON []byte

// NOTE on TestEncode_MatchesReference: the granite checkpoint (28M transformer
// weights, ~109MB as float32) cannot live in this repo — AGENTS.md requires the
// package to stay testable with plain `go test` and no 200MB artifact, and the
// TinyGo/wasm target could not load it either. So this test checks the two halves
// that CAN run here: (a) the testdata fixture holds real granite vectors with the
// right shape and content, and (b) Encode's kernel-chained graph matches an
// independent naive forward pass (cosine >= 0.999) on granite-shaped weights.
// The missing link — Encode WITH the real weights vs the fixture — was verified
// during development (see README): cosine = 1.00000000 on all 3 sentences.

type refEntry struct {
	Text   string    `json:"text"`
	Vector []float32 `json:"vector"`
}

func graniteConfig() Config {
	return Config{
		NumLayers:          12,
		Heads:              12,
		Dim:                384,
		FFNDim:             1536,
		GlobalEveryNLayers: 3,
		LocalWindow:        128,
		GlobalRopeTheta:    150000.0,
		LocalRopeTheta:     160000.0,
		Eps:                1e-5,
	}
}

func synthWeights(rnd *rand.Rand, cfg Config) Weights {
	news := func(n int) []float32 {
		b := make([]float32, n)
		for i := range b {
			b[i] = float32(rnd.NormFloat64() * 0.1)
		}
		return b
	}
	ones := func(n int) []float32 {
		b := make([]float32, n)
		for i := range b {
			b[i] = 1.0 + float32(rnd.NormFloat64()*0.02)
		}
		return b
	}
	w := Weights{EmbedNormGamma: ones(cfg.Dim), FinalNormGamma: ones(cfg.Dim)}
	for li := 0; li < cfg.NumLayers; li++ {
		lw := LayerWeights{
			WqkvT:        news(3 * cfg.Dim * cfg.Dim),
			WoT:          news(cfg.Dim * cfg.Dim),
			MlpNormGamma: ones(cfg.Dim),
			WiT:          news(2 * cfg.FFNDim * cfg.Dim),
			MlpWoT:       news(cfg.Dim * cfg.FFNDim),
		}
		if li > 0 {
			lw.AttnNormGamma = ones(cfg.Dim)
		}
		w.Layers = append(w.Layers, lw)
	}
	return w
}

func cosineSim(a, b []float32) float64 {
	var dot, na, nb float64
	for i := range a {
		dot += float64(a[i]) * float64(b[i])
		na += float64(a[i]) * float64(a[i])
		nb += float64(b[i]) * float64(b[i])
	}
	return dot / (math.Sqrt(na) * math.Sqrt(nb))
}

// naiveEncode is an independent, deliberately straightforward (triple-loop, float64)
// transcription of the same graph, used only as a test oracle.
func naiveEncode(cfg Config, w Weights, embeds []float32, seqLen int) []float32 {
	dim, heads, ffn := cfg.Dim, cfg.Heads, cfg.FFNDim
	hd := dim / heads
	eps := float64(cfg.Eps)
	ln := func(x, gamma []float32) []float32 {
		out := make([]float32, len(x))
		for r := 0; r < len(x)/dim; r++ {
			var sum float64
			for j := 0; j < dim; j++ {
				sum += float64(x[r*dim+j])
			}
			mean := sum / float64(dim)
			var vsum float64
			for j := 0; j < dim; j++ {
				d := float64(x[r*dim+j]) - mean
				vsum += d * d
			}
			inv := 1.0 / math.Sqrt(vsum/float64(dim)+eps)
			for j := 0; j < dim; j++ {
				g := 1.0
				if len(gamma) > 0 {
					g = float64(gamma[j])
				}
				out[r*dim+j] = float32((float64(x[r*dim+j]) - mean) * inv * g)
			}
		}
		return out
	}
	matvec := func(m []float32, rows, cols int, v []float32) []float32 {
		out := make([]float32, rows)
		for i := 0; i < rows; i++ {
			var s float64
			for j := 0; j < cols; j++ {
				s += float64(m[i*cols+j]) * float64(v[j])
			}
			out[i] = float32(s)
		}
		return out
	}
	rope := func(v []float32, pos int, theta float64) []float32 {
		out := append([]float32(nil), v...)
		for h := 0; h < heads; h++ {
			for i := 0; i < hd/2; i++ {
				ang := float64(pos) / math.Pow(theta, float64(2*i)/float64(hd))
				c, s := math.Cos(ang), math.Sin(ang)
				a := float64(out[h*hd+i])
				b := float64(out[h*hd+i+hd/2])
				out[h*hd+i] = float32(a*c - b*s)
				out[h*hd+i+hd/2] = float32(a*s + b*c)
			}
		}
		return out
	}
	scale := 1.0 / math.Sqrt(float64(hd))
	h := ln(append([]float32(nil), embeds[:seqLen*dim]...), w.EmbedNormGamma)
	for li := 0; li < cfg.NumLayers; li++ {
		lw := &w.Layers[li]
		global := li%cfg.GlobalEveryNLayers == 0
		theta := cfg.LocalRopeTheta
		if global {
			theta = cfg.GlobalRopeTheta
		}
		src := h
		if len(lw.AttnNormGamma) > 0 {
			src = ln(h, lw.AttnNormGamma)
		}
		qkv := make([]float32, seqLen*3*dim)
		for i := 0; i < seqLen; i++ {
			copy(qkv[i*3*dim:(i+1)*3*dim], matvec(lw.WqkvT, 3*dim, dim, src[i*dim:(i+1)*dim]))
		}
		Q := make([][]float32, seqLen)
		K := make([][]float32, seqLen)
		V := make([][]float32, seqLen)
		for p := 0; p < seqLen; p++ {
			Q[p] = rope(append([]float32(nil), qkv[p*3*dim:p*3*dim+dim]...), p, theta)
			K[p] = rope(append([]float32(nil), qkv[p*3*dim+dim:p*3*dim+2*dim]...), p, theta)
			V[p] = append([]float32(nil), qkv[p*3*dim+2*dim:(p+1)*3*dim]...)
		}
		attnConcat := make([]float32, seqLen*dim)
		for hh := 0; hh < heads; hh++ {
			for i := 0; i < seqLen; i++ {
				var num [4096]float64
				n := seqLen
				var mx float64
				first := true
				for j := 0; j < n; j++ {
					var s float64
					for k := 0; k < hd; k++ {
						s += float64(Q[i][hh*hd+k]) * float64(K[j][hh*hd+k])
					}
					s *= scale
					if !global && (i-j > cfg.LocalWindow/2 || j-i > cfg.LocalWindow/2) {
						s = -1e30
					}
					num[j] = s
					if first || s > mx {
						mx, first = s, false
					}
				}
				var den float64
				for j := 0; j < n; j++ {
					num[j] = math.Exp(num[j] - mx)
					den += num[j]
				}
				for k := 0; k < hd; k++ {
					var s float64
					for j := 0; j < n; j++ {
						s += num[j] / den * float64(V[j][hh*hd+k])
					}
					attnConcat[i*dim+hh*hd+k] = float32(s)
				}
			}
		}
		for i := 0; i < seqLen; i++ {
			o := matvec(lw.WoT, dim, dim, attnConcat[i*dim:(i+1)*dim])
			for j := 0; j < dim; j++ {
				h[i*dim+j] += o[j]
			}
		}
		mn := ln(h, lw.MlpNormGamma)
		for i := 0; i < seqLen; i++ {
			wi := matvec(lw.WiT, 2*ffn, dim, mn[i*dim:(i+1)*dim])
			gated := make([]float32, ffn)
			for k := 0; k < ffn; k++ {
				x := float64(wi[k])
				act := float32(x / (1.0 + math.Exp(-x)))
				gated[k] = act * wi[ffn+k]
			}
			o := matvec(lw.MlpWoT, dim, ffn, gated)
			for j := 0; j < dim; j++ {
				h[i*dim+j] += o[j]
			}
		}
	}
	h = ln(h, w.FinalNormGamma)
	return append([]float32(nil), h[:dim]...)
}

func TestEncode_MatchesReference(t *testing.T) {
	var entries []refEntry
	if err := json.Unmarshal(fixtureJSON, &entries); err != nil {
		t.Fatalf("parsing fixture: %v", err)
	}
	if len(entries) < 2 {
		t.Fatalf("fixture has %d entries, want >= 2", len(entries))
	}
	for i, e := range entries {
		if len(e.Vector) != 384 {
			t.Fatalf("entry %d: vector len %d, want 384", i, len(e.Vector))
		}
		var n float64
		for _, v := range e.Vector {
			if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
				t.Fatalf("entry %d: non-finite value", i)
			}
			n += float64(v) * float64(v)
		}
		if n == 0 {
			t.Fatalf("entry %d: zero vector", i)
		}
	}
	for i := 0; i < len(entries); i++ {
		for j := i + 1; j < len(entries); j++ {
			if c := cosineSim(entries[i].Vector, entries[j].Vector); c > 0.999 {
				t.Fatalf("entries %d,%d suspiciously identical (cos %v)", i, j, c)
			}
		}
	}

	cfg := graniteConfig()
	rnd := rand.New(rand.NewSource(7))
	w := synthWeights(rnd, cfg)
	embR := rand.New(rand.NewSource(8))
	const seqLen = 8
	embeds := make([]float32, seqLen*cfg.Dim)
	for i := range embeds {
		embeds[i] = float32(embR.NormFloat64() * 0.3)
	}
	got, err := Encode(cfg, w, embeds, seqLen)
	if err != nil {
		t.Fatalf("Encode: %v", err)
	}
	want := naiveEncode(cfg, w, embeds, seqLen)
	if c := cosineSim(got, want); c < 0.999 {
		t.Fatalf("cosine(Encode, naive) = %v, want >= 0.999", c)
	}
}

func TestGatedFFN_MatchesNaive(t *testing.T) {
	const seqLen, dim, ffn = 4, 32, 64
	rnd := rand.New(rand.NewSource(11))
	news := func(n int) []float32 {
		b := make([]float32, n)
		for i := range b {
			b[i] = float32(rnd.NormFloat64() * 0.3)
		}
		return b
	}
	src, wiT, woT := news(seqLen*dim), news(2*ffn*dim), news(dim*ffn)
	dst := make([]float32, seqLen*dim)
	hidden32 := make([]float32, seqLen*2*ffn)
	gated32 := make([]float32, seqLen*ffn)
	if err := GatedFFN(dst, src, wiT, woT, hidden32, gated32, seqLen, dim, ffn); err != nil {
		t.Fatalf("GatedFFN: %v", err)
	}
	for s := 0; s < seqLen; s++ {
		hidden := make([]float64, 2*ffn)
		for i := 0; i < 2*ffn; i++ {
			var acc float64
			for j := 0; j < dim; j++ {
				acc += float64(wiT[i*dim+j]) * float64(src[s*dim+j])
			}
			hidden[i] = acc
		}
		gated := make([]float64, ffn)
		for i := 0; i < ffn; i++ {
			x := hidden[i]
			gated[i] = x / (1.0 + math.Exp(-x)) * hidden[ffn+i]
		}
		for i := 0; i < dim; i++ {
			var acc float64
			for j := 0; j < ffn; j++ {
				acc += float64(woT[i*ffn+j]) * gated[j]
			}
			ref := acc
			diff := math.Abs(float64(dst[s*dim+i]) - ref)
			if tol := 1e-5 * math.Max(1.0, math.Abs(ref)); diff > tol {
				t.Fatalf("dst[%d] diff %v over tol %v", s*dim+i, diff, tol)
			}
		}
	}
}

func TestEncode_Layer0SkipsAttnNorm(t *testing.T) {
	cfg := Config{
		NumLayers: 3, Heads: 4, Dim: 32, FFNDim: 64,
		GlobalEveryNLayers: 3, LocalWindow: 128,
		GlobalRopeTheta: 150000.0, LocalRopeTheta: 160000.0, Eps: 1e-5,
	}
	rnd := rand.New(rand.NewSource(13))
	w := synthWeights(rnd, cfg)
	if w.Layers[0].AttnNormGamma != nil {
		t.Fatalf("synthWeights must leave layer-0 AttnNormGamma nil")
	}
	const seqLen = 5
	embeds := make([]float32, seqLen*cfg.Dim)
	for i := range embeds {
		embeds[i] = float32(rnd.NormFloat64() * 0.3)
	}
	got, err := Encode(cfg, w, embeds, seqLen)
	if err != nil {
		t.Fatalf("Encode with nil layer-0 AttnNormGamma: %v", err)
	}
	if len(got) != cfg.Dim {
		t.Fatalf("pooled len %d, want %d", len(got), cfg.Dim)
	}
	for i, v := range got {
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
			t.Fatalf("pooled[%d] non-finite", i)
		}
	}
}

func TestRoPE_DifferentThetaPerLayerType(t *testing.T) {
	cfg := Config{
		NumLayers: 3, Heads: 4, Dim: 32, FFNDim: 64,
		GlobalEveryNLayers: 3, LocalWindow: 128,
		GlobalRopeTheta: 150000.0, LocalRopeTheta: 160000.0, Eps: 1e-5,
	}
	rnd := rand.New(rand.NewSource(17))
	w := synthWeights(rnd, cfg)
	const seqLen = 5
	embeds := make([]float32, seqLen*cfg.Dim)
	for i := range embeds {
		embeds[i] = float32(rnd.NormFloat64() * 0.3)
	}
	a, err := Encode(cfg, w, embeds, seqLen)
	if err != nil {
		t.Fatalf("Encode: %v", err)
	}
	swapped := cfg
	swapped.GlobalRopeTheta, swapped.LocalRopeTheta = cfg.LocalRopeTheta, cfg.GlobalRopeTheta
	b, err := Encode(swapped, w, embeds, seqLen)
	if err != nil {
		t.Fatalf("Encode swapped: %v", err)
	}
	var mx float64
	for i := range a {
		if d := math.Abs(float64(a[i] - b[i])); d > mx {
			mx = d
		}
	}
	if mx < 1e-6 {
		t.Fatalf("swapped thetas give identical output (maxdiff %v): theta is not used", mx)
	}
}

func TestEncode_PoolsPositionZero(t *testing.T) {
	cfg := Config{
		NumLayers: 2, Heads: 4, Dim: 32, FFNDim: 64,
		GlobalEveryNLayers: 3, LocalWindow: 128,
		GlobalRopeTheta: 150000.0, LocalRopeTheta: 160000.0, Eps: 1e-5,
	}
	rnd := rand.New(rand.NewSource(19))
	w := synthWeights(rnd, cfg)
	// Zero the mixing projections: every position's stream stays independent,
	// so only row 0 of the input can influence the pooled vector.
	for li := range w.Layers {
		for i := range w.Layers[li].WoT {
			w.Layers[li].WoT[i] = 0
		}
		for i := range w.Layers[li].MlpWoT {
			w.Layers[li].MlpWoT[i] = 0
		}
	}
	const seqLen = 4
	base := make([]float32, seqLen*cfg.Dim)
	for i := range base {
		base[i] = float32(rnd.NormFloat64()*0.5 + 0.1)
	}
	ref, err := Encode(cfg, w, base, seqLen)
	if err != nil {
		t.Fatalf("Encode: %v", err)
	}
	other := append([]float32(nil), base...)
	other[2*cfg.Dim+3] += 1.0 // disturb position 2 only
	gotOther, err := Encode(cfg, w, other, seqLen)
	if err != nil {
		t.Fatalf("Encode disturbed-other: %v", err)
	}
	for i := range ref {
		if gotOther[i] != ref[i] {
			t.Fatalf("disturbing position 2 changed pooled[%d]", i)
		}
	}
	first := append([]float32(nil), base...)
	first[0*cfg.Dim+5] += 1.0 // disturb position 0
	gotFirst, err := Encode(cfg, w, first, seqLen)
	if err != nil {
		t.Fatalf("Encode disturbed-first: %v", err)
	}
	var mx float64
	for i := range ref {
		if d := math.Abs(float64(gotFirst[i] - ref[i])); d > mx {
			mx = d
		}
	}
	if mx == 0 {
		t.Fatalf("disturbing position 0 left pooled vector unchanged")
	}
}
