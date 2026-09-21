package transformer

import (
	"math"

	"webtyp.com/fmt"
)

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

// Weights holds the full encoder parameters in Encode-ready layout.
type Weights struct {
	EmbedNormGamma []float32      // [Dim]
	Layers         []LayerWeights // len == Config.NumLayers
	FinalNormGamma []float32      // [Dim]
}

// GatedFFN computes the gated feed-forward block: wiT projects to 2*ffnDim, splits into
// two halves, applies the activation to the FIRST half and multiplies elementwise by the
// second half (matching ModernBertMLP.forward: input, gate = Wi(x).chunk(2); Wo(act(input)*gate)),
// then woT projects back down. Built on MatmulT and SiLU from kernels.go.
func GatedFFN(dst, src, wiT, woT []float32, seqLen, dim, ffnDim int) error {
	if seqLen <= 0 || dim <= 0 || ffnDim <= 0 {
		return fmt.Err("transformer: invalid dimensions for gatedffn")
	}
	if len(src) < seqLen*dim || len(dst) < seqLen*dim {
		return fmt.Err("transformer: buffer too short for gatedffn")
	}
	if len(wiT) < 2*ffnDim*dim || len(woT) < dim*ffnDim {
		return fmt.Err("transformer: weight buffer too short for gatedffn")
	}

	hidden := make([]float32, seqLen*2*ffnDim)
	if err := MatmulT(hidden, src, wiT, seqLen, dim, 2*ffnDim); err != nil {
		return err
	}

	gated := make([]float32, seqLen*ffnDim)
	for s := 0; s < seqLen; s++ {
		row := hidden[s*2*ffnDim : (s+1)*2*ffnDim]
		first := row[:ffnDim]
		gate := row[ffnDim:]
		if err := SiLU(first); err != nil {
			return err
		}
		out := gated[s*ffnDim : (s+1)*ffnDim]
		for i := 0; i < ffnDim; i++ {
			out[i] = first[i] * gate[i]
		}
	}

	return MatmulT(dst, gated, woT, seqLen, ffnDim, dim)
}

// isGlobalLayer reports whether layer li uses full attention. Layers with
// li % globalEvery == 0 are global; the rest use a sliding local window.
// Matches ModernBertAttention: `if layer_id % global_attn_every_n_layers != 0: local`.
func isGlobalLayer(li, globalEvery int) bool {
	return li%globalEvery == 0
}

// Encode runs one forward pass. tokenEmbeds is seqLen*Dim float32 values — the embedding
// rows FOR THIS SEQUENCE'S TOKENS ONLY, already gathered and dequantized by the caller
// (webtyp/embed's future adapter — NOT this package; do not add embedding-table gather
// code here, and do not import webtyp/weights to do it "properly"). Returns the pooled
// (CLS) vector, Dim elements long. Does NOT L2-normalize — that stays embed's job, same
// boundary as stage 1.
func Encode(cfg Config, w Weights, tokenEmbeds []float32, seqLen int) ([]float32, error) {
	if cfg.NumLayers <= 0 || cfg.Heads <= 0 || cfg.Dim <= 0 || cfg.FFNDim <= 0 {
		return nil, fmt.Err("transformer: invalid config for encode")
	}
	if cfg.Dim%cfg.Heads != 0 {
		return nil, fmt.Err("transformer: dim not divisible by heads for encode")
	}
	if cfg.GlobalEveryNLayers <= 0 || cfg.LocalWindow < 0 {
		return nil, fmt.Err("transformer: invalid attention config for encode")
	}
	if seqLen <= 0 {
		return nil, fmt.Err("transformer: invalid seqLen for encode")
	}
	if len(tokenEmbeds) < seqLen*cfg.Dim {
		return nil, fmt.Err("transformer: tokenEmbeds too short for encode")
	}
	if len(w.Layers) != cfg.NumLayers {
		return nil, fmt.Err("transformer: layer count mismatch for encode")
	}
	if len(w.EmbedNormGamma) < cfg.Dim || len(w.FinalNormGamma) < cfg.Dim {
		return nil, fmt.Err("transformer: norm weight too short for encode")
	}

	dim := cfg.Dim
	heads := cfg.Heads
	headDim := dim / heads
	ffnDim := cfg.FFNDim
	for li := range w.Layers {
		lw := &w.Layers[li]
		if len(lw.AttnNormGamma) > 0 && len(lw.AttnNormGamma) < dim {
			return nil, fmt.Err("transformer: attn norm too short for encode")
		}
		if len(lw.WqkvT) < 3*dim*dim || len(lw.WoT) < dim*dim {
			return nil, fmt.Err("transformer: attn weight too short for encode")
		}
		if len(lw.MlpNormGamma) < dim {
			return nil, fmt.Err("transformer: mlp norm too short for encode")
		}
		if len(lw.WiT) < 2*ffnDim*dim || len(lw.MlpWoT) < dim*ffnDim {
			return nil, fmt.Err("transformer: mlp weight too short for encode")
		}
	}

	h := make([]float32, seqLen*dim)
	copy(h, tokenEmbeds[:seqLen*dim])
	if err := LayerNorm(h, h, w.EmbedNormGamma, nil, dim, cfg.Eps); err != nil {
		return nil, err
	}

	attnIn := make([]float32, seqLen*dim)
	qkv := make([]float32, seqLen*3*dim)
	qh := make([]float32, seqLen*headDim)
	kh := make([]float32, seqLen*headDim)
	vhT := make([]float32, headDim*seqLen)
	scores := make([]float32, seqLen*seqLen)
	ctxHead := make([]float32, seqLen*headDim)
	attnConcat := make([]float32, seqLen*dim)
	attnOut := make([]float32, seqLen*dim)
	mlpIn := make([]float32, seqLen*dim)
	mlpOut := make([]float32, seqLen*dim)

	scale := float32(1.0 / math.Sqrt(float64(headDim)))
	halfWindow := cfg.LocalWindow / 2

	for li := 0; li < cfg.NumLayers; li++ {
		lw := &w.Layers[li]
		global := isGlobalLayer(li, cfg.GlobalEveryNLayers)
		theta := cfg.LocalRopeTheta
		if global {
			theta = cfg.GlobalRopeTheta
		}

		// Attention block with pre-norm (Identity for layer 0: gamma is nil).
		src := h
		if len(lw.AttnNormGamma) > 0 {
			if err := LayerNorm(attnIn, h, lw.AttnNormGamma, nil, dim, cfg.Eps); err != nil {
				return nil, err
			}
			src = attnIn
		}
		if err := MatmulT(qkv, src, lw.WqkvT, seqLen, dim, 3*dim); err != nil {
			return nil, err
		}
		for pos := 0; pos < seqLen; pos++ {
			qPos := qkv[pos*3*dim : pos*3*dim+dim]
			kPos := qkv[pos*3*dim+dim : pos*3*dim+2*dim]
			if err := RoPE(qPos, kPos, pos, theta, dim, heads); err != nil {
				return nil, err
			}
		}
		for hd := 0; hd < heads; hd++ {
			for i := 0; i < seqLen; i++ {
				copy(qh[i*headDim:(i+1)*headDim], qkv[i*3*dim+hd*headDim:i*3*dim+(hd+1)*headDim])
				copy(kh[i*headDim:(i+1)*headDim], qkv[i*3*dim+dim+hd*headDim:i*3*dim+dim+(hd+1)*headDim])
				vRow := qkv[i*3*dim+2*dim+hd*headDim : i*3*dim+2*dim+(hd+1)*headDim]
				for k := 0; k < headDim; k++ {
					vhT[k*seqLen+i] = vRow[k]
				}
			}
			if err := MatmulT(scores, qh, kh, seqLen, headDim, seqLen); err != nil {
				return nil, err
			}
			for i := 0; i < seqLen; i++ {
				row := scores[i*seqLen : (i+1)*seqLen]
				for j := 0; j < seqLen; j++ {
					row[j] *= scale
					if !global && (i-j > halfWindow || j-i > halfWindow) {
						row[j] = -1e30
					}
				}
				if err := Softmax(row); err != nil {
					return nil, err
				}
			}
			if err := MatmulT(ctxHead, scores, vhT, seqLen, seqLen, headDim); err != nil {
				return nil, err
			}
			for i := 0; i < seqLen; i++ {
				copy(attnConcat[i*dim+hd*headDim:i*dim+(hd+1)*headDim], ctxHead[i*headDim:(i+1)*headDim])
			}
		}
		if err := MatmulT(attnOut, attnConcat, lw.WoT, seqLen, dim, dim); err != nil {
			return nil, err
		}
		if err := Add(h, attnOut); err != nil {
			return nil, err
		}

		// MLP block with pre-norm.
		if err := LayerNorm(mlpIn, h, lw.MlpNormGamma, nil, dim, cfg.Eps); err != nil {
			return nil, err
		}
		if err := GatedFFN(mlpOut, mlpIn, lw.WiT, lw.MlpWoT, seqLen, dim, ffnDim); err != nil {
			return nil, err
		}
		if err := Add(h, mlpOut); err != nil {
			return nil, err
		}
	}

	if err := LayerNorm(h, h, w.FinalNormGamma, nil, dim, cfg.Eps); err != nil {
		return nil, err
	}

	pooled := make([]float32, dim)
	copy(pooled, h[:dim])
	return pooled, nil
}
