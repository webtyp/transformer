package transformer

import (
	"math/rand"
	"testing"

	"webtyp.com/vector"
)

// BenchmarkEncode_20x12x384 chains the kernels twelve times at the shape of the
// leading candidate — 20 tokens, 384 dims, 6 heads, FFN 1536 — over random weights.
//
// This is a COST MODEL, not an encoder. It makes no correctness claim: the order is
// the generic one (QKV projection, attention, softmax, output projection, FFN,
// activation, two norms, two residuals), not any specific model's. Getting RoPE
// placement, masking or GeGLU-vs-SiLU exactly right belongs to stage 2 and does not
// change the FLOP count this measures.
func BenchmarkEncode_20x12x384(b *testing.B) {
	const (
		numLayers = 12
		seqLen    = 20
		dim       = 384
		heads     = 6
		headDim   = dim / heads // 64
		ffnDim    = 1536
	)

	rnd := rand.New(rand.NewSource(42))
	fillRand := func(buf []float32) {
		for i := range buf {
			buf[i] = float32(rnd.NormFloat64() * 0.1)
		}
	}

	x := make([]float32, seqLen*dim)
	fillRand(x)

	// Pre-allocate weight matrices
	wQKV_T := make([]float32, 3*dim*dim)
	fillRand(wQKV_T)

	wO_T := make([]float32, dim*dim)
	fillRand(wO_T)

	wUp_T := make([]float32, ffnDim*dim)
	fillRand(wUp_T)

	wDown_T := make([]float32, dim*ffnDim)
	fillRand(wDown_T)

	gamma := make([]float32, dim)
	beta := make([]float32, dim)
	for i := 0; i < dim; i++ {
		gamma[i] = 1.0
		beta[i] = 0.0
	}

	// Pre-allocate intermediate operational buffers
	xNorm1 := make([]float32, seqLen*dim)
	xNorm2 := make([]float32, seqLen*dim)
	qkv := make([]float32, seqLen*3*dim)
	attnConcat := make([]float32, seqLen*dim)
	attnOut := make([]float32, seqLen*dim)
	scores := make([]float32, seqLen*seqLen)
	vHeadT := make([]float32, headDim*seqLen)
	headCtx := make([]float32, seqLen*headDim)
	ffnHidden := make([]float32, seqLen*ffnDim)
	ffnOut := make([]float32, seqLen*dim)

	b.ResetTimer()

	for n := 0; n < b.N; n++ {
		for layer := 0; layer < numLayers; layer++ {
			// 1. LayerNorm 1
			_ = LayerNorm(xNorm1, x, gamma, beta, dim, 1e-5)

			// 2. QKV Projection: [20, 384] x [1152, 384]^T -> [20, 1152]
			_ = MatmulT(qkv, xNorm1, wQKV_T, seqLen, dim, 3*dim)

			// 3. RoPE on Q and K for each token position
			for pos := 0; pos < seqLen; pos++ {
				qPos := qkv[pos*(3*dim) : pos*(3*dim)+dim]
				kPos := qkv[pos*(3*dim)+dim : pos*(3*dim)+2*dim]
				_ = RoPE(qPos, kPos, pos, dim, heads)
			}

			// 4. Multi-head Attention
			for h := 0; h < heads; h++ {
				// Compute Attention Scores: Q_h [20, 64] x K_h^T [20, 64]^T -> Scores [20, 20]
				for i := 0; i < seqLen; i++ {
					qRow := qkv[i*(3*dim)+h*headDim : i*(3*dim)+(h+1)*headDim]
					for j := 0; j < seqLen; j++ {
						kRow := qkv[j*(3*dim)+dim+h*headDim : j*(3*dim)+dim+(h+1)*headDim]
						scores[i*seqLen+j] = vector.Dot(qRow, kRow)
					}
				}

				// Softmax per row
				for i := 0; i < seqLen; i++ {
					_ = Softmax(scores[i*seqLen : (i+1)*seqLen])
				}

				// Prepare transposed V_h [64, 20]
				for j := 0; j < seqLen; j++ {
					vRow := qkv[j*(3*dim)+2*dim+h*headDim : j*(3*dim)+2*dim+(h+1)*headDim]
					for kIdx := 0; kIdx < headDim; kIdx++ {
						vHeadT[kIdx*seqLen+j] = vRow[kIdx]
					}
				}

				// Context = Scores [20, 20] x V_h^T [64, 20]^T -> [20, 64]
				_ = MatmulT(headCtx, scores, vHeadT, seqLen, seqLen, headDim)

				// Copy headCtx into attnConcat [20, 384]
				for i := 0; i < seqLen; i++ {
					copy(attnConcat[i*dim+h*headDim:i*dim+(h+1)*headDim], headCtx[i*headDim:(i+1)*headDim])
				}
			}

			// 5. Output Projection: [20, 384] x [384, 384]^T -> [20, 384]
			_ = MatmulT(attnOut, attnConcat, wO_T, seqLen, dim, dim)

			// 6. Residual Add 1
			_ = Add(x, attnOut)

			// 7. LayerNorm 2
			_ = LayerNorm(xNorm2, x, gamma, beta, dim, 1e-5)

			// 8. FFN Up: [20, 384] x [1536, 384]^T -> [20, 1536]
			_ = MatmulT(ffnHidden, xNorm2, wUp_T, seqLen, dim, ffnDim)

			// GELU Activation
			_ = GELU(ffnHidden)

			// FFN Down: [20, 1536] x [384, 1536]^T -> [20, 384]
			_ = MatmulT(ffnOut, ffnHidden, wDown_T, seqLen, ffnDim, dim)

			// 9. Residual Add 2
			_ = Add(x, ffnOut)
		}
	}
}
