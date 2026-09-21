package transformer

import (
	"math"

	"webtyp.com/fmt"
	"webtyp.com/vector"
)

// MatmulT computes C[m,n] = A[m,k] · Bᵀ[n,k], i.e. B is stored TRANSPOSED so every
// inner product is over contiguous memory — which is what vector.Dot is fast at.
func MatmulT(dst, a, bT []float32, m, k, n int) error {
	if m <= 0 || k <= 0 || n <= 0 {
		return fmt.Err("transformer: invalid matmul dimensions")
	}
	if len(a) < m*k || len(bT) < n*k || len(dst) < m*n {
		return fmt.Err("transformer: buffer too short for matmul")
	}

	for i := 0; i < m; i++ {
		rowA := a[i*k : (i+1)*k]
		outRow := dst[i*n : (i+1)*n]
		for j := 0; j < n; j++ {
			rowB := bT[j*k : (j+1)*k]
			outRow[j] = vector.Dot(rowA, rowB)
		}
	}
	return nil
}

// LayerNorm normalizes src over the last dimension `dim` with gamma and beta scaling.
// Mean and variance are accumulated in float64 to prevent drift.
func LayerNorm(dst, src, gamma, beta []float32, dim int, eps float32) error {
	if dim <= 0 {
		return fmt.Err("transformer: invalid dimension for layernorm")
	}
	if len(src)%dim != 0 || len(dst) != len(src) {
		return fmt.Err("transformer: buffer size mismatch for layernorm")
	}
	if len(gamma) > 0 && len(gamma) < dim {
		return fmt.Err("transformer: gamma buffer too short for layernorm")
	}
	if len(beta) > 0 && len(beta) < dim {
		return fmt.Err("transformer: beta buffer too short for layernorm")
	}

	numRows := len(src) / dim
	for r := 0; r < numRows; r++ {
		off := r * dim
		sRow := src[off : off+dim]
		dRow := dst[off : off+dim]

		var sum float64
		for _, v := range sRow {
			sum += float64(v)
		}
		mean := sum / float64(dim)

		var varSum float64
		for _, v := range sRow {
			diff := float64(v) - mean
			varSum += diff * diff
		}
		variance := varSum / float64(dim)
		invStd := 1.0 / math.Sqrt(variance+float64(eps))

		for j := 0; j < dim; j++ {
			xNorm := (float64(sRow[j]) - mean) * invStd
			g := 1.0
			if len(gamma) > 0 {
				g = float64(gamma[j])
			}
			b := 0.0
			if len(beta) > 0 {
				b = float64(beta[j])
			}
			dRow[j] = float32(xNorm*g + b)
		}
	}
	return nil
}

// RMSNorm normalizes src over the last dimension `dim` using root-mean-square.
// Sum of squares is accumulated in float64 to prevent precision loss.
func RMSNorm(dst, src, gamma []float32, dim int, eps float32) error {
	if dim <= 0 {
		return fmt.Err("transformer: invalid dimension for rmsnorm")
	}
	if len(src)%dim != 0 || len(dst) != len(src) {
		return fmt.Err("transformer: buffer size mismatch for rmsnorm")
	}
	if len(gamma) > 0 && len(gamma) < dim {
		return fmt.Err("transformer: gamma buffer too short for rmsnorm")
	}

	numRows := len(src) / dim
	for r := 0; r < numRows; r++ {
		off := r * dim
		sRow := src[off : off+dim]
		dRow := dst[off : off+dim]

		var sqSum float64
		for _, v := range sRow {
			vf := float64(v)
			sqSum += vf * vf
		}
		rms := 1.0 / math.Sqrt(sqSum/float64(dim)+float64(eps))

		for j := 0; j < dim; j++ {
			g := 1.0
			if len(gamma) > 0 {
				g = float64(gamma[j])
			}
			dRow[j] = float32(float64(sRow[j]) * rms * g)
		}
	}
	return nil
}

// Softmax computes softmax in-place over `x`.
// It subtracts the maximum value first to prevent numerical overflow (+Inf/NaN).
func Softmax(x []float32) error {
	if len(x) == 0 {
		return nil
	}

	maxVal := x[0]
	for _, v := range x[1:] {
		if v > maxVal {
			maxVal = v
		}
	}

	var sum float64
	for i, v := range x {
		expVal := math.Exp(float64(v - maxVal))
		x[i] = float32(expVal)
		sum += expVal
	}

	if sum > 0 {
		invSum := float32(1.0 / sum)
		for i := range x {
			x[i] *= invSum
		}
	}
	return nil
}

// GELU applies GELU activation function in-place over `x`.
func GELU(x []float32) error {
	for i, v := range x {
		vf := float64(v)
		x[i] = float32(0.5 * vf * (1.0 + math.Erf(vf/math.Sqrt2)))
	}
	return nil
}

// SiLU applies SiLU (Swish-1) activation function in-place over `x`.
func SiLU(x []float32) error {
	for i, v := range x {
		vf := float64(v)
		x[i] = float32(vf / (1.0 + math.Exp(-vf)))
	}
	return nil
}

// Add performs element-wise in-place addition: dst[i] += src[i].
func Add(dst, src []float32) error {
	if len(dst) != len(src) {
		return fmt.Err("transformer: length mismatch for add")
	}
	for i, v := range src {
		dst[i] += v
	}
	return nil
}

// RoPE applies Rotary Position Embedding in-place to q and k for position `pos`.
// theta is the rotary base frequency (granite uses 150000/160000 per layer type).
// Rotation is GPT-NeoX style (first/second half), matching rotate_half in
// modeling_modernbert.py — not interleaved pairs.
func RoPE(q, k []float32, pos int, theta float64, dim, heads int) error {
	if dim <= 0 || heads <= 0 || dim%heads != 0 {
		return fmt.Err("transformer: invalid dimensions for rope")
	}
	headDim := dim / heads
	if headDim%2 != 0 {
		return fmt.Err("transformer: head dimension must be even for rope")
	}

	if len(q) > 0 && len(q) < dim {
		return fmt.Err("transformer: q buffer too short for rope")
	}
	if len(k) > 0 && len(k) < dim {
		return fmt.Err("transformer: k buffer too short for rope")
	}
	if len(q) == 0 && len(k) == 0 {
		return fmt.Err("transformer: no buffers provided for rope")
	}

	rotate := func(v []float32) {
		for h := 0; h < heads; h++ {
			head := v[h*headDim : (h+1)*headDim]
			half := headDim / 2
			for i := 0; i < half; i++ {
				freq := 1.0 / math.Pow(theta, float64(2*i)/float64(headDim))
				angle := float64(pos) * freq
				cosA := float32(math.Cos(angle))
				sinA := float32(math.Sin(angle))

				x0 := head[i]
				x1 := head[i+half]
				head[i] = x0*cosA - x1*sinA
				head[i+half] = x0*sinA + x1*cosA
			}
		}
	}

	if len(q) >= dim {
		rotate(q)
	}
	if len(k) >= dim {
		rotate(k)
	}
	return nil
}
