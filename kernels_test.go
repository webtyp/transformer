package transformer

import (
	"math"
	"testing"

	"webtyp.com/vector"
)

func TestMatmulT_MatchesNaive(t *testing.T) {
	m, k, n := 4, 8, 5
	a := make([]float32, m*k)
	bT := make([]float32, n*k)
	for i := range a {
		a[i] = float32(i+1) * 0.1
	}
	for i := range bT {
		bT[i] = float32(i+1) * 0.05
	}

	dst := make([]float32, m*n)
	err := MatmulT(dst, a, bT, m, k, n)
	if err != nil {
		t.Fatalf("MatmulT returned unexpected error: %v", err)
	}

	// Naive reference calculation
	for i := 0; i < m; i++ {
		for j := 0; j < n; j++ {
			var expected float32
			for p := 0; p < k; p++ {
				expected += a[i*k+p] * bT[j*k+p]
			}
			actual := dst[i*n+j]
			diff := math.Abs(float64(actual - expected))
			if diff > 1e-5*math.Max(1.0, float64(math.Abs(float64(expected)))) {
				t.Errorf("At dst[%d,%d]: got %v, expected %v (diff %v)", i, j, actual, expected, diff)
			}
		}
	}
}

func TestMatmulT_Dimensions(t *testing.T) {
	dst := make([]float32, 10)
	a := make([]float32, 10)
	bT := make([]float32, 10)

	// Dimension mismatch (dst too short for 4x4=16)
	err := MatmulT(dst, a, bT, 4, 2, 4)
	if err == nil {
		t.Errorf("Expected error for buffer too short, got nil")
	}

	// Invalid zero/negative dimensions
	err = MatmulT(dst, a, bT, 0, 2, 2)
	if err == nil {
		t.Errorf("Expected error for m <= 0, got nil")
	}
}

func TestSoftmax_SumsToOne(t *testing.T) {
	x := []float32{1.0, 2.0, 3.0, 4.0, 5.0}
	err := Softmax(x)
	if err != nil {
		t.Fatalf("Softmax returned unexpected error: %v", err)
	}

	var sum float64
	for _, v := range x {
		sum += float64(v)
	}
	if math.Abs(sum-1.0) > 1e-6 {
		t.Errorf("Softmax sum = %v, expected ~1.0", sum)
	}
}

func TestSoftmax_LargeLogitsNoNaN(t *testing.T) {
	x := []float32{-100.0, 100.0, 50.0, -50.0, 1000.0}
	err := Softmax(x)
	if err != nil {
		t.Fatalf("Softmax returned unexpected error: %v", err)
	}

	for i, v := range x {
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
			t.Fatalf("Softmax produce NaN or Inf at index %d: %v", i, v)
		}
	}

	// The max element (1000.0) should dominate and sum should be 1.0
	var sum float64
	for _, v := range x {
		sum += float64(v)
	}
	if math.Abs(sum-1.0) > 1e-6 {
		t.Errorf("Softmax sum = %v, expected ~1.0", sum)
	}
}

func TestSoftmax_Uniform(t *testing.T) {
	n := 5
	x := []float32{2.5, 2.5, 2.5, 2.5, 2.5}
	err := Softmax(x)
	if err != nil {
		t.Fatalf("Softmax returned unexpected error: %v", err)
	}

	expected := float32(1.0 / float32(n))
	for i, v := range x {
		if math.Abs(float64(v-expected)) > 1e-6 {
			t.Errorf("Index %d: got %v, expected %v", i, v, expected)
		}
	}
}

func TestLayerNorm_ZeroMeanUnitVar(t *testing.T) {
	dim := 64
	src := make([]float32, dim)
	for i := range src {
		src[i] = float32(i)*0.5 - 10.0
	}
	dst := make([]float32, dim)

	// No gamma/beta scaling
	err := LayerNorm(dst, src, nil, nil, dim, 1e-5)
	if err != nil {
		t.Fatalf("LayerNorm returned error: %v", err)
	}

	var sum float64
	for _, v := range dst {
		sum += float64(v)
	}
	mean := sum / float64(dim)

	var varSum float64
	for _, v := range dst {
		diff := float64(v) - mean
		varSum += diff * diff
	}
	variance := varSum / float64(dim)

	if math.Abs(mean) > 1e-5 {
		t.Errorf("LayerNorm mean = %v, expected ~0", mean)
	}
	if math.Abs(variance-1.0) > 1e-4 {
		t.Errorf("LayerNorm variance = %v, expected ~1", variance)
	}
}

func TestLayerNorm_Float64Accumulation(t *testing.T) {
	dim := 384
	src := make([]float32, dim)
	gamma := make([]float32, dim)
	beta := make([]float32, dim)
	for i := range src {
		src[i] = 1000.0 + float32(i)*0.01
		gamma[i] = 1.0
		beta[i] = 0.0
	}
	dst := make([]float32, dim)

	err := LayerNorm(dst, src, gamma, beta, dim, 1e-5)
	if err != nil {
		t.Fatalf("LayerNorm returned error: %v", err)
	}

	// Compute float64 reference
	var sum float64
	for _, v := range src {
		sum += float64(v)
	}
	mean64 := sum / float64(dim)

	var varSum float64
	for _, v := range src {
		diff := float64(v) - mean64
		varSum += diff * diff
	}
	variance64 := varSum / float64(dim)
	invStd64 := 1.0 / math.Sqrt(variance64+1e-5)

	for i := 0; i < dim; i++ {
		ref := float32((float64(src[i]) - mean64) * invStd64)
		diff := math.Abs(float64(dst[i] - ref))
		if diff > 1e-6 {
			t.Errorf("At index %d: dst=%v, ref=%v (diff %v)", i, dst[i], ref, diff)
		}
	}
}

func TestGELU_KnownValues(t *testing.T) {
	// GELU(0) = 0
	// GELU(large negative) = ~0
	// GELU(large positive) = ~x
	x := []float32{0.0, -10.0, 10.0, 1.0}
	err := GELU(x)
	if err != nil {
		t.Fatalf("GELU returned error: %v", err)
	}

	if math.Abs(float64(x[0])) > 1e-6 {
		t.Errorf("GELU(0) = %v, expected 0", x[0])
	}
	if math.Abs(float64(x[1])) > 1e-5 {
		t.Errorf("GELU(-10) = %v, expected ~0", x[1])
	}
	if math.Abs(float64(x[2]-10.0)) > 1e-5 {
		t.Errorf("GELU(10) = %v, expected ~10", x[2])
	}

	// GELU(1) = 1 * Phi(1) = 1 * 0.8413447 = ~0.8413447
	expected1 := float32(0.8413447)
	if math.Abs(float64(x[3]-expected1)) > 1e-5 {
		t.Errorf("GELU(1) = %v, expected ~%v", x[3], expected1)
	}
}

func TestSiLU_KnownValues(t *testing.T) {
	// SiLU(0) = 0
	// SiLU(-10) = -10 / (1 + e^10) ~ -0.000454
	// SiLU(10) = 10 / (1 + e^-10) ~ 9.999546
	x := []float32{0.0, -10.0, 10.0, 1.0}
	err := SiLU(x)
	if err != nil {
		t.Fatalf("SiLU returned error: %v", err)
	}

	if math.Abs(float64(x[0])) > 1e-6 {
		t.Errorf("SiLU(0) = %v, expected 0", x[0])
	}
	if math.Abs(float64(x[1]-(-0.0004539787))) > 1e-5 {
		t.Errorf("SiLU(-10) = %v, expected ~ -0.0004539787", x[1])
	}
	if math.Abs(float64(x[2]-9.999546)) > 1e-5 {
		t.Errorf("SiLU(10) = %v, expected ~9.999546", x[2])
	}

	// SiLU(1) = 1 / (1 + e^-1) = 1 / (1 + 0.36787944) = ~0.7310586
	expected1 := float32(0.7310586)
	if math.Abs(float64(x[3]-expected1)) > 1e-5 {
		t.Errorf("SiLU(1) = %v, expected ~%v", x[3], expected1)
	}
}

func TestRoPE_RotationPreservesNorm(t *testing.T) {
	dim := 64
	heads := 4
	q := make([]float32, dim)
	k := make([]float32, dim)

	for i := range q {
		q[i] = float32(i+1) * 0.1
		k[i] = float32(i+1) * 0.2
	}

	qNormBefore := vector.Norm(q)
	kNormBefore := vector.Norm(k)

	err := RoPE(q, k, 5, dim, heads)
	if err != nil {
		t.Fatalf("RoPE returned error: %v", err)
	}

	qNormAfter := vector.Norm(q)
	kNormAfter := vector.Norm(k)

	if math.Abs(float64(qNormBefore-qNormAfter)) > 1e-5 {
		t.Errorf("q norm before=%v, after=%v", qNormBefore, qNormAfter)
	}
	if math.Abs(float64(kNormBefore-kNormAfter)) > 1e-5 {
		t.Errorf("k norm before=%v, after=%v", kNormBefore, kNormAfter)
	}
}

func TestRoPE_PositionZeroIsIdentity(t *testing.T) {
	dim := 64
	heads := 4
	q := make([]float32, dim)
	qOrig := make([]float32, dim)
	for i := range q {
		q[i] = float32(i+1) * 0.33
		qOrig[i] = q[i]
	}

	err := RoPE(q, nil, 0, dim, heads)
	if err != nil {
		t.Fatalf("RoPE returned error: %v", err)
	}

	for i := range q {
		if math.Abs(float64(q[i]-qOrig[i])) > 1e-6 {
			t.Errorf("At index %d: got %v, expected %v (position 0 modified vector)", i, q[i], qOrig[i])
		}
	}
}

func TestAdd_InPlace(t *testing.T) {
	dst := []float32{1.0, 2.0, 3.0}
	src := []float32{4.0, 5.0, 6.0}

	err := Add(dst, src)
	if err != nil {
		t.Fatalf("Add returned error: %v", err)
	}

	expected := []float32{5.0, 7.0, 9.0}
	for i, v := range dst {
		if v != expected[i] {
			t.Errorf("dst[%d] = %v, expected %v", i, v, expected[i])
		}
	}
}
