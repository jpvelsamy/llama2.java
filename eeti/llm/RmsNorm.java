package eeti.llm;

import java.nio.FloatBuffer;

public class RmsNorm {

	
	
	public void normalize(float[] o, float[] x, FloatBuffer weight, int size) {
        // calculate sum of squares
        float sumSquares = 0.0f;
        for (int j = 0; j < size; j++) {
            sumSquares += x[j] * x[j];
        }
        sumSquares /= size;
        sumSquares += 1e-5f;
        sumSquares = 1.0f / (float) Math.sqrt(sumSquares);
        // normalize and scale
        for (int j = 0; j < size; j++) {
            o[j] = weight.get(j) * (sumSquares * x[j]);
        }
    }
}
