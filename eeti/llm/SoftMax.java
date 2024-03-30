package eeti.llm;

public class SoftMax {

	 public void excecute(float[] x, int xOffset, int size) {
	        // find max value (for numerical stability)
	        float max_val = x[0 + xOffset];
	        for (int i = 1; i < size; i++) {
	            if (x[i + xOffset] > max_val) {
	                max_val = x[i + xOffset];
	            }
	        }
	        // exp and sum
	        float sum = 0.0f;
	        for (int i = 0; i < size; i++) {
	            x[i + xOffset] = (float) Math.exp(x[i + xOffset] - max_val);
	            sum += x[i + xOffset];
	        }
	        // normalize
	        for (int i = 0; i < size; i++) {
	            x[i + xOffset] /= sum;
	        }
	    }
}
