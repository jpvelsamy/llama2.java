package eeti.llm;

import java.util.Arrays;
import java.util.stream.IntStream;

public class Feedforward {

	private SoftMax softMax;
	private RmsNorm rmsNorm;
	private KarpathyTransformer transformer;
	private MatMul matMul;
	private LlamaConfig llamaConfig;
	private LlamaWeights llamaWeights;
	private LlamaRunState runState;
	
	public Feedforward(RmsNorm rmsNorm, KarpathyTransformer transformer, MatMul matMul, SoftMax softMax)
	{
		this.rmsNorm = rmsNorm;
		this.transformer = transformer;
		this.matMul = matMul;
		this.llamaConfig = transformer.getConfig();
		this.llamaWeights = transformer.weights;
		this.runState = transformer.state;
		this.softMax = softMax;
	}
	
	 float[] forward(KarpathyTransformer transformer, int token, int pos) {
        
        
        
        int dim = llamaConfig.dim;
        int hidden_dim = llamaConfig.hidden_dim;
        int head_size = llamaConfig.head_size;
        int kv_dim = (llamaConfig.dim * llamaConfig.n_kv_heads) / llamaConfig.n_heads;
        int kv_mul = llamaConfig.n_heads / llamaConfig.n_kv_heads; // integer multiplier of the kv sharing in multiquery

        // copy the token embedding into x
        llamaWeights.token_embedding_table.get(token * dim, runState.x, 0, dim);

        // forward all the layers
        for (int l = 0; l < llamaConfig.n_layers; l++) {

            // attention rmsnorm
        	
        	rmsNorm.normalize(runState.xb, runState.x, llamaWeights.rms_att_weight[l], dim);

            // qkv matmuls for this position
            matMul.execute(runState.q, runState.xb, llamaWeights.wq[l], dim, dim);
            matMul.execute(runState.k, runState.xb, llamaWeights.wk[l], dim, kv_dim);
            matMul.execute(runState.v, runState.xb, llamaWeights.wv[l], dim, kv_dim);

            // RoPE relative positional encoding: complex-valued rotate q and k in each head
            for (int i = 0; i < dim; i+=2) {
                int head_dim = i % head_size;
                float freq = (float) (1.0 / Math.pow(10000.0f, head_dim / (float) head_size));
                float val = pos * freq;
                float fcr = (float) Math.cos(val);
                float fci = (float) Math.sin(val);
                int rotn = i < kv_dim ? 2 : 1; // how many vectors? 2 = q & k, 1 = q only
                for (int v = 0; v < rotn; v++) {
                    float[] vec = v == 0 ? runState.q : runState.k; // the vector to rotate (query or key)
                    float v0 = vec[i];
                    float v1 = vec[i + 1];
                    vec[i] = v0 * fcr - v1 * fci;
                    vec[i + 1] = v0 * fci + v1 * fcr;
                }
            }

            // save key,value at this time step (pos) to our kv cache
            //int loff = l * p.seq_len * kv_dim; // kv cache layer offset for convenience
            System.arraycopy(runState.k, 0, runState.key_cache[l], pos * kv_dim, kv_dim);
            System.arraycopy(runState.v, 0, runState.value_cache[l], pos * kv_dim, kv_dim);


            final int curLayer = l;

            // multihead attention. iterate over all heads
            IntStream.range(0, llamaConfig.n_heads).parallel().forEach(h -> {
                // get the query vector for this head
                // float* q = s.q + h * head_size;
                int qOffset = h * head_size;

                // attention scores for this head
                // float* att = s.att + h * p.seq_len;
                int attOffset = h * llamaConfig.getSeq_len();

                // iterate over all timesteps, including the current one
                for (int t = 0; t <= pos; t++) {
                    // get the key vector for this head and at this timestep
                    // float* k = s->key_cache + loff + t * kv_dim + (h / kv_mul) * head_size;
                    int keyCacheOffset = t * kv_dim + (h / kv_mul) * head_size;
                    // calculate the attention score as the dot product of q and k
                    float score = 0.0f;
                    for (int i = 0; i < head_size; i++) {
                        score += runState.q[qOffset + i] * runState.key_cache[curLayer][keyCacheOffset + i];
                    }
                    score /= (float) Math.sqrt(head_size);
                    // save the score to the attention buffer
                    runState.att[attOffset + t] = score;
                }

                // softmax the scores to get attention weights, from 0..pos inclusively
                this.softMax.excecute(runState.att, attOffset, pos + 1);

                // weighted sum of the values, store back into xb
                // float* xb = s.xb + h * head_size;
                int xbOffset = h * head_size;
                // memset(xb, 0, head_size * sizeof(float));
                Arrays.fill(runState.xb, xbOffset, xbOffset + head_size, 0f);

                for (int t = 0; t <= pos; t++) {
                    // get the value vector for this head and at this timestep
                    // float* v = s->value_cache + loff + t * kv_dim + (h / kv_mul) * head_size;
                    int vOffset = t * kv_dim + (h / kv_mul) * head_size;
                    // get the attention weight for this timestep
                    float a = runState.att[attOffset + t];
                    // accumulate the weighted value inconfigto xb
                    for (int i = 0; i < head_size; i++) {
                        runState.xb[xbOffset + i] += a * runState.value_cache[curLayer][vOffset + i];
                    }
                }
            });

            // final matmul to get the output of the attention
            matMul.execute(runState.xb2, runState.xb, llamaWeights.wo[l], dim, dim);

            // residual connection back into x
            for (int i = 0; i < dim; i++) {
                runState.x[i] += runState.xb2[i];
            }

            // ffn rmsnorm
            rmsNorm.normalize(runState.xb, runState.x, llamaWeights.rms_ffn_weight[l], dim);

            // Now for FFN in PyTorch we have: self.w2(F.silu(self.w1(x)) * self.w3(x))
            // first calculate self.w1(x) and self.w3(x)
            matMul.execute(runState.hb, runState.xb, llamaWeights.w1[l], dim, llamaConfig.hidden_dim);
            matMul.execute(runState.hb2, runState.xb, llamaWeights.w3[l], dim, llamaConfig.hidden_dim);

            // SwiGLU non-linearity
            for (int i = 0; i < hidden_dim; i++) {
                float val = runState.hb[i];
                // silu(x)=x*σ(x), where σ(x) is the logistic sigmoid
                val *= (1.0f / (1.0f + Math.exp(-val)));
                // elementwise multiply with w3(x)
                runState.hb[i] = val;
            }

            // elementwise multiply with w3(x)
            for (int i = 0; i < hidden_dim; i++) {
                runState.hb[i] = runState.hb[i] * runState.hb2[i];
            }

            // final matmul to get the output of the ffn
            matMul.execute(runState.xb, runState.hb, llamaWeights.w2[l], llamaConfig.hidden_dim, dim);

            // residual connection
            for (int i = 0; i < dim; i++) {
                runState.x[i] += runState.xb[i];
            }
        }

        // final rmsnorm
        rmsNorm.normalize(runState.x, runState.x, llamaWeights.rms_final_weight, dim);

        // classifier into logits
        matMul.execute(runState.logits, runState.x, llamaWeights.wcls, dim, llamaConfig.vocab_size);
        return runState.logits;
    }
}
