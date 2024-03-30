package eeti.llm;
import java.io.IOException;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.channels.FileChannel;
import java.nio.file.Paths;
import java.nio.file.StandardOpenOption;

public final class KarpathyTransformer {
    private final LlamaConfig config; // the hyperparameters of the architecture (the blueprint)
    final LlamaWeights weights; // the weights of the model
    final LlamaRunState state; // buffers for the "wave" of activations in the forward pass
    // some more state needed to properly clean up the memory mapping (sigh)
    final Arena memoryArena; // scope of the memory mapping
    final MemorySegment data; // memory mapped data pointer
    final long file_size; // size of the checkpoint file in bytes

    public KarpathyTransformer(String checkpoint_path) throws IOException {
        try (FileChannel fileChannel = FileChannel.open(Paths.get(checkpoint_path), StandardOpenOption.READ)) {
            this.file_size = fileChannel.size();
            this.memoryArena = Arena.ofAuto();
            MemorySegment mappedFile = fileChannel.map(FileChannel.MapMode.READ_ONLY, 0, this.file_size, this.memoryArena);
            this.data = mappedFile;
            int configSize = 7 * Integer.BYTES;
            // read in the config header
            ByteBuffer configBuffer = mappedFile.asSlice(0, configSize).asByteBuffer().order(ByteOrder.LITTLE_ENDIAN);
            this.config = new LlamaConfig(configBuffer);
            System.out.println(config);
            this.state = new LlamaRunState(config);
            this.weights = new LlamaWeights(config, mappedFile.asSlice(configSize));
        }
    }

	public LlamaConfig getConfig() {
		return config;
	}

	public LlamaWeights getWeights() {
		return weights;
	}

	public LlamaRunState getState() {
		return state;
	}

	public Arena getMemoryArena() {
		return memoryArena;
	}

	public MemorySegment getData() {
		return data;
	}

	public long getFile_size() {
		return file_size;
	}
    
    
}