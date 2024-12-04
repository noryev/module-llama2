# Use PyTorch as base image
FROM pytorch/pytorch:latest

# Set the working directory in the container
WORKDIR /workspace

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install transformers \
    accelerate \
    bitsandbytes \
    sentencepiece

# Create necessary directories
RUN mkdir -p /root/.cache/huggingface /outputs \
    && chmod 777 /outputs

# Create model download script
RUN echo 'import os\n\
from transformers import AutoTokenizer, AutoModelForCausalLM\n\
import torch\n\
\n\
print("Starting model download...")\n\
model_id = "facebook/opt-350m"\n\
save_directory = "/workspace/model_cache"\n\
\n\
# Download and save model\n\
print("Downloading model...")\n\
model = AutoModelForCausalLM.from_pretrained(\n\
    model_id,\n\
    device_map="auto",\n\
    torch_dtype=torch.float16,\n\
    low_cpu_mem_usage=True\n\
)\n\
model.save_pretrained(save_directory)\n\
\n\
# Download and save tokenizer\n\
print("Downloading tokenizer...")\n\
tokenizer = AutoTokenizer.from_pretrained(model_id)\n\
tokenizer.save_pretrained(save_directory)\n\
print("Model and tokenizer downloaded successfully!")' > download_model.py

# Download the model during build
RUN python download_model.py

# Copy the inference script
COPY run_llama.py /workspace/run_llama.py
RUN chmod +x /workspace/run_llama.py

# Set default environment variables
ENV DEFAULT_PROMPT="Tell me a story."

# Set the entrypoint
ENTRYPOINT ["python", "/workspace/run_llama.py"]

# Set a default command that can be overridden
CMD ["${DEFAULT_PROMPT}"]