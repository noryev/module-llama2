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

# Create necessary directories and set permissions
RUN mkdir -p /root/.cache/huggingface /outputs \
    && chmod 777 /outputs

# Copy the Python script into the container at the correct path
COPY run_llama.py /workspace/run_llama.py
RUN chmod +x /workspace/run_llama.py

# Set default environment variables
ENV DEFAULT_PROMPT="Tell me about LLaMA."

# Set the entrypoint to run the Python script with the correct path
ENTRYPOINT ["python", "/workspace/run_llama.py"]

# Set a default command that can be overridden
CMD ["${DEFAULT_PROMPT}"]