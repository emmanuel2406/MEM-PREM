# Profiling the LLaMA Model

## Overview
This README provides instructions on how to profile the LLaMA model by modifying its serving code to log embeddings from each `LlamaDecoderLayer` during the forward pass of the `LlamaModel`. This process involves tracking the layer ID and timestamp for each output and saving these embeddings to local files for further analysis.

## Modifications
1. **Update the Model Code**:
   - Path: `vllm/model_executor/models/llama.py`
   - Modification: The code has been altered to log the output embeddings of each `LlamaDecoderLayer` and to save these embeddings along with their corresponding layer IDs and timestamps into local files.

## Steps for Profiling
1. **Prepare the Environment**:
   - Ensure that your environment is set up with all necessary dependencies installed, including PyTorch and any libraries required by the LLaMA model.

2. **Run the Model**:
   - Execute the model using the command:
     ```
     nohup python test_llama.py &
     ```
   - This command runs the model in a non-blocking mode, allowing it to process in the background and enabling the system to log all outputs even if the session disconnects.

3. **Collect and Analyze Outputs**:
   - After the model has finished running, outputs are logged in the `nohup.out` file.
   - Use the following script to extract the embeddings and their indices based on layer ID and timestamp:
     ```
     python probing_parser.py nohup.out
     ```
   - This script parses the output file and organizes the embeddings for easier analysis.
