# TCQ4 Channel Reordering Calibration Tool

This tool computes channel permutations for TCQ4 quantization based on activation
statistics. The permutations group outlier channels together to improve Runtime
Smooth (RRS) effectiveness.

## Background

The RRS paper (Section 3.2) describes offline channel reordering:
1. Run calibration samples through the model
2. Collect max absolute activation value per channel
3. Sort channels by magnitude (outliers first)
4. Reorder weights to match permutation
5. At runtime, permute activations before FWHT

This reduces "victims" - normal values that get incorrectly scaled because they
share a Runtime Smooth group with outliers.

## Requirements

```bash
pip install llama-cpp-python numpy
```

## Usage

### Quick Start (Model Structure Based)

```bash
# Generate permutations based on model structure
python calibrate.py --model model-f16.gguf --output perms.json
```

This uses heuristics based on model architecture. Fast but less accurate.

### With Calibration Data

```bash
# Use WikiText-2 style calibration
python calibrate.py --model model-f16.gguf --output perms.json --calibration wikitext

# Use custom calibration text
python calibrate.py --model model-f16.gguf --output perms.json --calibration mydata.txt
```

### Options

- `--model, -m`: Path to input GGUF model (F16 recommended)
- `--output, -o`: Path to output JSON file
- `--calibration, -c`: Calibration source ('wikitext' or file path)
- `--samples, -n`: Number of calibration samples (default: 128)
- `--verbose, -v`: Enable verbose output

## Output Format

The tool outputs a JSON file with the following structure:

```json
{
  "version": 1,
  "description": "TCQ4 channel permutations for Runtime Smooth optimization",
  "reorder_enabled": true,
  "permutations": {
    "blk.0.attn_q.weight": [3, 1, 0, 2, ...],
    "blk.0.attn_k.weight": [...],
    ...
  }
}
```

Each permutation array contains indices `[0, K)` sorted by channel importance
(outliers first).

## Using with Quantization

After generating permutations:

```bash
# Quantize with channel reordering
llama-quantize --tcq4-perms perms.json model-f16.gguf model-tcq4.gguf TCQ4_K32
```

The quantizer will:
1. Load permutations from JSON
2. Reorder weight columns for each tensor
3. Store permutations in GGUF metadata
4. Quantize reordered weights to TCQ4_K32

At runtime, the model loader will:
1. Read permutations from GGUF
2. Register with GPU backend
3. Apply permutation to activations before FWHT

## Limitations

Current implementation uses heuristics rather than actual activation capture.
For best results, a full calibration implementation would require:
1. Hooks into GGML computation graph
2. Capture activations at each linear layer input
3. Aggregate statistics across calibration samples

This is planned for future versions.

## References

- RRS Paper: "Rotated Runtime Smooth for Accurate 4-bit LLM Inference"
- Section 3.2: "Offline Channel Reordering"