"""Compare a captured query projection with independent CPU arithmetic."""
from pathlib import Path
import argparse
import json
import sys

import numpy as np

repository = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(repository / 'third_party/llama.cpp/gguf-py'))
from gguf import GGMLQuantizationType
from gguf.quants import dequantize


def errors(actual, expected):
    difference = actual.astype(np.float64) - expected.astype(np.float64)
    return {
        'max_abs': float(np.abs(difference).max()),
        'rms': float(np.sqrt(np.mean(difference * difference))),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pack', type=Path, required=True)
    parser.add_argument('--vectors', type=Path, required=True)
    args = parser.parse_args()
    with np.load(args.vectors, allow_pickle=False) as vectors:
        inputs = vectors['input']
        window = vectors['window']
        chunk = vectors['chunk']
    assert inputs.dtype == window.dtype == chunk.dtype == np.float32
    assert window.shape == chunk.shape

    metadata = dict(line.split(' ', 1) for line in (args.pack / 'ds4.txt').read_text().splitlines())
    files = {}
    entry = None
    tensor = 'layers.0.attn.wq_b.weight'
    for line in (args.pack / 'tensors.txt').read_text().splitlines():
        fields = line.split()
        if fields[0] == 'file':
            files[int(fields[1])] = fields[2]
        elif fields[:2] == ['t', tensor]:
            entry = fields
    assert entry is not None and entry[2] == 'Q6_K'
    file_id, offset, byte_count, rows, columns = map(int, entry[3:])
    assert columns == inputs.size and rows == window.size * 2
    row_bytes = columns // 256 * 210
    assert columns % 256 == 0 and byte_count == rows * row_bytes
    raw = np.memmap(
        Path(metadata['model_dir']) / files[file_id],
        dtype=np.uint8, mode='r', offset=offset, shape=(window.size, row_bytes),
    )
    weights = dequantize(raw, GGMLQuantizationType.Q6_K).astype(np.float32)
    assert weights.shape == (window.size, inputs.size)
    reference = weights.astype(np.float64) @ inputs.astype(np.float64)

    blocks = inputs.reshape(-1, 32)
    scales = np.max(np.abs(blocks), axis=1, keepdims=True) / np.float32(127)
    scaled = np.divide(blocks, scales, out=np.zeros_like(blocks), where=scales != 0)
    q8_inputs = (np.rint(scaled) * scales).reshape(-1)
    q8_reference = weights.astype(np.float64) @ q8_inputs.astype(np.float64)
    fp16_reference = (
        weights.astype(np.float16).astype(np.float64)
        @ inputs.astype(np.float16).astype(np.float64)
    )
    result = {
        'tensor': tensor, 'dtype': 'Q6_K', 'rank': 0,
        'shape': list(weights.shape), 'input_position': 16,
        'window_vs_full_fp64': errors(window, reference),
        'chunk_vs_full_fp64': errors(chunk, reference),
        'window_vs_q8_cpu': errors(window, q8_reference),
        'chunk_vs_fp16_cpu': errors(chunk, fp16_reference),
        'window_vs_chunk': errors(window, chunk),
    }
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
