#!/usr/bin/env bash

# Keep Triton and TorchInductor build files off the quota-limited /tmp mount.
TORCH_TMP=/home/baris/torch-tmp
mkdir -p "$TORCH_TMP/inductor" "$TORCH_TMP/triton"

# Do not override CFLAGS here. Python 3.11 already defines _POSIX_C_SOURCE;
# overriding it makes Triton's cuda_utils extension emit a redefinition warning.
nohup env \
	TMPDIR="$TORCH_TMP" \
	TORCHINDUCTOR_CACHE_DIR="$TORCH_TMP/inductor" \
	TRITON_CACHE_DIR="$TORCH_TMP/triton" \
	/home/baris/anaconda3/envs/py311/bin/jupyter notebook &