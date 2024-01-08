#!/bin/sh

set -ex

export QEMU_BIN=/usr/src/qemu-riscv/qemu-riscv64

MLIR_PATH=$1
INPUT=$2
MODULE_PATH=${MLIR_PATH%.*}.vmfb

iree-compile --iree-hal-target-backends=vmvx $MLIR_PATH -o $MODULE_PATH

time ${QEMU_BIN} \
-cpu rv64 \
-L ${RISCV_TOOLCHAIN_ROOT}/sysroot/ \
../iree-build-riscv/tools/iree-run-module \
--device=local-task \
--module=$MODULE_PATH \
--function=forward \
--input=@$INPUT \
--output=-
