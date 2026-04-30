#!/bin/bash
# Entrypoint script for vLLM XPU container
# Sources Intel oneAPI environment and executes vLLM

set -e

# Source Intel oneAPI environment
source /opt/intel/oneapi/setvars.sh --force
source /opt/intel/oneapi/ccl/2021.15/env/vars.sh --force

# Execute the command passed to the container
exec "$@"
