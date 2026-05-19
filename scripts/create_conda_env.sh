#!/usr/bin/env bash
set -euo pipefail

ENV_FILE="${1:-environment.yml}"
SOLVER="${CONDA_SOLVER:-auto}"

if [[ ! -f "${ENV_FILE}" ]]; then
  echo "Environment file not found: ${ENV_FILE}" >&2
  exit 1
fi

if [[ "${SOLVER}" == "auto" ]]; then
  if command -v mamba >/dev/null 2>&1; then
    SOLVER="mamba"
  else
    SOLVER="conda"
  fi
fi

if ! command -v "${SOLVER}" >/dev/null 2>&1; then
  echo "Conda-compatible solver not found: ${SOLVER}" >&2
  exit 1
fi

ENV_NAME="$(awk '/^name:/ {print $2; exit}' "${ENV_FILE}")"
if [[ -z "${ENV_NAME}" ]]; then
  echo "Could not read environment name from ${ENV_FILE}" >&2
  exit 1
fi

if conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
  "${SOLVER}" env update -n "${ENV_NAME}" -f "${ENV_FILE}" --prune
else
  "${SOLVER}" env create -f "${ENV_FILE}"
fi

echo "Activate with: conda activate ${ENV_NAME}"
