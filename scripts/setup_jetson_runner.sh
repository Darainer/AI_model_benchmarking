#!/usr/bin/env bash
#
# setup_jetson_runner.sh — register an NVIDIA Jetson as a self-hosted
# GitHub Actions runner for this repository, installed as a systemd service
# so it survives reboots.
#
# Run this ONCE per Jetson device. It is idempotent enough to re-run after a
# failed attempt (it refuses to clobber an already-configured runner).
#
# Prereqs on the Jetson (see CI.md for details):
#   - Docker + the nvidia container runtime working (`docker info` shows nvidia)
#   - `make pull && make build` succeed (base image present)
#   - curl, tar (standard on JetPack)
#
# Usage:
#   # 1. Grab a short-lived registration token (expires in ~1h):
#   #    GitHub → repo → Settings → Actions → Runners → "New self-hosted runner"
#   #    OR with a PAT (repo scope):  see CI.md "Token automation"
#   #
#   # 2. On the Jetson:
#   sudo ./scripts/setup_jetson_runner.sh \
#       --repo Darainer/AI_model_benchmarking \
#       --token <REGISTRATION_TOKEN> \
#       [--name jetson-orin-01] \
#       [--labels jetson,orin-nano,jp6] \
#       [--version 2.323.0] \
#       [--workdir /opt/actions-runner]
#
set -euo pipefail

# --- defaults -------------------------------------------------------------
REPO=""
TOKEN=""
NAME="$(hostname)"
EXTRA_LABELS="jetson"
RUNNER_VERSION=""          # empty => resolve the latest release from the API
WORKDIR="/opt/actions-runner"
RUNNER_USER="${SUDO_USER:-$(id -un)}"

# --- arg parsing ----------------------------------------------------------
while [[ $# -gt 0 ]]; do
  case "$1" in
    --repo)     REPO="$2"; shift 2 ;;
    --token)    TOKEN="$2"; shift 2 ;;
    --name)     NAME="$2"; shift 2 ;;
    --labels)   EXTRA_LABELS="$2"; shift 2 ;;
    --version)  RUNNER_VERSION="$2"; shift 2 ;;
    --workdir)  WORKDIR="$2"; shift 2 ;;
    --user)     RUNNER_USER="$2"; shift 2 ;;
    -h|--help)
      sed -n '2,40p' "$0"; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; exit 1 ;;
  esac
done

# --- validation -----------------------------------------------------------
[[ -n "$REPO"  ]] || { echo "ERROR: --repo OWNER/REPO is required" >&2; exit 1; }
[[ -n "$TOKEN" ]] || { echo "ERROR: --token <REGISTRATION_TOKEN> is required" >&2; exit 1; }

ARCH="$(uname -m)"
if [[ "$ARCH" != "aarch64" && "$ARCH" != "arm64" ]]; then
  echo "WARNING: expected aarch64 (Jetson), got '$ARCH'. Continuing anyway." >&2
fi

# The GitHub runner must NOT run as root. If invoked with sudo, drop to the
# invoking user for the configure/run steps; only the service install is root.
if [[ "$RUNNER_USER" == "root" ]]; then
  echo "ERROR: the Actions runner cannot run as root. Re-run via 'sudo' from a" >&2
  echo "       normal user account, or pass --user <name>." >&2
  exit 1
fi

echo ">> Repo:    $REPO"
echo ">> Name:    $NAME"
echo ">> Labels:  self-hosted (implicit), ${EXTRA_LABELS}"
echo ">> User:    $RUNNER_USER"
echo ">> Workdir: $WORKDIR"

# --- resolve latest runner version if not pinned --------------------------
if [[ -z "$RUNNER_VERSION" ]]; then
  echo ">> Resolving latest actions/runner release ..."
  RUNNER_VERSION="$(curl -fsSL https://api.github.com/repos/actions/runner/releases/latest \
    | grep -oP '"tag_name":\s*"v\K[^"]+')"
  [[ -n "$RUNNER_VERSION" ]] || { echo "ERROR: could not resolve latest version" >&2; exit 1; }
fi
echo ">> Runner version: $RUNNER_VERSION"

TARBALL="actions-runner-linux-arm64-${RUNNER_VERSION}.tar.gz"
URL="https://github.com/actions/runner/releases/download/v${RUNNER_VERSION}/${TARBALL}"

# --- install ---------------------------------------------------------------
mkdir -p "$WORKDIR"
chown "$RUNNER_USER":"$RUNNER_USER" "$WORKDIR"

if [[ -f "$WORKDIR/.runner" ]]; then
  echo "ERROR: $WORKDIR already has a configured runner (.runner present)." >&2
  echo "       Remove it first:  cd $WORKDIR && sudo ./svc.sh uninstall && ./config.sh remove --token <TOKEN>" >&2
  exit 1
fi

echo ">> Downloading $TARBALL ..."
sudo -u "$RUNNER_USER" bash -c "cd '$WORKDIR' && curl -fsSL -o '$TARBALL' '$URL' && tar xzf '$TARBALL' && rm -f '$TARBALL'"

# --- configure (as the runner user, unattended) ---------------------------
echo ">> Configuring runner ..."
sudo -u "$RUNNER_USER" bash -c "cd '$WORKDIR' && ./config.sh \
  --unattended \
  --url 'https://github.com/${REPO}' \
  --token '${TOKEN}' \
  --name '${NAME}' \
  --labels '${EXTRA_LABELS}' \
  --work '_work' \
  --replace"

# --- install + start as a systemd service (root) --------------------------
echo ">> Installing systemd service (auto-start on boot) ..."
( cd "$WORKDIR" && ./svc.sh install "$RUNNER_USER" && ./svc.sh start )

echo
echo "✓ Runner '${NAME}' registered and running as a service."
echo "  Verify:  GitHub → repo → Settings → Actions → Runners (should show '${NAME}' Idle)"
echo "  Status:  cd $WORKDIR && sudo ./svc.sh status"
echo "  Logs:    journalctl -u 'actions.runner.*' -f"
