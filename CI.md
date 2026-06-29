# CI on Jetson — self-hosted GitHub Actions runners

This repo benchmarks models on real Jetson GPUs, so CI has to run **on the
Jetson hardware itself**. GitHub's cloud runners are x86 and have no GPU, so we
register each Jetson as a **self-hosted runner**. Once registered, a Jetson
shows up as a CI machine in your GitHub account and picks up jobs labelled
`self-hosted` + `jetson`.

You have several devices — register each one the same way; they form a pool that
shares the workload (or you can pin specific jobs to specific devices with
labels, see [Targeting specific devices](#targeting-specific-devices)).

---

## How it fits together

```
GitHub repo (Darainer/AI_model_benchmarking)
        │  push / PR / nightly cron / manual dispatch
        ▼
.github/workflows/benchmark.yml   runs-on: [self-hosted, jetson]
        │
        ▼
   any idle Jetson in the pool
   (runner service installed by scripts/setup_jetson_runner.sh)
        │  make pull → build → gpu-check → bench
        ▼
   results/*.csv  ──►  uploaded as a workflow artifact
```

Self-hosted runners can be registered at **repo**, **org**, or **enterprise**
scope. For a single repo, register at repo scope (simplest). If you want the
same Jetson pool shared across several repos, register at **org** scope instead
(same steps, but use the org's Settings → Actions → Runners page and the
`https://github.com/<org>` URL).

---

## Per-device setup

Do this **once per Jetson**. Budget ~15 min for the first device (most of it is
the one-time base-image pull).

### 1. Prerequisites on the Jetson

The CI jobs reuse the existing Docker stack, so each device needs the same
working setup the benchmark already requires:

```bash
# Docker + nvidia container runtime must be working:
docker info | grep -i runtime          # should list "nvidia"

# The benchmark image must build and the GPU must be visible:
cd ~/AI_model_benchmarking
make pull        # one-time ~10-15 GB base image
make build
make gpu-check   # should print CUDA/TensorRT execution providers
```

If `make gpu-check` shows the GPU providers, the device is ready to be a runner.

> The runner process must **not** run as root, but it does need to run `docker`.
> Add the runner's user to the `docker` group once:
> ```bash
> sudo usermod -aG docker $USER && newgrp docker
> ```

### 2. Get a registration token

In GitHub: **repo → Settings → Actions → Runners → New self-hosted runner**.
GitHub shows a `--token ghr_…` value there. It is **short-lived (~1 hour)** and
single-use per registration — grab a fresh one for each device, or automate it
(see [Token automation](#token-automation)).

### 3. Run the setup script

From a checkout of this repo on the Jetson:

```bash
sudo ./scripts/setup_jetson_runner.sh \
  --repo  Darainer/AI_model_benchmarking \
  --token ghr_xxxxxxxxxxxxxxxxxxxxx \
  --name  jetson-orin-01 \
  --labels jetson,orin-nano,jp6
```

What the script does:

- downloads the latest `actions/runner` build for `linux-arm64` (or pin with
  `--version 2.323.0`),
- configures it against your repo with the label `jetson` (plus any extras you
  pass) — `self-hosted` and `ARM64` are added by GitHub automatically,
- installs it as a **systemd service** so it auto-starts on boot and reconnects
  after a reboot or power cycle,
- starts it.

Give each device a unique `--name` (e.g. `jetson-orin-01`, `-02`, …) so you can
tell them apart in the Runners list and in artifact names.

### 4. Verify

GitHub → repo → Settings → Actions → Runners — the device should appear as
**Idle**. On the device:

```bash
cd /opt/actions-runner && sudo ./svc.sh status
journalctl -u 'actions.runner.*' -f      # live runner logs
```

Repeat steps 2–4 for each Jetson.

---

## The workflow

[`.github/workflows/benchmark.yml`](.github/workflows/benchmark.yml) runs on
push to `main`, on PRs, nightly (03:00 UTC), and on manual dispatch. It:

1. checks out the repo,
2. ensures the base image is present (`make pull` — fast no-op after first run),
3. builds the image (`make build`),
4. runs the GPU sanity check (`make gpu-check`),
5. runs the **synthetic** benchmark (`make bench`),
6. uploads `results/*.csv` as an artifact.

CI deliberately runs **synthetic frames only** — no hardware video decode.
HW decode currently hangs inside the container (see
[`open_issues.md`](open_issues.md)), and an unbounded `nvv4l2decoder` failure
has wedged a device hard enough to need a restart. Keep CI on synthetic until
that's resolved; run HW-decode tests manually and bounded.

`concurrency` ensures a newer push cancels an in-flight run for the same ref so
jobs don't pile onto a busy device.

---

## Targeting specific devices

`runs-on: [self-hosted, jetson]` runs on **any** idle Jetson in the pool — good
for load-sharing. To pin work to particular hardware, give devices distinct
labels at setup time (e.g. `--labels jetson,orin-nano` vs `--labels jetson,agx-orin`)
and target them:

```yaml
runs-on: [self-hosted, jetson, agx-orin]
```

To benchmark on **every** device on each run, use a matrix over the per-device
labels:

```yaml
jobs:
  benchmark:
    strategy:
      matrix:
        device: [orin-nano, agx-orin, xavier-nx]
    runs-on: [self-hosted, jetson, "${{ matrix.device }}"]
```

---

## Token automation

Registration tokens expire hourly, which is tedious across many devices. With a
**PAT** (fine-grained: *Administration → read/write*, or classic with `repo`
scope) you can mint one on the fly:

```bash
REG_TOKEN=$(curl -fsSL -X POST \
  -H "Authorization: Bearer $GH_PAT" \
  -H "Accept: application/vnd.github+json" \
  https://api.github.com/repos/Darainer/AI_model_benchmarking/actions/runners/registration-token \
  | grep -oP '"token":\s*"\K[^"]+')

sudo ./scripts/setup_jetson_runner.sh \
  --repo Darainer/AI_model_benchmarking --token "$REG_TOKEN" --name "$(hostname)"
```

For org-scope runners use `.../orgs/<org>/actions/runners/registration-token`.

---

## Maintenance

```bash
cd /opt/actions-runner

sudo ./svc.sh status          # is it running?
sudo ./svc.sh stop|start      # control the service
journalctl -u 'actions.runner.*' -f   # tail logs

# Upgrade the runner: GitHub auto-updates self-hosted runners by default.

# Remove a runner cleanly (needs a fresh removal token from the Runners page):
sudo ./svc.sh uninstall
./config.sh remove --token <REMOVAL_TOKEN>
```

### Security notes

- Self-hosted runners on a **public** repo are risky: a malicious PR can run
  arbitrary code on your hardware. This repo is private-scope; if it ever goes
  public, disable "Run workflows from fork pull requests" or require approval
  for outside collaborators (Settings → Actions → General).
- The runner persists its working directory between jobs. `make` reuses the
  cached base image and `models/`/`results/` are gitignored, so state carries
  over by design — keep that in mind when debugging flaky runs.
