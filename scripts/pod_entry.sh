#!/usr/bin/env bash
# Container entrypoint for a headless pod. Fetched by the pod's start command:
#
#   curl -fsSL .../scripts/pod_entry.sh -o /pod_entry.sh && bash /pod_entry.sh
#
# `run_all.sh` assumes it can report failures by pushing. This script exists
# because that assumption breaks in exactly the situation where it matters most -
# the run dying before it ever gets as far as configuring git - and a headless
# pod that cannot say why it failed is a pod that silently bills by the hour.
#
# The first attempt at this did all three of these wrong and cost an hour of A100
# for zero information:
#
#   1. Its start command ended. A container whose main process exits is
#      *restarted* by RunPod, and the restart re-ran a `rm -rf` clone - so every
#      attempt wiped the stage markers that were supposed to make the run
#      resumable, and the pod sat in a ~6.5 minute crash loop. Hence `sleep
#      infinity`: the container must outlive the work, and the watchdog - not the
#      exit code - owns the pod's lifetime.
#   2. Nothing reported. RunPod exposes no log API, ssh needs a key this side
#      does not have, and the disk dies with the pod, so a failure before the
#      first push was completely invisible. Hence: capture everything, publish it
#      to the repository, and do it whatever happened.
#   3. The watchdog was armed *inside* run_all.sh, after preflight. A failure in
#      preflight therefore left an immortal pod. It is armed here first, before
#      anything that can fail.
set -uo pipefail

WORK="${WORK:-/workspace}"
REPO="$WORK/TLabUED"
REMOTE="https://github.com/Leamich/TLabUED"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
BOOT="$WORK/boot_$STAMP.log"
mkdir -p "$WORK"

# --- 1. the watchdog, before anything that can fail ------------------------
if [ "${TEARDOWN:-1}" = "1" ] && [ -n "${RUNPOD_POD_ID:-}" ] && [ -n "${RUNPOD_API_KEY:-}" ]; then
  export RUNPOD_API_KEY RUNPOD_POD_ID
  setsid nohup bash -c '
    sleep '"$(( ${MAX_HOURS:-4} * 3600 ))"'
    curl -s -X DELETE -H "Authorization: Bearer $RUNPOD_API_KEY" \
      "https://rest.runpod.io/v1/pods/$RUNPOD_POD_ID" >/dev/null
  ' >/dev/null 2>&1 &
  echo "watchdog armed: ${MAX_HOURS:-4}h"
fi

# --- 2. publish whatever we know, scrubbed ---------------------------------
# Called on every exit path. Sets up its own git auth because the run may have
# died before `run_all.sh` set up its own.
publish() {
  local tag="$1"
  cd "$REPO" 2>/dev/null || { echo "publish: no repo to push from"; return 0; }
  git config user.name "${GIT_NAME:-Mikhail Leontyev}"
  git config user.email "${GIT_EMAIL:-michlea.tlt@gmail.com}"
  # Credential helper, never the URL: git echoes the remote in push errors, and
  # this repository is public.
  git config --local credential.helper \
    '!f() { echo username=x-access-token; echo "password=$GH_TOKEN"; }; f'
  git config --local remote.origin.url "$REMOTE"

  mkdir -p results/logs
  sed -e "s#${GH_TOKEN:-__no_token__}#***GH_TOKEN***#g" \
      -e "s#${RUNPOD_API_KEY:-__no_key__}#***RUNPOD_API_KEY***#g" \
      -e 's#https://[^@/]*@github#https://***@github#g' \
      "$BOOT" > "results/logs/boot_${STAMP}_${tag}.log" 2>/dev/null

  git add -f results/logs/ >/dev/null 2>&1
  git diff --cached --quiet && { echo "publish: nothing to commit"; return 0; }
  git commit -q -m "Pod boot log ($tag, $STAMP)" \
    -m "Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>" || return 0
  local i
  for i in 1 2 3 4; do
    git pull --rebase -q origin master && git push -q origin HEAD:master && {
      echo "publish: pushed"
      return 0
    }
    sleep $((i * 10))
  done
  echo "publish: PUSH FAILED"
}

# --- 3. the run ------------------------------------------------------------
{
  echo "=== pod_entry $STAMP ==="
  echo "user=$(whoami) pwd=$PWD work=$WORK"
  echo "pod=${RUNPOD_POD_ID:-none} stop_after=${STOP_AFTER:-none} max_hours=${MAX_HOURS:-4}"
  echo "gh_token=${GH_TOKEN:+set} runpod_key=${RUNPOD_API_KEY:+set}"
  echo "--- gpu ---";    nvidia-smi 2>&1 | head -12 || echo "no nvidia-smi"
  echo "--- disk ---";   df -h "$WORK" 2>&1 | tail -2
  echo "--- memory ---"; free -g 2>&1 | head -2
  echo "--- tools ---";  for t in git curl python3 uv; do
                           echo "  $t: $(command -v $t || echo MISSING)"
                         done

  cd "$WORK" || exit 1
  # Clone only when absent. The previous version wiped and re-cloned on every
  # container start, which is what made the crash loop unrecoverable: the stage
  # markers that exist to resume the run were deleted by the thing restarting it.
  if [ ! -d "$REPO/.git" ]; then
    echo "--- cloning ---"
    git clone "$REMOTE" "$REPO" 2>&1 | tail -5
  else
    echo "--- repo present, updating ---"
    git -C "$REPO" fetch -q origin && git -C "$REPO" reset -q --hard origin/master
  fi
  cd "$REPO" || exit 1
  echo "repo at $(git rev-parse --short HEAD)"

  echo "--- run_all ---"
  RUN_ALL_DETACHED=1 bash scripts/run_all.sh
  echo "=== run_all exited $? ==="
} > "$BOOT" 2>&1

tail -40 "$BOOT"
publish "done" >> "$BOOT" 2>&1
tail -5 "$BOOT"

# --- 4. never exit ---------------------------------------------------------
# An exiting container is a restarting container. If the work finished, the
# teardown inside run_all.sh has already deleted the pod and this line is never
# reached; if it did not, the watchdog will. Idling is the safe failure mode -
# it is visible and bounded, whereas a restart loop is neither.
echo "pod_entry: holding the container open; watchdog owns the lifetime"
sleep infinity
