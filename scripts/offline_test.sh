#!/usr/bin/env bash
# Offline verification: prove the pipeline runs with no network.
#
# The weak version of this test is "turn off Wi-Fi and run it" -- weak because a
# long-running container may already hold the model in memory, which proves
# nothing about what happens after a restart. This restarts the container while
# the network is down, forcing every artifact to reload from the on-disk cache.
# That is what a site reboot at 3am with no WAN actually looks like.
#
# Run it with Wi-Fi already OFF. Everything is captured to the log so the run is
# evidence rather than a claim.
#
# Usage:
#     bash scripts/offline_test.sh data/bag.MOV

set -u
VIDEO="${1:-data/bag.MOV}"
LOG="artifacts/offline-run.log"
CONTAINER="$(docker ps --format '{{.Names}}' | head -1)"

mkdir -p artifacts
exec > >(tee "$LOG") 2>&1

echo "=============================================================="
echo "OFFLINE VERIFICATION"
echo "date          $(date)"
echo "host          $(uname -srm)"
echo "container     ${CONTAINER}"
echo "=============================================================="

echo
echo "--- 1. network state -----------------------------------------"
networksetup -getairportpower en0
echo "default route:"
route -n get default 2>&1 | grep -E 'gateway|interface' || echo "  no default route"

echo
echo "--- 2. prove the internet is unreachable ----------------------"
# Both of these MUST fail. If either succeeds, the test is invalid.
if curl -s -m 8 -o /dev/null -w '%{http_code}' https://api.roboflow.com 2>/dev/null; then
    echo "  !! api.roboflow.com REACHABLE -- test invalid, network is up"
else
    echo "  api.roboflow.com unreachable (expected)"
fi
if ping -c 1 -W 3000 1.1.1.1 >/dev/null 2>&1; then
    echo "  !! 1.1.1.1 REACHABLE -- test invalid, network is up"
else
    echo "  1.1.1.1 unreachable (expected)"
fi

echo
echo "--- 3. on-disk cache before restart ---------------------------"
du -sh /private/tmp/model-cache
ls /private/tmp/model-cache/yolo_world/
echo "workflow definition cached:"
find /private/tmp/model-cache/workflow -name '*.json' | head -3

echo
echo "--- 4. restart the container (clears anything held in memory) --"
docker restart "${CONTAINER}"
printf "  waiting for :9001 "
for _ in $(seq 1 60); do
    if curl -s -m 2 http://localhost:9001/info >/dev/null 2>&1; then
        echo " up"
        break
    fi
    printf "."
    sleep 2
done
curl -s http://localhost:9001/info
echo

echo
echo "--- 5. run the pipeline, fully offline ------------------------"
./.venv/bin/python -m audit.pipeline "${VIDEO}" --out out/audit-offline.mp4 2>/dev/null

echo
echo "=============================================================="
echo "END OFFLINE VERIFICATION  $(date)"
echo "=============================================================="
