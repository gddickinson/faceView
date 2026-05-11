#!/usr/bin/env bash
# Drive the GUI through a sequence of body effects and capture screenshots.
# Usage: bash tools/_gui_tour.sh <gender> <outdir>
set -euo pipefail
GENDER="${1:-male}"
OUTDIR="${2:-/tmp/gui_tour_${GENDER}}"
PORT="${PORT:-8765}"
BASE="http://127.0.0.1:${PORT}"
mkdir -p "${OUTDIR}"

# Body effects to test (body-only — skip face/eye/head effects)
EFFECTS=(
  body_bow body_lean_back body_lean_left body_lean_right
  body_twist_left body_twist_right body_sway
  wave_left wave_right
  arms_up arms_out arms_crossed
  hands_on_hips shrug
  point_left point_right thinking
  clap stretch_up salute curtsy
  kick_left kick_right squat
  lunge_left lunge_right jump
)

PERSONA="ict_${GENDER}"
MORPH=$([ "${GENDER}" = "male" ] && echo "1.0" || echo "-1.0")

echo "Setting persona=${PERSONA} morph=${MORPH}"
curl -s -m 3 -X POST "${BASE}/avatar/persona" -H 'Content-Type: application/json' \
    -d "{\"name\":\"${PERSONA}\"}" >/dev/null
curl -s -m 3 -X POST "${BASE}/effects/slider" -H 'Content-Type: application/json' \
    -d '{"key":"show_body","value":1.0}' >/dev/null
curl -s -m 3 -X POST "${BASE}/effects/slider" -H 'Content-Type: application/json' \
    -d "{\"key\":\"body_morph\",\"value\":${MORPH}}" >/dev/null
curl -s -m 3 -X POST "${BASE}/effects/slider" -H 'Content-Type: application/json' \
    -d '{"key":"camera_zoom","value":0.7}' >/dev/null
curl -s -m 3 -X POST "${BASE}/effects/slider" -H 'Content-Type: application/json' \
    -d '{"key":"camera_focus_y","value":0.0}' >/dev/null
sleep 1

curl -s -m 3 -X POST "${BASE}/effects/stop_all" >/dev/null
sleep 0.5
curl -s -m 5 -X POST "${BASE}/screenshot" -H 'Content-Type: application/json' -d '{}' >/dev/null
cp docs/images/shot.png "${OUTDIR}/00_neutral.png"
echo "captured neutral"

for E in "${EFFECTS[@]}"; do
  curl -s -m 3 -X POST "${BASE}/effects/stop_all" >/dev/null
  sleep 0.3
  curl -s -m 3 -X POST "${BASE}/effects/trigger" -H 'Content-Type: application/json' \
      -d "{\"name\":\"${E}\",\"duration\":6.0,\"intensity\":1.0}" >/dev/null
  sleep 2.5
  curl -s -m 5 -X POST "${BASE}/screenshot" -H 'Content-Type: application/json' -d '{}' >/dev/null
  cp docs/images/shot.png "${OUTDIR}/${E}.png"
  echo "captured ${E}"
done

curl -s -m 3 -X POST "${BASE}/effects/stop_all" >/dev/null
echo "tour complete: ${OUTDIR}"
