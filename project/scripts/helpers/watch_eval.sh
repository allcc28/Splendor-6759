#!/bin/bash
# watch_eval.sh — Live progress monitor for robust eval log
# Usage: bash project/scripts/helpers/watch_eval.sh [logfile]
LOG="${1:-project/logs/robust_eval_v5_event.log}"
INTERVAL=5
N=10

strip(){ sed 's/\x1b\[[0-9;]*[mGKHF]//g; s/\r//g'; }

bar(){
  local d=$1 w=20 b="" f=$(( $1 * 20 / $2 )) e=$(( 20 - $1 * 20 / $2 ))
  for i in $(seq 1 $f 2>/dev/null); do b="${b}█"; done
  for i in $(seq 1 $e 2>/dev/null); do b="${b}░"; done
  echo "[$b]"
}

cnt_batches(){
  python3 - "$1" "$LOG" <<'PYEOF'
import re, sys
opp, log = sys.argv[1], sys.argv[2]
try:
    t = open(log).read()
except: print(0); sys.exit()
t = re.sub(r'\x1b\[[0-9;]+m', '', t)
m = re.search(re.escape('Evaluating vs ' + opp), t)
if not m: print(0); sys.exit()
s = t[m.end():]
nx = re.search(r'Evaluating vs (?!' + re.escape(opp) + r')', s)
if nx: s = s[:nx.start()]
print(len(re.findall(r'^\s+Batch \d+:', s, re.MULTILINE)))
PYEOF
}

batch_lines(){
  python3 - "$1" "$LOG" <<'PYEOF'
import re, sys
opp, log = sys.argv[1], sys.argv[2]
try:
    t = open(log).read()
except: sys.exit()
t = re.sub(r'\x1b\[[0-9;]+m', '', t)
m = re.search(re.escape('Evaluating vs ' + opp), t)
if not m: sys.exit()
s = t[m.end():]
nx = re.search(r'Evaluating vs (?!' + re.escape(opp) + r')', s)
if nx: s = s[:nx.start()]
for l in re.findall(r'  Batch \d+:.*', s):
    print(l)
PYEOF
}

while true; do
  clear
  echo "══════════════════════════════════════════"
  printf "  Monitor  %s\n" "$(date '+%H:%M:%S')"
  printf "  %s\n" "$LOG"
  echo "══════════════════════════════════════════"
  echo ""

  if [ ! -f "$LOG" ]; then
    echo "  [waiting for log file...]"
    sleep $INTERVAL; continue
  fi

  for opp in "Random (wrapper)" "RandomAgent" "GreedyAgent"; do
    n=$(cnt_batches "$opp")
    b=$(bar $n $N)
    if [ "$n" -ge "$N" ]; then sym="✅"
    elif [ "$n" -gt 0 ]; then sym="🔄"
    else sym="⏳"; fi
    printf "  %s vs %-22s %s  %d/%d\n" "$sym" "$opp" "$b" "$n" "$N"
    batch_lines "$opp" | tail -3 | sed 's/^/       /'
    echo ""
  done

  inf=$(cat "$LOG" 2>/dev/null | strip | grep -oE '(Random \(wrapper\)|RandomAgent|GreedyAgent) batch [0-9]+/[0-9]+' | tail -1)
  [ -n "$inf" ] && printf "  ▶  %s  (in-flight)\n\n" "$inf"

  if grep -q "Saved" "$LOG" 2>/dev/null; then
    echo "  ✅ EVALUATION COMPLETE"
    cat "$LOG" | strip | grep -E "Pooled win|Wilson|Saved" | head -10 | sed 's/^/    /'
  fi

  printf "\n  [every %ds — Ctrl+C to exit]\n" $INTERVAL
  sleep $INTERVAL
done
