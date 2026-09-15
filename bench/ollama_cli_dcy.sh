#!/usr/bin/env bash
# Ollama CLI + DCY: does the model actually consume the injected context?
# For each task: ask `ollama run` WITHOUT context, then WITH dcy context.
# No JSON protocol, no HTTP — raw CLI piping.
#
# Matching is done on the FULL answer; truncation is display-only (an earlier
# version truncated before matching and scored correct answers as misses).
set -uo pipefail

DCY=${DCY:-build/dcy}
DB=${DB:-build/tree-sitter-lib.sqlite}
MODEL=${MODEL:-qwen2.5:0.5b-instruct}
BUDGET=${BUDGET:-512}
TASKS=${TASKS:-bench/tree-sitter-lib-tasks.jsonl}
LIMIT=${LIMIT:-6}

ask() {  # ask <prompt>  -> single-line answer, untruncated
  printf '%s' "$1" | ollama run "$MODEL" 2>/dev/null \
    | tr -d '\r' | tr '\n' ' ' | sed 's/  */ /g; s/^ //; s/ $//'
}

printf '%-16s %-34s %-26s %-30s %s\n' "TASK" "GOLD" "NO_CONTEXT" "WITH_DCY" "FLAGS"
printf '%.0s-' {1..125}; echo

n=0; hit_none=0; hit_dcy=0; consumed=0; gold_in_page=0
while IFS= read -r line; do
  [ -z "$line" ] && continue
  n=$((n+1)); [ "$n" -gt "$LIMIT" ] && { n=$((n-1)); break; }

  id=$(printf '%s' "$line"   | python3 -c 'import sys,json;print(json.load(sys.stdin)["id"])')
  goal=$(printf '%s' "$line" | python3 -c 'import sys,json;print(json.load(sys.stdin)["goal"])')
  gold=$(printf '%s' "$line" | python3 -c 'import sys,json;print(json.load(sys.stdin)["gold"][0].rsplit(":",1)[1])')

  ans_none=$(ask "$(printf 'Question: %s\nAnswer with ONLY the C function name, nothing else.' "$goal")")

  ctx=$("$DCY" query "$DB" "$goal" "$BUDGET" 2>/dev/null)
  ans_dcy=$(ask "$(printf 'Repository context:\n%s\n\nQuestion: %s\nAnswer with ONLY the C function name from the context above, nothing else.' "$ctx" "$goal")")

  flags=""
  # Is the gold symbol even present in the distilled page? (retrieval check)
  if printf '%s' "$ctx" | grep -qw "$gold"; then
    gold_in_page=$((gold_in_page+1)); flags="gold_in_page"
  else flags="gold_MISSING"; fi

  # Did the model's answer come from the page? (consumption check)
  tok=$(printf '%s' "$ans_dcy" | grep -oE '[A-Za-z_][A-Za-z0-9_]{3,}' | head -n 1)
  if [ -n "$tok" ] && printf '%s' "$ctx" | grep -qw "$tok"; then
    consumed=$((consumed+1)); flags="$flags CONSUMED"
  else flags="$flags off-page"; fi

  printf '%s' "$ans_none" | grep -qw "$gold" && hit_none=$((hit_none+1))
  printf '%s' "$ans_dcy"  | grep -qw "$gold" && { hit_dcy=$((hit_dcy+1)); flags="$flags CORRECT"; }

  printf '%-16s %-34s %-26.26s %-30.30s %s\n' "$id" "$gold" "$ans_none" "$ans_dcy" "$flags"
done < "$TASKS"

echo
echo "model=$MODEL  tasks=$n  budget=${BUDGET}B  db=$DB"
echo "gold symbol present in DCY page : $gold_in_page/$n   (retrieval ceiling)"
echo "answer copied from injected page: $consumed/$n   (consumption)"
echo "gold-name hits  no-context: $hit_none/$n   with-DCY: $hit_dcy/$n"
