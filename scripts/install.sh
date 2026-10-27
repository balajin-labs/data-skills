#!/usr/bin/env bash
# Install the data-skills skills into one or more coding agents' skill folders.
#
#   ./scripts/install.sh                      # ~/.agents/skills + ~/.claude/skills: covers most agents
#   ./scripts/install.sh cursor copilot       # only these agents ("all" for every folder below)
#   ./scripts/install.sh --project ~/my-repo  # into one repository instead (commit it to share with the team)
#   ./scripts/install.sh --link               # symlink instead of copy, so `git pull` updates the skills
#   ./scripts/install.sh --uninstall          # remove what this script installed
#
# Agents and the folders they read:
#   agents   ~/.agents/skills            Codex, Devin, Windsurf, Gemini CLI, OpenCode, Amp and other .agents tools
#            (alias: codex)
#   claude   ~/.claude/skills            Claude Code (or use the plugin: /plugin marketplace add balajin-labs/data-skills)
#   devin    ~/.config/devin/skills      Devin CLI's own folder; only needed if you skip "agents" (it reads both)
#   cursor   ~/.cursor/skills            Cursor
#   copilot  ~/.copilot/skills           GitHub Copilot
# With --project DIR the same agents map to DIR/.agents/skills, DIR/.claude/skills, DIR/.devin/skills,
# DIR/.cursor/skills and DIR/.github/skills.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC="$ROOT/skills"
ALL=(agents claude devin cursor copilot)
targets=() project="" mode="copy" action="install"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --project) project="${2:?--project needs a directory}"; shift 2 ;;
    --link) mode="link"; shift ;;
    --uninstall) action="uninstall"; shift ;;
    -h|--help) sed -n '2,22p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    all) targets=("${ALL[@]}"); shift ;;
    codex) targets+=(agents); shift ;;
    agents|claude|devin|cursor|copilot) targets+=("$1"); shift ;;
    *) echo "unknown option or agent: $1 (run with --help)" >&2; exit 2 ;;
  esac
done
[[ ${#targets[@]} -eq 0 ]] && targets=(agents claude)
# shellcheck disable=SC2207
targets=($(printf '%s\n' "${targets[@]}" | awk '!seen[$0]++'))

dest_for() {
  if [[ -n "$project" ]]; then
    case "$1" in
      agents) echo "$project/.agents/skills" ;; claude) echo "$project/.claude/skills" ;;
      devin) echo "$project/.devin/skills" ;; cursor) echo "$project/.cursor/skills" ;;
      copilot) echo "$project/.github/skills" ;;
    esac
  else
    case "$1" in
      agents) echo "$HOME/.agents/skills" ;; claude) echo "$HOME/.claude/skills" ;;
      devin) echo "${XDG_CONFIG_HOME:-$HOME/.config}/devin/skills" ;; cursor) echo "$HOME/.cursor/skills" ;;
      copilot) echo "$HOME/.copilot/skills" ;;
    esac
  fi
}

skills=()
for d in "$SRC"/*/; do [[ -f "$d/SKILL.md" ]] && skills+=("$(basename "$d")"); done

for t in "${targets[@]}"; do
  dest="$(dest_for "$t")"
  for s in "${skills[@]}"; do
    target="$dest/$s"
    if [[ "$action" == "uninstall" ]]; then
      [[ -e "$target" || -L "$target" ]] && rm -rf "$target"
      continue
    fi
    mkdir -p "$dest"
    rm -rf "$target"
    if [[ "$mode" == "link" ]]; then
      ln -s "$SRC/$s" "$target"
    else
      mkdir -p "$target"
      (cd "$SRC/$s" && tar cf - --exclude '__pycache__' --exclude '*.pyc' .) | (cd "$target" && tar xf -)
    fi
  done
  echo "$t: ${action}ed ${#skills[@]} skills in $dest"
done

if [[ "$action" == "install" ]]; then
  echo
  echo "The check scripts need Python 3.9+ with pandas, numpy and scikit-learn:"
  echo "  pip install -r \"$ROOT/requirements.txt\""
fi
