# Shared model/pack/config resolution for TP2 launchers. Source, do not exec.
#
# Selection order for the pack:
#   1. MAYA_PACK / --pack PATH   — explicit pack directory
#   2. MAYA_MODEL / --model NAME — $MAYA_MODELS_DIR/NAME/pack
#   3. default Maya-S-v2-IQ2_XXS
#
# Serve launchers also need a base JSON (tokenizer, env, exe). Selection:
#   1. MAYA_CONFIG / --config PATH
#   2. maya-<model>-hip.json beside the repo root when that file exists
#   3. maya-maya-s-v2-iq2_xxs-hip.json
#
# MAYA_MODELS_DIR defaults to /home/ugo/software/Maya-data/models.

maya_models_dir() {
  printf '%s\n' "${MAYA_MODELS_DIR:-/home/ugo/software/Maya-data/models}"
}

maya_default_model() {
  printf '%s\n' "${MAYA_MODEL:-Maya-S-v2-IQ2_XXS}"
}

# Print the resolved pack path. Does not check existence.
maya_resolve_pack() {
  if [ -n "${MAYA_PACK:-}" ]; then
    printf '%s\n' "$MAYA_PACK"
    return 0
  fi
  printf '%s\n' "$(maya_models_dir)/$(maya_default_model)/pack"
}

# Config basename used by the installer (Maya-S-v2-IQ2_XXS -> maya-maya-s-v2-iq2_xxs-hip.json).
maya_config_basename_for_model() {
  printf 'maya-%s-hip.json\n' "$(printf '%s' "$1" | tr '[:upper:]' '[:lower:]')"
}

# Print the resolved base config path. Does not check existence.
maya_resolve_config() {
  local root="${1:-${ROOT:-}}"
  if [ -n "${MAYA_CONFIG:-}" ]; then
    printf '%s\n' "$MAYA_CONFIG"
    return 0
  fi
  if [ -n "${MAYA_MODEL:-}" ] && [ -n "$root" ]; then
    local candidate="$root/$(maya_config_basename_for_model "$MAYA_MODEL")"
    if [ -f "$candidate" ]; then
      printf '%s\n' "$candidate"
      return 0
    fi
  fi
  if [ -n "$root" ]; then
    printf '%s\n' "$root/maya-maya-s-v2-iq2_xxs-hip.json"
  else
    printf '%s\n' "maya-maya-s-v2-iq2_xxs-hip.json"
  fi
}

# Emit a config JSON on stdout with --glm-pack (and tokenizer when present) rewritten to $2.
maya_config_with_pack() {
  local config="$1" pack="${2:-}"
  if [ -z "$pack" ]; then
    cat "$config"
    return 0
  fi
  jq --arg pack "$pack" '
    (.args // []) as $a
    | ($a | index("--glm-pack")) as $i
    | (if $i == null
       then .args = ($a + ["--glm-pack", $pack])
       else .args = ($a | .[$i + 1] = $pack)
       end)
    | if (.tokenizer // "") != "" then .tokenizer = ($pack + "/tokenizer") else . end
  ' "$config"
}
