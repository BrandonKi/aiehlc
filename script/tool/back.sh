#!/usr/bin/env bash
###############################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
###############################################################################
###############################################################################
# back.sh - Directory bookmark stack (source this file, do not execute it)
#
# Usage:
#   back                  - list all bookmarked paths
#   back push [p1 p2 ..] - push paths (default: cwd) onto the stack
#   back <n>              - cd to the nth bookmarked path (1-based)
#   back pop              - pop the last entry and cd to it
#   back rm <n>           - remove the nth entry
#   back clear            - clear the entire stack
#
# The stack persists in ~/.back_stack across all terminals.
###############################################################################

back() {
    local stack_file="$HOME/.back_stack"

    # Ensure the stack file exists
    [ -f "$stack_file" ] || touch "$stack_file"

    # --- No arguments: list ---
    if [ $# -eq 0 ]; then
        if [ ! -s "$stack_file" ]; then
            echo "(stack is empty)"
            return 0
        fi
        local i=1
        while IFS= read -r line; do
            printf "  %2d  %s\n" "$i" "$line"
            i=$((i + 1))
        done < "$stack_file"
        return 0
    fi

    local cmd="$1"
    shift

    case "$cmd" in
        # --- push ----------------------------------------------------------
        push)
            if [ $# -eq 0 ]; then
                # push cwd when no path given
                set -- "$(pwd)"
            fi
            for p in "$@"; do
                local abs
                abs="$(cd "$p" 2>/dev/null && pwd)" || {
                    echo "back: invalid path: $p"
                    continue
                }
                echo "$abs" >> "$stack_file"
                echo "  pushed: $abs"
            done
            echo "  ($(wc -l < "$stack_file" | tr -d ' ') entries)"
            ;;

        # --- pop -----------------------------------------------------------
        pop)
            if [ ! -s "$stack_file" ]; then
                echo "back: stack is empty"
                return 1
            fi
            local last
            last="$(tail -1 "$stack_file")"
            sed -i '$ d' "$stack_file"
            cd "$last" || { echo "back: cd failed: $last"; return 1; }
            echo "  popped -> $last"
            exec bash
            ;;

        # --- rm <n> --------------------------------------------------------
        rm)
            local idx="${1:-}"
            if [ -z "$idx" ] || ! [[ "$idx" =~ ^[0-9]+$ ]]; then
                echo "Usage: back rm <n>"
                return 1
            fi
            local total
            total="$(wc -l < "$stack_file" | tr -d ' ')"
            if [ "$idx" -lt 1 ] || [ "$idx" -gt "$total" ]; then
                echo "back: index $idx out of range (1..$total)"
                return 1
            fi
            local removed
            removed="$(sed -n "${idx}p" "$stack_file")"
            sed -i "${idx}d" "$stack_file"
            echo "  removed #$idx: $removed"
            ;;

        # --- clear ---------------------------------------------------------
        clear)
            > "$stack_file"
            echo "  stack cleared"
            ;;

        # --- <number>: cd to nth entry -------------------------------------
        [0-9]*)
            local idx="$cmd"
            if [ ! -s "$stack_file" ]; then
                echo "back: stack is empty"
                return 1
            fi
            local target
            target="$(sed -n "${idx}p" "$stack_file")"
            if [ -z "$target" ]; then
                local total
                total="$(wc -l < "$stack_file" | tr -d ' ')"
                echo "back: index $idx out of range (1..$total)"
                return 1
            fi
            cd "$target" || { echo "back: cd failed: $target"; return 1; }
            echo "  -> $target"
            exec bash
            ;;

        # --- help / unknown ------------------------------------------------
        *)
            echo "Usage: back [push [path ...] | pop | rm <n> | clear | <n>]"
            echo ""
            echo "  back              list all bookmarked paths"
            echo "  back push [p ..]  push paths (default: cwd)"
            echo "  back <n>          cd to nth bookmarked path"
            echo "  back pop          pop last entry and cd to it"
            echo "  back rm <n>       remove nth entry"
            echo "  back clear        clear the entire stack"
            return 1
            ;;
    esac
}

# --- Tab completion --------------------------------------------------------
_back_completions() {
    local cur="${COMP_WORDS[COMP_CWORD]}"
    local prev="${COMP_WORDS[COMP_CWORD-1]}"

    if [ "$COMP_CWORD" -eq 1 ]; then
        COMPREPLY=( $(compgen -W "push pop rm clear help" -- "$cur") )
        # Also offer numbers if stack is non-empty
        if [ -f "$HOME/.back_stack" ] && [ -s "$HOME/.back_stack" ]; then
            local total
            total="$(wc -l < "$HOME/.back_stack" | tr -d ' ')"
            local nums
            nums="$(seq 1 "$total")"
            COMPREPLY+=( $(compgen -W "$nums" -- "$cur") )
        fi
    elif [ "$prev" = "push" ]; then
        # Complete directories
        COMPREPLY=( $(compgen -d -- "$cur") )
    elif [ "$prev" = "rm" ]; then
        if [ -f "$HOME/.back_stack" ] && [ -s "$HOME/.back_stack" ]; then
            local total
            total="$(wc -l < "$HOME/.back_stack" | tr -d ' ')"
            COMPREPLY=( $(compgen -W "$(seq 1 "$total")" -- "$cur") )
        fi
    fi
}
complete -F _back_completions back
