#!/usr/bin/env bash

# SPDX-License-Identifier: Apache-2.0
# Copyright 2024 The Linux Foundation <matthew.watkins@linuxfoundation.org>

# Thin wrapper script to test workflow YAML code directly from a shell

set -eu -o pipefail
DEBUG="false"
export DEBUG

# Check script arguments

if [ $# -ne 1 ]; then
    # Provide a Github action/workflow YAML file as argument
    echo "Usage: $0 [workflow YAML file]"; exit 1
else
    SOURCE_FILE="$1"
    if [ ! -f "$SOURCE_FILE" ]; then
        echo "Specified file could not be read: $SOURCE_FILE"; exit 1
    fi
    SETUP_FILE="setup.txt"
    if [ -f "$SETUP_FILE" ]; then
        echo "Sourcing script actions/variables from: $SETUP_FILE"
        # shellcheck disable=SC1090
        source "$SETUP_FILE"
    else
        echo "No file found specifying inputs: $SETUP_FILE"
    fi
fi

# Check for required binaries

GIT_CMD=$(which git)
if [ ! -x "$GIT_CMD" ]; then
    echo "GIT command was NOT found in PATH"; exit 1
fi

MKTEMP_CMD=$(which mktemp)
if [ ! -x "$MKTEMP_CMD" ]; then
    echo "MKTEMP command was NOT found in PATH"; exit 1
fi

# Script debugging options

if [ $DEBUG = "true" ]; then
    # set -xv
    SHELL_SCRIPT="extracted.sh"
    PATH=".:$PATH"
    if [ -f "$SHELL_SCRIPT" ]; then
        # Remove any previously extracted code on subsequent runs
        rm "$SHELL_SCRIPT"
    fi
else
    SHELL_SCRIPT=$(mktemp -t script-XXXXXXXX.sh)
fi

# Functions

change_dir_error() {
    echo "Could not change directory"; exit 1
}

check_for_local_branch() {
    BRANCH="$1"
    git show-ref --quiet refs/heads/"$BRANCH"
    return $?
}

check_for_remote_branch() {
    BRANCH="$1"
    git ls-remote --exit-code --heads origin "$BRANCH"
    return $?
}

cleanup_on_exit() {
    if [ -f "$SHELL_SCRIPT" ]; then
        echo "Removing temporary shell code"
        rm "$SHELL_SCRIPT"
    fi
}

# Main script entry point

# Get organisation and repository name
URL=$(git config --get remote.origin.url)

# Take the above and store it converted as ORG_AND_REPO
# e.g. ModeSevenIndustrialSolutions/test-bootstrap
ORG_AND_REPO=${URL/%.git}
ORG_AND_REPO=${ORG_AND_REPO//:/ }
ORG_AND_REPO=$(echo "$ORG_AND_REPO" | awk '{ print $2 }')
# Variable below is currently unused
# HEAD_BRANCH=$("$GIT_CMD" rev-parse --abbrev-ref HEAD)
REPO_DIR=$(git rev-parse --show-toplevel)

# Change to top-level of GIT repository
CURRENT_DIR=$(pwd)
if [ "$REPO_DIR" != "$CURRENT_DIR" ]; then
    echo "Changing directory to: $REPO_DIR"
    cd "$REPO_DIR" || change_dir_error
fi

# The section below extracts shell code from the YAML file
echo "Attempting to parse shell code from: $SOURCE_FILE"
EXTRACT="false"
while read -r LINE; do
    if [[ "$LINE" = *"#SHELLCODEEND"* ]]; then
        EXTRACT="complete"
        break

    elif [[ "$LINE" = *"#SHELLCODESTART"* ]]; then
        EXTRACT="true"
        touch "$SHELL_SCRIPT"
        chmod a+x "$SHELL_SCRIPT"
        continue

    elif [ "$EXTRACT" = "true" ]; then
        echo "$LINE" >> "$SHELL_SCRIPT"
    fi
done < "$SOURCE_FILE"

# Only remove temporary files when NOT debugging
if [ "$DEBUG" != "true" ]; then
    trap cleanup_on_exit EXIT
fi

if [ -f "$SHELL_SCRIPT" ] && [ "$DEBUG" = "true" ]; then
    echo "Extracted code to file: $SHELL_SCRIPT"
fi

if [ "$EXTRACT" = "complete" ]; then
    echo "Executing extracted shell script/code..."
    # Shell code executed is temporary and cannot be checked by linting
    # https://www.shellcheck.net/wiki/SC1090
    # shellcheck disable=SC1090
    "$SHELL_SCRIPT"
else
    echo "Error: start/stop markers not found in file"
    exit 1
fi
