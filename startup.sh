#!/bin/bash

export is_share="False"

# kill previous process of live_voice.py
ps aux | grep '[l]ive_voice.py' | awk '{print $2}' | xargs -r kill -9

# get and enter script directory
SCRIPT_PATH=$(realpath "$0")
SCRIPT_DIR=$(dirname "$SCRIPT_PATH")
cd $SCRIPT_DIR

# activate conda environment and run live_voice.py
source activate gpt_sovits
nohup python live_voice.py > app.log 2>&1 &
