#!/usr/bin/env bash

pip install repairwheel

repairwheel -o ./dist/ -l "${OPENCV_LINK_PATHS}" "$@"
