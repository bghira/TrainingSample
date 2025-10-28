#!/usr/bin/env bash

pip install repairwheel

repairwheel -o $(dirname "$1") -l "${OPENCV_LINK_PATHS}" "$1"
