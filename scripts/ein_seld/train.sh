#!/bin/bash

GPU_ID=3

CUDA_VISIBLE_DEVICES=$GPU_ID python code/main.py train --port=12360