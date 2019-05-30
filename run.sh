#!/bin/sh

GPU=$1

CUDA_VISIBLE_DEVICES=$GPU python agent.py
