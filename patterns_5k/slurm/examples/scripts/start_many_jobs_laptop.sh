#!/bin/bash

for seed in {0..9}; do
  python -m optlearn.main --config=configs/heuristic/t16/t16_uniform.yaml --random_seed=$seed &
  python -m optlearn.main --config=configs/heuristic/t16/t16_uniform_9.yaml --random_seed=$seed &
  python -m optlearn.main --config=configs/heuristic/t16/t16_uniform_8.yaml --random_seed=$seed &
  python -m optlearn.main --config=configs/heuristic/t16/t16_uniform_7.yaml --random_seed=$seed &
  python -m optlearn.main --config=configs/heuristic/t16/t16_uniform_6.yaml --random_seed=$seed &
  python -m optlearn.main --config=configs/heuristic/t16/t16_uniform_5.yaml --random_seed=$seed &

#   python -m optlearn.main --config=configs/heuristic/t16/t16_adaptive.yaml --random_seed=$seed --n_targets=16 &
done

