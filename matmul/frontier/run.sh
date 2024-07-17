#!/bin/bash

#SBATCH -N 1
#SBATCH -q debug
#SBATCH --time=00:15:00
#SBATCH -A csc547
#SBATCH --gpus-per-node=1
#SBATCH --exclusive

export TENSILE_DB=0x40

: '
m_list=(2048)
k_multiplier_list=(1 1 1 4)
n_multiplier_list=(3 1 4 1)
h_list=(12288)

for m in "${m_list[@]}"
do
        for i in {0..3}; do
		k_multiplier="${k_multiplier_list[i]}"
                n_multiplier="${n_multiplier_list[i]}"

		for h in "${h_list[@]}"
		do
			k=$((k_multiplier * h))
			n=$((n_multiplier * h))
			srun -N 1 ./gemm-rectangular.x "$m" "$k_multiplier" "$n_multiplier" "$h" > "NT/${m}_${k}_${n}.out"
		done
	done
done
'

srun -N 1 ./gemm-rectangular.x "2048" "1" "1.5" "12288" > "TN/2048_12288_18432.out"
srun -N 1 ./gemm-rectangular.x "2048" "0.5" "1" "12288" > "TN/2048_6144_12288.out"
srun -N 1 ./gemm-rectangular.x "2048" "1" "2" "12288" > "TN/2048_12288_24576.out"
srun -N 1 ./gemm-rectangular.x "2048" "2" "1" "12288" > "TN/2048_24576_12288.out"

