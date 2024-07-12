#!/bin/bash

#SBATCH -N 1
#SBATCH -q debug
#SBATCH --time=00:30:00
#SBATCH -A csc547
#SBATCH --gpus-per-node=1
#SBATCH --exclusive

export TENSILE_DB=0x40

m_list=(2048 4096 8192 16384)
k_multiplier_list=(1 1 1 4)
n_multiplier_list=(3 1 4 1)
h_list=(4096 5120 7168 9216 12288)

for m in "${m_list[@]}"
do
        for i in {0..3}; do
		k_multiplier="${k_multiplier_list[i]}"
                n_multiplier="${n_multiplier_list[i]}"

		for h in "${h_list[@]}"
		do
			k=$((k_multiplier * h))
			n=$((n_multiplier * h))
			srun -N 1 ./gemm-rectangular.x "$m" "$k_multiplier" "$n_multiplier" "$h" > "TT/${m}_${k}_${n}.out"
		done
	done
done

