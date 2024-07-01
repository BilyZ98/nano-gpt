#rm slurm*

yhbatch -n 1 -N 1 -p GPU_A800 test.sh
