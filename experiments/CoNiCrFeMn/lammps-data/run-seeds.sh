awk -v seed=0 'BEGIN { srand(seed); for (i=1; i<=250; i++) print int(rand() * 10000) }' |
while read num; do
    ./lmp_gpu -in relax.in -var seed $num -log none
done

