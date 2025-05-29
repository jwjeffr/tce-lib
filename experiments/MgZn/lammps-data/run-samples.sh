cd samples/;

for dir in */; do
    cd $dir;
    pwd;
    ../../lmp_gpu -in ../../relax.in -log none
    cd ..;
done

cd ..