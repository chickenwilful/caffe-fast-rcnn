make clean
make all -j8
make pycaffe
cd ..
./experiments/scripts/default_caffenet.sh