ACT=$1
PREFIX="cnn"
PYTHONPATH=$PWD/../caffe/python

mkdir -p ${PREFIX}
python ../train.py --model 1 --prefix $PREFIX"/1step" --lr 0.0001 --num_act ${ACT} --T 5 --K 4 --num_step 1 --batch_size 32 --test_batch_size 50  --num_iter 1500000
python ../train.py --model 1 --prefix $PREFIX"/3step" --lr 0.00001 --num_act ${ACT} --T 7 --K 4 --num_step 3 --batch_size 8 --test_batch_size 50  --weights $PREFIX"/1step_iter_1500000.caffemodel.h5" --num_iter 1000000
python ../train.py --model 1 --prefix $PREFIX"/5step" --lr 0.00001 --num_act ${ACT} --T 9 --K 4 --num_step 5 --batch_size 8 --test_batch_size 50  --weights $PREFIX"/3step_iter_1000000.caffemodel.h5" --num_iter 1000000
