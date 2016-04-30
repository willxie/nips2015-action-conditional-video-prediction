if [ -z "${4}" ] 
  then echo "[num_frames_in_stack] [num_action] [num_models (4 max)] [weights_i (.caffemodel)]... "; exit 0
fi

MODEL=1
NUMMODELS=${3}
WEIGHTS=${4}
if [ -z "${5}" ]; then
WEIGHTS2='--weights2 ${5}'
fi
if [ -z "${6}" ]; then
WEIGHTS3='--weights3 ${6}'
fi
if [ -z "${7}" ]; then
WEIGHTS4='--weights4 ${7}'
fi
ACT=${2}
K=${1}
STEP=1
GPU=0

PYTHONPATH=$PWD/../caffe/python
python ../test_sequential.py --model $MODEL --num_act ${ACT} --K $K --num_step $STEP --gpu $GPU --num_models $NUMMODELS --weights $WEIGHTS $WEIGHTS2 $WEIGHTS3 $WEIGHTS4 ${8} ${9} ${10} ${11} ${12} ${13} ${14} ${15} ${16} ${17} ${18} ${19} ${20}
