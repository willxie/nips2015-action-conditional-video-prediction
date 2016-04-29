if [ -z "${4}" ] 
  then echo "[weights (.caffemodel)] [num_frames_in_stack] [num_action] [num_step]"; exit 0
fi

MODEL=1
WEIGHTS=${1}
ACT=${3}
K=${2}
STEP=${4}
GPU=0

PYTHONPATH=$PWD/../caffe/python
python ../test_sequential.py --model $MODEL --weights $WEIGHTS --num_act ${ACT} --K $K --num_step $STEP --gpu $GPU ${5} ${6} ${7} ${8} ${9} ${10} ${11} ${12} ${13} ${14} ${15} ${16} ${17} ${18} ${19} ${20}
