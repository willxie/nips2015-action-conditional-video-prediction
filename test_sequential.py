from argparse import ArgumentParser
import tempfile
import sys
import caffe
import net as N
import six
import numpy as np
from collections import OrderedDict
from caffe.proto import caffe_pb2 as PB
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont
import matplotlib.pyplot as plt
import pylab
import cv2
import os
import subprocess as sp

def tag(i):
  return '-{:0>2d}'.format(i)

# Data is [0, 1]
def pre_process(data, mean, scale):
  t = data.copy().squeeze().astype('float64')
  t = t.transpose([2, 0, 1])
  t /= scale
  t -= mean
  t *= scale
  # t = t.clip(0, 255)
  return t.squeeze()


def post_process(data, mean, scale):
  t = data.copy().squeeze()
  t /= scale
  t += mean
  t = t.clip(0, 255)
  return t.astype('uint8').squeeze().transpose([1, 0, 2]).transpose([0, 2, 1])

# T is the position in time and K is the number of frames used per prediction 
def main(model, num_models, weights, weights2, weights3, weights4, K, num_act, num_step, num_iter,
        gpu, data, mean, video, ):
  font = ImageFont.truetype('/usr/share/fonts/dejavu/DejaVuSans.ttf', 20)
  caffe.set_mode_gpu()
  caffe.set_device(gpu)
  #caffe.set_mode_cpu()

  # A list of trained networks
  net_list = []
  for model_idx in range(num_models):
    data_net_file, net_proto = N.create_netfile(model, 
        data, mean, K + num_step, K, 1, num_act, num_step=num_step, mode='data',
        # file_name='data.prototxt'
        )
    test_net_file, net_proto = N.create_netfile(model, data, mean, K, K, 
        1, num_act, num_step=1, mode='test', 
        # file_name='model.prototxt'
        )

    data_net = caffe.Net(data_net_file, caffe.TEST)
    test_net = caffe.Net(test_net_file, caffe.TEST)
    test_net.copy_from(weights)
    net_list.append(test_net)

  # Mean array used for element wise subtraction
  mean_blob = caffe.proto.caffe_pb2.BlobProto()
  mean_bin = open(mean, 'rb').read()
  mean_blob.ParseFromString(mean_bin)
  mean_arr = caffe.io.blobproto_to_array(mean_blob).squeeze()

  # Not useful
  #  mu = np.array(caffe.io.blobproto_to_array(mean_blob))
  #  mu = mu[0]
  #  mu = mu.mean(1).mean(1)

  transformer = caffe.io.Transformer({'data': (50, 3, 210, 160)})
  #  transformer.set_transpose('data', (2,0,1))  # move image channels to outermost dimension
  #  transformer.set_mean('data', mean_arr)            # subtract the dataset-mean value in each channel
  #  transformer.set_raw_scale('data', 255)      # rescale from [0, 1] to [0, 255]
  transformer.set_channel_swap('data', (2,1,0))  # swap channels from RGB to BGR

  # Read the actions
  with open("/work/04018/wxie/maverick/nips2015-action-conditional-video-prediction/example/test/0000/" + "act.log", 'rb') as f: 
    action_list = [ int(next(f).rstrip('\n')) for x in range(num_iter) ]
 
  for i in range(0, num_iter):
    print("iteration " + str(i) + "/" + str(num_iter)) 
    data_net.forward()

     # TODO stack them into one tensor 

    image_path = "/work/04018/wxie/maverick/nips2015-action-conditional-video-prediction/example/test/0000/{0:05d}.png".format(i)
    print(image_path)
    image = caffe.io.load_image(image_path) # RGB h x w x c
    # Change color 
    image_bgr = image.copy()
    image_bgr[:,:,0] = image[:,:,2]
    image_bgr[:,:,1] = image[:,:,1]
    image_bgr[:,:,2] = image[:,:,0]
    
    # transformed_image = transformer.preprocess('data', image)
    processed_image = pre_process(image, mean_arr, 1./255)
    # t = post_process(processed_image, mean_arr, 1./255) 
    # how_img = np.hstack((t, t))
    # test_img = Image.fromarray(how_img)
    # cv2.imwrite("img/1_{0:05d}.jpg".format(i), np.array(test_img))


    # data_blob = transformed_image
    # Expand to the right dim 
    data_blob = processed_image
    data_blob = np.expand_dims(data_blob, axis=0)
    data_blob = np.expand_dims(data_blob, axis=0)
    
    # data_blob = data_net.blobs['data'].data
    # act_blob = data_net.blobs['act'].data

    act_blob = np.array([[[0.]*num_act]])
    act_blob[:,:,action_list[i]] = 1.

    pred_img_list = []
    pred_data = np.zeros((3, 210, 160), np.float)
    true_data = np.zeros((3, 210, 160), np.float)

    for test_net in net_list:
      test_net.blobs['data'].data[:] = data_blob[:, 0:K, :, :, :]
      test_net.blobs['act'].data[:] = act_blob[:, K-1, :]

      test_net.forward()

      pred_data[:] = test_net.blobs['x_hat'+tag(K+1)].data[:]
        # true_data[:] = data_net.blobs['data'].data[:, K+step, :, :, :]
      pred_img = post_process(pred_data, mean_arr, 1./255)
      pred_img_list.append(pred_img.copy())

    true_data[:] = pre_process(image_bgr, mean_arr, 1./255)
    true_img = post_process(true_data, mean_arr, 1./255)
    pred_img_list.append(true_img)

    # display
    print(pred_img_list[0].shape)
    show_img = np.hstack(tuple(pred_img_list))
    top_pad = np.zeros((35, show_img.shape[1], show_img.shape[2]), np.uint8)
    show_img = np.vstack((top_pad, show_img))
    img = Image.fromarray(show_img)
    draw = ImageDraw.Draw(img)
    draw.text((10, 10), 'Step:' , fill=(255, 255, 255), font=font)
    # cv2.imshow('Display', np.array(img))
    cv2.imwrite("img/{0:05d}.jpg".format(i), np.array(img))

if __name__ == "__main__":
  parser = ArgumentParser()
  parser.add_argument("--model", type=int, dest="model",
                      default=1, help="1:CNN 2:RNN")
  parser.add_argument("--weights", type=str, dest="weights",
                      default="", help="Pre-trained caffemodel")
  parser.add_argument("--data", type=str, dest="data",
                      default="test", help="Test data directory")
  parser.add_argument("--K", type=int, dest="K",
                      default=11, help="Number of initial frames")
  parser.add_argument("--mean", type=str, dest="mean",
                      default="mean.binaryproto", help="Mean file")
  parser.add_argument("--num_act", type=int, dest="num_act",
                      default=0, help="Number of actions")
  parser.add_argument("--num_step", type=int, dest="num_step",
                      default=1, help="Number of steps")
  parser.add_argument("--num_iter", type=int, dest="num_iter",
                      default=30, help="Number of iterations")
  parser.add_argument("--gpu", type=int, dest="gpu",
                      default=0, help="GPU device id")
  parser.add_argument("--video", type=str, dest="video",
                      default="", help="Output video directory")
  parser.add_argument("--num_models", type=int, dest="num_models",
                      default=1, help="The number of models we are using (4 max)")
  parser.add_argument("--weights2", type=str, dest="weights2",
                      default="", help="Pre-trained caffemodel 2")
  parser.add_argument("--weights3", type=str, dest="weights3",
                      default="", help="Pre-trained caffemodel 3")
  parser.add_argument("--weights4", type=str, dest="weights4",
                      default="", help="Pre-trained caffemodel 4")


  args = parser.parse_args()
  main(**vars(args))
