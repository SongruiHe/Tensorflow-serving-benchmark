from __future__ import print_function

import base64
import requests
from subprocess import Popen, PIPE


# The server URL specifies the endpoint of your server running the ResNet
# model with the name "resnet" and using the predict interface.
SERVER_URL = 'http://localhost:8501/v1/models/resnet:predict'

# The image URL is the location of the image we should send to the server
IMAGE_URL = '/home/kitten_small.jpg'


def main():

  pic = open(IMAGE_URL,"rb").read()
  jpeg_bytes = base64.b64encode(pic).decode('utf-8')
  predict_request = '{"instances" : [{"b64": "%s"}]}' % jpeg_bytes
  # print(predict_request)

  with open('input', 'w') as f:
    f.write(predict_request)

  input=open('input', 'r').read()

  response = requests.post(SERVER_URL, data=input)
  response.raise_for_status()
  print(response.json()['predictions'][0]['classes'])


  ab_cmd = "ab -c 4 -n 1000 -k -p /home/input -T text/plain http://0.0.0.0:8501/v1/models/resnet:predict > /home/result.txt"
  execute(ab_cmd, wait=True)

  # Send few requests to warm-up the model.
  # for _ in range(3):
  #   response = requests.post(SERVER_URL, data=predict_request)
  #   response.raise_for_status()


def execute(command, wait=False, stdout=None, stderr=None, shell=True):
    print(command)
    cmd = Popen(command, shell=shell, close_fds=True, stdout=stdout, stderr=stderr, universal_newlines=True)
    if wait:
        cmd.wait()
    return cmd


if __name__ == '__main__':
  main()
