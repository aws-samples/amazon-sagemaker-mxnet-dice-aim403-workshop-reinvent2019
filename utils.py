from __future__ import absolute_import

import base64
from base64 import b64decode
import contextlib
import io
import json
import os
import time
import shlex
import shutil
import subprocess
import sys
import tempfile


import boto3
from IPython.display import HTML, Audio
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image





def get_execution_role(role_name="sagemaker", aws_account=None, aws_region=None):
    """
    Create sagemaker execution role to perform sagemaker task

    Args:
        role_name (string): name of the role to be created
        aws_account (string): aws account of the ECR repo
        aws_region (string): aws region where the repo is located
    """
    session = boto3.Session()
    aws_account = aws_account or session.client("sts").get_caller_identity()['Account']
    aws_region = aws_region or session.region_name

    assume_role_policy_document = json.dumps({
        "Version": "2012-10-17",
        "Statement": [
            {
                "Effect": "Allow",
                "Principal": {
                    "Service": ["sagemaker.amazonaws.com", "robomaker.amazonaws.com"]
                },
                "Action": "sts:AssumeRole"
            }
        ]
    })

    client = session.client('iam')
    try:
        client.get_role(RoleName=role_name)
    except client.exceptions.NoSuchEntityException:
        client.create_role(
            RoleName=role_name,
            AssumeRolePolicyDocument=str(assume_role_policy_document)
        )

        print("Created new sagemaker execution role: %s" % role_name)

    client.attach_role_policy(
        PolicyArn='arn:aws:iam::aws:policy/AmazonSageMakerFullAccess',
        RoleName=role_name
    )

    return client.get_role(RoleName=role_name)['Role']['Arn']



def show_webcam():
    quality = 0.8
    size = (512, 384)
    
    
    VIDEO_HTML = """
    <h3>Live Feed</h3>
    <video autoplay width=%d height=%d></video>
    <h3>Predicted Feed</h3>
    <canvas id="annotated" width=%d height=%d ></canvas>
    <script>

    var video = document.querySelector('video')
    var canvas = document.getElementById('annotated')


    navigator.mediaDevices.getUserMedia({ video: true })
      .then(stream=> video.srcObject = stream)


    function handle_output(output) {
        var canvas = document.getElementById('annotated')
        var ctx = canvas.getContext("2d");
        var image = new Image();
        image.onload = function() {
           ctx.drawImage(image, 0, 0);
        };
        image.src = output.content.text
        update_image()
    }

    function update_image() {
        var video = document.querySelector('video')
        var canvas = document.createElement('canvas')
        var [w,h] = [video.offsetWidth, video.offsetHeight]
        canvas.width = w
        canvas.height = h
        canvas.getContext('2d')
              .drawImage(video, 0, 0, w, h)
        command = 'get_annotated_image("' + canvas.toDataURL('image/jpeg', %f) + '")'

        var callbacks = {
            iopub : {
                 output : handle_output,
            }
        }
        IPython.notebook.kernel.execute(command, callbacks)
    }

    setTimeout(update_image, 1000);

    </script>
    """

    display(HTML(VIDEO_HTML % (size[0], size[1], size[0], size[1], quality)))
    