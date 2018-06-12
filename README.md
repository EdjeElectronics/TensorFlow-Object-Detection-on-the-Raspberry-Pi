# How to Use the TensorFlow Object Detection API on the Raspberry Pi

## Introduction
This guide provides step-by-step instructions for how to set up TensorFlow’s Object Detection API on the Raspberry Pi. By following the steps in this guide, you will be able to use your Raspberry Pi to perform object detection on live video feeds from a Picamera or USB webcam. Combine this guide with my <link> tutorial on how to train your own neural network to identify specific objects</link>, and you use your Pi for unique detection applications such as:

* Letting you know when your cat wants to be let inside or outside
* Telling you if there are any parking spaces available in front of your apartment building
* [Beehive bee counter](http://matpalm.com/blog/counting_bees/)
* [Counting cards at the blackjack table??](https://hackaday.io/project/27639-rainman-20-blackjack-robot)
* And anything else you can think of!

*Picture of kitty cat detector coming soon*

I will also post a YouTube video that walks through this guide step-by-step. Please check back later for a link to the video.

The guide walks through the following steps:
1. Update the Raspberry Pi
2. Install TensorFlow
3. Install OpenCV
4. Compile and install Protobuf
5. Set up TensorFlow directory structure and the PYTHONPATH variable
6. Detect objects!!

The repository also includes the Object_detection_picamera.py script, which is a Python script that loads an object detection model in TensorFlow and uses it to detect objects in a Picamera video feed. The guide was written for TensorFlow v1.8.0 on a Raspberry Pi Model 3B running Raspbian Stretch v9. It will likely work for newer versions of TensorFlow.

## Steps
### 1. Update the Raspberry Pi
First, the Raspberry Pi needs to be fully updated. Open a terminal and issue:
```
sudo apt-get update
sudo apt-get dist-upgrade
```
Depending on how long it’s been since you’ve updated your Pi, the upgrade could take anywhere between a minute and an hour.

### 2. Install TensorFlow
