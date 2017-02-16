# keras-frcnn
Keras implementation of Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks

CURRENT STATUS:
- only resnet50 architecture is currently supported
- weights for theano backend coming shortly

USAGE:
- train_frcnn.py can be used to train a model. To train on Pascal VOC data, simply do:
python train_frcnn.py /path/to/pascalvoc/
- the Pascal VOC data set (images and annotations for bounding boxes around the classified objects) can be obtained from: http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar

- simple_parser.py provides an alternative way to input data, using a text file. Simply provide a text file, with each
line containing:

`filepath,x1,y1,x2,y2,class_name`

For example:

/data/imgs/img_001.jpg,837,346,981,456,cow
/data/imgs/img_002.jpg,215,312,279,391,cat

- test_frcnn.py can be used to perform inference, given pretrained weights. Specify a path to the folder containing
images:
python test_frcnn.py /path/to/imgs/

NOTES:
config.py contains all settings for the train or test run. The default settings match those in the original Faster-RCNN
paper. The anchor box sizes are [128, 256, 512] and the ratios are [1:1, 1:2, 2:1].

Example output:

![ex1](http://i.imgur.com/UtGXhtd.jpg)
![ex2](http://i.imgur.com/Szf78o2.jpg)
![ex3](http://i.imgur.com/OjVXTbn.jpg)
![ex4](http://i.imgur.com/9Fbe2Ow.jpg)
