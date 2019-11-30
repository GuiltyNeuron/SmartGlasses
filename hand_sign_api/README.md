# :ok_hand: Hand Sign API

## Introduction
This API is responsible of detecting human hand then recognise the sign if exists.
For the detection, we are going to train one of those model : Faster RCNN, SSD or YOLO.
And for the recognition, we are going to use and train a CNN model.

The goal of using this API is as a User interface in place of the chatbot, because in places 
where there is a lot of noise (public spaces) our smart glasses can't detect voice properly.
For the hand signs, we have those as classes :
![accuracy](images/amer_sign2.png)

## Results
##### CNN sign classifier training :
![accuracy](images/accuracy.png) ![accuracy](images/loss.png)


## :books: Documentation
- Dataset links for detector [link1](http://vision.soic.indiana.edu/projects/egohands/), [link2](https://sites.google.com/view/11khands)
- Dataset link for sign classification [Link](https://www.kaggle.com/datamunge/sign-language-mnist)
## Licence
GuideMeGlasses
:eyeglasses: