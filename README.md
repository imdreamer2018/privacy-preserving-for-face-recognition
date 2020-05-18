# **Privacy-preserving for face-recognition based on secure multiparty computation and siamese network**
## Paper model structure

![paper](http://imdreamer.oss-cn-hangzhou.aliyuncs.com/picGo/model%20architecture.png)

## Table of Contents

- [Background](https://github.com/imdreamer2018/privacy-preserving-for-face-recognition#background)
- [Install](https://github.com/imdreamer2018/privacy-preserving-for-face-recognition#install)
- [Improvement](https://github.com/imdreamer2018/privacy-preserving-for-face-recognition#Improvement)
- [Future](https://github.com/imdreamer2018/privacy-preserving-for-face-recognition#Future)
- [License](https://github.com/imdreamer2018/privacy-preserving-for-face-recognition#license)

## Background

<p align="center"><font face="roman" size=5><b>Abstract</b></font></p>

<p><font face="roman" size=4>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Face recognition is a kind of biometric recognition technology based on the facial feature information of a person. With the rapid development of Internet technology, in recent years, the technology of analyzing images based on deep learning convolutional neural networks has achieved great success. Face recognition has been increasingly applied to various fields. With the widespread use of biometrics, important privacy issues have also arisen. The face recognition system collects the user's face data for commercial use. Face data is usually unique and irreplaceable, once leaked, it will cause great damage to the user's privacy. This paper first proposes a face recognition privacy protection scheme based on secure multi-party computing and Siamese neural network, and establishes a face recognition privacy protection model. By calculating sensitive data from multiple sources, it can perform face recognition. At the same time, the privacy of face data can be guaranteed.
  <br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;The solution of this paper is divided into two phases: extracts the face embedding data phase and the face recognition privacy preserving phase.
 <br>
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<font color="red">Extracts the face embedding data phase.</font> The offline client extracts facial feature embeddings. First, the face image is pre-processed through face detection and alignment, and then the deep learning model based on the Siamese neural network is used to process the face features to extract low-dimensional face representations (face embeddings). Face embedding refers to removing the final classification layer after the neural network training is completed, and using the output of the previously fully connected layer as a low-dimensional face representation. The client then sends the private data of face embeddings to two remote non-competing servers in a secret sharing manner.
  <br>
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<font color="red">The face recognition privacy preserving phase.</font> Training and prediction of face recognition privacy preserving model on online server. Two non-competitive servers private train models using cloud computing through joint multifaceted face embedding data. The parameters of the face recognition privacy protection model after training are stored by the two non-competing servers in the form of secret shared secret text. Later, the trained model is used to recognizer the face by combining the face embedding data to be recognized. The recognition result is still in the form of secret ciphertext, and the recognition result is returned to the client. The client reconstructs the secret to obtain the plain text recognition result, and completes the person face recognition.
  <br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Finally, this paper analyzes the correctness and security of the scheme, and implements the scheme through experiments. The results show that the face recognition privacy protection scheme based on secure multiparty computing and Siamese neural network not only has reliable security. And has the advantages of light weight, high accuracy, and computational efficiency. This solution has important application value and practical significance for building a safe and efficient face recognition privacy protection system.
	<br>
<b>Keywords</b>: face recognition; privacy preserving; secure multiparty computing; secret sharing; deep learning; siamese neural network
  </p>

------

### Application characteristics including:


- FaceImageGenerator tool
- Face align by dlib model
- Extract face embedded by openFace model 
- Train encrypted classification model base on MPC and Secret Sharing
- FaceRecoginzer tool

------

## Install

```shell
#environment python3
git clone https://github.com/imdreamer2018/privacy-preserving-for-face-recognition.git
cd privacy-preserving-for-face-recognition
pip install -r requirements.txt
```

------

### Download model

You can click this [link](http://imdreamer.oss-cn-hangzhou.aliyuncs.com/picGo/models.zip) download some models include dlib,openFace nn4 small etc.Then put models on project root directory.

------

### Application execution steps

```python
#step1:Collect some face images by faceGenerator.py,then put these images on dir faceImages
python faceGenerator.py
#step2:Generate face embedded by generateFaceEmbedded.py,then put these embedded on dir faceData
python generateFaceEmbedded.py
#step3:Train your privacy preserving face recognition model.And these model will be save in dir weights
python encryptedClassification.py
#step4:Run face recognition program tool.
python faceRecognizer.py
```

## Tree

```shell
.
├── faceImages
├── faceData
├── models
├── mpc
├── openface
├── visualization_process
├── weights
├── config.py
├── faceGenerator.py
├── faceRecognizer.py
├── generateFaceEmbedded.py
├── encryptedClassification.py
├── encryptedPrediction.py
├── requirements.txt
└── README.md
```

### Face recognition

![](http://imdreamer.oss-cn-hangzhou.aliyuncs.com/picGo/face.png)

## Improvement

- In this paper, high-order continuous polynomials are used to approximate nonlinear activation functions such as sigmoid and softmax. Compared with the original nonlinear activation functions, not only will there be a large computational cost, but also a small amount of accuracy loss after training. At present, no new friendly activation function is found to be a better alternative method than using Taylor series to approximate the exponential function. In the future work, we can make better improvements in the friendly activation function, which can improve the calculation efficiency and the accuracy of the model.
- When extracting face embeddings from offline clients, this paper uses the OpenFace pre-training model to obtain face embeddings. The pre-training model directly affects the accuracy of the face recognition privacy protection model. In future work, you can choose a pre-trained model with better performance and higher accuracy, which can improve the accuracy of the face recognition privacy protection model.
- More complex neural network models can be designed to improve the accuracy of the face recognition privacy protection model.

## Future

In order to ensure that there is no network delay, and in the case of limited funds and my capacity.Actually,the privacy preserving for face recognition project running in local computer, not really running on a cloud server.

In future,I hope this project can really achieve privacy preserving face recognition by MPC and Secret.Because of I have writen the train protocol in MPC,so I will use the [tf-encrypted](https://github.com/tf-encrypted/tf-encrypted) achieve my paper.Wait and see...

## License

[MIT](https://github.com/imdreamer2018/privacy-preserving-for-face-recognition/blob/master/LICENSE)@Imdreamer

