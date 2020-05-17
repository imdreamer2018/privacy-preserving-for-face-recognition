# face-recognition
**Paper Title:**  **Privacy-preserving for face-recognition based on secure multiparty computation and siamese network**

**Paper model structure**:

![paper](http://imdreamer.oss-cn-hangzhou.aliyuncs.com/picGo/model%20architecture.png)

## Table of Contents

- [Background](https://github.com/imdreamer2018/SimpleBlog#background)
- [Install](https://github.com/imdreamer2018/SimpleBlog#install)
- [License](https://github.com/imdreamer2018/SimpleBlog#license)

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
  </p></font>

------

<p><font face="roman" size=5>
  Application characteristics including:
</font></p>

<p><font face="roman" size=3 color="blue">
- FaceImageGenerator tool<br>
- Face align by dlib model<br>
- Extract face embedded by openFace model <br>
- Train encrypted classification model base on MPC and Secret Sharing<br>
- FaceRecoginzer tool</font></p>

------

## Install







