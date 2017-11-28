# LSTM_action_recognition
Fine grained tennis action recognition using LSTM.

1) Dowload the THETIS dataset from http://thetis.image.ece.ntua.gr/

2) Extract features from THETIS in 1) using Inception model trained on Imagenet. Using 'getFeatsHMDB_tosubmit.py' and the trained model available from https://github.com/tensorflow/models/blob/master/tutorials/image/imagenet/classify_image.py

3) Train the LSTM using features extracted in 2) and the 'learnExternalData' function in 'LSTM_hmdb_topredic_tosubmit.py'

4) Make predictions using 'predictExternalData' from 'LSTM_hmdb_topredic_tosubmit.py'
