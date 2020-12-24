This is a handwritten digit recognizer based on MNIST.  



ROOT  
│  CNN.py: the first net we made, with great number of connections, not so easy to train on PC   
│  LN5.py: based on the LeNet-5 by Yann LeCun, and we did some adaptations   
│  KFold.py: added KFold on LN5 to improve   
│  log_kf.txt: training accuracy of models in KFold.py, maybe for weight adjustment in the future   
│  presentation.pptx   
│  
├─conv_10_100.7z: '10' models with '100' episodes trained each   
│      conv_10_100_[0].pkl   
│      ...   
│      conv_10_100_[9].pkl   
│      
└─csv.7z   
│      sample_submission.csv: sample from Kaggle   
│      submission.csv: prediction of test data   
│      test.csv: test data   
│      train.csv: train data   
        
