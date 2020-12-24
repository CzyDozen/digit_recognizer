### digit_recognizer
This is a handwritten digit recognizer based on MNIST.  
We group of three designed it, attended the [Kaggle competition](https://www.kaggle.com/c/digit-recognizer) and submited it as homework.   

---

### File tree  
    - CNN.py: the first net we made, with great number of connections, not so easy to train on PC   
    - LN5.py: based on the LeNet-5 by Yann LeCun, and we did some adaptations   
    - KFold.py: added KFold on LN5 to improve   
    
    - log_kf.txt: training accuracy of models in KFold.py, maybe for weight adjustment in the future   
    - presentation.pptx   
     
    - conv_10_100.7z: '10' models with '100' episodes trained each   
        │conv_10_100_[0].pkl   
        │...   
        │conv_10_100_[9].pkl   
    - csv.7z   
        │sample_submission.csv: sample from Kaggle   
        │submission.csv: prediction of test data   
        │test.csv: test data   
        │train.csv: train data   

---

### HOW TO START:   
Unzip both the 7z files, put the content with the python files, NOT IN THE FOLDERS.   
If you run LN5.py, a brand new model will be train. After predicted, the model won't be saved.   
If you run KFold.py, you should decide to load_pkl() or kfold() at the end of file. kfold() takes long time to build fresh models and auto-save, please concern about FILE COVERAGE issue.   
