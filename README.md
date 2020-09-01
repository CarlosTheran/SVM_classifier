**This work is sponsor by _CENTER FOR THE ADVANCEMENT OF WEARABLE TECHNOLOGIES_.**

![](CAWT.png)

_This repository provides an base code to analyze data using support vector machine. In order to run the code please install the following [requerements](https://github.com/CarlosTheran/SVM_classifier/blob/master/requerements.txt)_

In this repository you will find a Support Vector Machine (SVM) implementation file _dl_regressors_svm.py._ to predict material propersties from elemental composition. This code is a extention of the code provide in [Elemnet](https://github.com/CarlosTheran/ElemNet). 
To analyze the data using SVM you must to copy the file _dl_regressors_svm.py._ into the folder [elemnet](https://github.com/CarlosTheran/ElemNet/tree/master/elemnet) and renamed as _dl_regressors.py._. After that you must to run the code using the following command.

```bash
python dl_regressors.py --config_file sample/sample-run.config
```