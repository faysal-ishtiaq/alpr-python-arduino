# alpr-python-arduino

The web cam is always looking for a license plate in the feed. When it finds one reads the license number and check it with the program database. If matches it sends a signal to the arduino via serial communication. An LED lits up. After four seconds it repeats the whole process. And a signal to turn off LED is sent if the license doesn't match.


The program uses OpenCV, KNN Algorithm


![img_20170108_020349](https://cloud.githubusercontent.com/assets/5615972/21744793/20ce18b2-d547-11e6-9fe0-4d22a498b3ea.jpg)
