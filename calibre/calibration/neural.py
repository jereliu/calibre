"""Function Computing Kernel Scoring Rules

#### References

[1]:    Gneiting, T., Raftery, A.E.: Strictly proper scoring rules, prediction, and estimation.
        _J. Am. Stat. Assoc. 102, 359–378_. 2007.
[2]:    Gneiting, T., Balabdaoui, F., and Raftery, A. E. Probabilistic Forecasts,
        Calibration and Sharpness, _Journal of the Royal Statistical Society,Ser.B_. 2007.
"""

import tensorflow as tf


#def kernel_score(g=tf.abs):
