This is the source code for the paper: 

Relation Extraction: Perspective from Convolutional Neural Networks

Thien Huu Nguyen and Ralph Grishman, in Proceedings of NAACL Workshop on Vector Space Modeling for NLP, Denver, Colorado, June, 2015.

----------------

There are two steps to run this code:

* Preprocessing: using file ```dist_process_data.py```

You will need to have the ACE 2005 data set in the format required by this file. We cannot include the data in this release due to licence issues.

* Train and test the model: using file ```dist_conv_net_sentence_oneFold.py```

This step takes the output file in step 1.

THE CODE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND.