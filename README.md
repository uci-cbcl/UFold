# UFold

## Fast and Accurate RNA Secondary Structure Prediction with Deep Learning

For many RNA molecules, the secondary structure is essential for the correct function of the RNA. Predicting RNA secondary structure from nucleotide sequences is a long-standing problem in genomics, but the prediction performance has reached a plateau over time. Traditional RNA secondary structure prediction algorithms are primarily based on thermodynamic models through free energy minimization, which imposes strong prior assumptions and is slow to run. Here we propose a deep learning-based method, called UFold, for RNA secondary structure prediction, trained directly on annotated data and base-pairing rules. Ufold proposes a novel image-like representation of RNA sequences, which can be efficiently processed by Fully Convolutional Networks (FCNs). UFold improves upon previous models, with approximately 10~30% improvement over traditional thermodynamic models and up to 27% improvement over other learning-based methods in terms of base-pair prediction accuracy on an RNA structure prediction benchmark dataset. UFold is also fast with an inference time of about 160ms per sequence up to 1600bp in length.

## Prerequisites
--python >= 3.6

--torch >= 1.4 with cudnn >=10.0

--[munch](https://pypi.org/project/munch/2.0.2/)


## Usage

### Recommended
We recommend users use our [UFold webserver](https://ufold.ics.uci.edu), which is user-friendly and easy to use. Everyone could upload or typein your own candidate RNA sequence in our web without further installation, our backend server will calculate and give the prediction result to the user. User can choose to download the predict ct file result as well as visualize them online directly.

### Training
You can train our model using pre-defined data, or use our customed data generate script to generate your own data. After that, you can run the model training script:
<pre><code>$ python ufold_train.py
</code></pre> 

### Evaluating & Predicting
We provide test script for users to evaluate the prediction result and predict their own sequence using pre-trained model. After generating a suitable format input data, users can predict their own sequences and get ct file as output.
<pre><code>$ python ufold_test.py
</code></pre> 

## Citation

If you use our tool, please cite our work: 

UFold: Fast and Accurate RNA Secondary Structure Prediction with Deep Learning

Laiyi Fu*, Yingxin Cao*, Jie Wu, Qinke Peng, Qing Nie, Xiaohui Xie

bioRxiv 2020.08.17.254896; doi: https://doi.org/10.1101/2020.08.17.254896
