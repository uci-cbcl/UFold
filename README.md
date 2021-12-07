<img src='https://github.com/uci-cbcl/UFold/blob/main/ufold/UFold_logonew1.png' width=300 height=200>


## UFold: Fast and Accurate RNA Secondary Structure Prediction with Deep Learning

For many RNA molecules, the secondary structure is essential for the correct function of the RNA. Predicting RNA secondary structure from nucleotide sequences is a long-standing problem in genomics, but the prediction performance has reached a plateau over time. Traditional RNA secondary structure prediction algorithms are primarily based on thermodynamic models through free energy minimization, which imposes strong prior assumptions and is slow to run. Here we propose a deep learning-based method, called UFold, for RNA secondary structure prediction, trained directly on annotated data and base-pairing rules. Ufold proposes a novel image-like representation of RNA sequences, which can be efficiently processed by Fully Convolutional Networks (FCNs). UFold improves upon previous models, with approximately 10~30% improvement over traditional thermodynamic models and up to 27% improvement over other learning-based methods in terms of base-pair prediction accuracy on an RNA structure prediction benchmark dataset. UFold is also fast with an inference time of about 160ms per sequence up to 1600bp in length.

## Prerequisites
--python >= 3.6.6

--torch >= 1.4 with cudnn >=10.0

--[munch](https://pypi.org/project/munch/2.0.2/)

--[subprocess](https://docs.python.org/3/library/subprocess.html)

--[collections](https://docs.python.org/2.7/library/collections.html#)

## Installation 
Clone the repository.

```
git clone https://github.com/uci-cbcl/UFold.git
```

Navigate to the root of this repo and setup the conda environment.

```
conda env create -f UFold.yml
```

Activate conda environment.

```
conda activate UFold
``` 

### Check if the installation succeed. 
Please check the [predicting section](https://github.com/uci-cbcl/UFold#evaluating--predicting) to further validate whether the installation is succeeded and the software works as expected.  

## Pre-trained models 

Pre-trained models are deposited in our [drive](https://drive.google.com/drive/folders/1Sq7MVgFOshGPlumRE_hpNXadvhJKaryi?usp=sharing). Please download them and put them into [models folder](https://github.com/uci-cbcl/UFold/tree/main/models).

## Usage

### Recommended
We recommend users use our [UFold webserver](https://ufold.ics.uci.edu), which is user-friendly and easy to use. Everyone could upload or typein your own candidate RNA sequence in our web without further installation, our backend server will calculate and give the prediction result to the user. User can choose to download the predict ct file result as well as visualize them online directly.

### Data generator
You can put their bpseq formatted files in their own directory and specify it in this process by running:
<pre><code>$ python process_data_newdataset.py your_own_directory_containing_bpseq_files
</code></pre> 
After that you will get a pickle file format, which is compatible with our model. Then put the data into [data folder](https://github.com/uci-cbcl/UFold/tree/main/data).

### Training
You can train our model using pre-defined data, or use our customed data generate script to generate your own data(Mentioned before). After that, you can run the model training script:
<pre><code>$ python ufold_train.py --train_files dataset_A dataset_B 
--train_files: optinal parameter, default is all the datasets mentioned in the paper.
</code></pre> 
Noted that this script will include all the data for training as default.

### Evaluating & Predicting
We provide test script for users to evaluate the prediction result and predict their own sequence using pre-trained model. Users can predict their own sequences and get ct file as output from the webserver. The script is for evaluating the results.
<pre><code>$ python ufold_test.py --test_files TS2 
--test_files: optional parameter, set the test set name from one of test sets(['ArchiveII','TS0','bpnew','TS1','TS2','TS3']).
</code></pre> 
After running the above example, you will genarate the example output as the following shows:   

<img src='https://github.com/uci-cbcl/UFold/blob/main/ufold/UFold_example_output.png' width=390 height=130>  

If the above output pops up, then you have succeeded installed our software.  

We have also provided an offline version of our prediction tool for users to directly predict RNA secondary structure using their own fasta sequences. All they need to do is put their fasta file (named input.txt) into data folder. And then run the following codes:  
<pre><code>$ python ufold_predict.py --nc False
--nc: optional parameter, whether to predict non-canonical or nor, default is False.
</code></pre>  
After running the above command, you will get the output ct file,bpseq file, and figures in the results folder.

## Citation

If you use our tool, please cite our work: 

[UFold: Fast and Accurate RNA Secondary Structure Prediction with Deep Learning](https://academic.oup.com/nar/advance-article/doi/10.1093/nar/gkab1074/6430845?searchresult=1)

Laiyi Fu*, Yingxin Cao*, Jie Wu, Qinke Peng, Qing Nie, Xiaohui Xie, UFold: fast and accurate RNA secondary structure prediction with deep learning, Nucleic Acids Research, 2021;, gkab1074, https://doi.org/10.1093/nar/gkab1074



