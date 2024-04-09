* Embodided Conversation Agent (ECA) NLP Research Prototype (for AI-blackbox)
  : Crystal Island (CI)

* RLEngine_DialogAgent.py
: default approach of dialog agent is set to 1 (S-BERT based question matching)

For approach 2 (response matching) or 3 (response index prediction)
: Download the language models from the google drive: https://drive.google.com/drive/folders/1jvz6oEsds_oeeiSSoLQJGLvoXm1rccHY?usp=sharing and place them in the directory name: './model'
   - model/model_ci_a2
   - model/model_ci_a3

==============================================================================================
5. Development evnrionment
    * OS: CentOS Linux 7 (Core)
    * miniconda install: https://docs.conda.io/projects/conda/en/latest/user-guide/install/
    * packages (see the installation guide at the bottom):
        python=3.9 
        numpy=1.19.2         
        tensorflow-gpu=2.4.1
        cudatoolkit=10.1
        cudnn=7.6.5 
        pytorch=1.2.1
        transformers=4.23.1
        sentence-transformers=2.2.2
        datasets=2.5.1
        nvidia-ml-py3=7.352.0
        nltk=3.7
        scikit-learn=1.1.1
        bs4=4.11.1
        
6. Installation guide with conda:
    1) Linux
        $ conda create -n nlp python=3.9 numpy=1.19
        $ conda install tensorflow-gpu=2.4.1 cudatoolkit=10.1 cudnn=7.6.5 
             # tensorflow gpu check: $ python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"   
        $ conda install pytorch=1.2.1 torchvision=0.2.2 -c pytorch 
             # torch gpu check: $ python -c "import torch; print(torch.cuda.is_available())"
    
    2) mac (M1)
        #install miniconda: https://docs.conda.io/projects/conda/en/latest/user-guide/install/
             # download miniconda > bash Miniconda3-latest-MacOSX-arm64.sh     
        $ conda create -name nlp python=3.9 numpy=1.19 
        # Tensorflow 
            # version compatibility: https://developer.apple.com/metal/tensorflow-plugin/    
            # https://www.mrdbourke.com/setup-apple-m1-pro-and-m1-max-for-machine-learning-and-data-science/
        $ conda install -c apple tensorflow-deps 
        $ python -m pip install tensorflow-macos==2.8 
        $ python -m pip install tensorflow-metal==0.4 
             # If TypeError: Descriptors cannot not be created directly.  --> downgrade protobuf: $ pip install protobuf==3.20
             # pip install numpy==1.19 (check if tensorflow gpu still working)
        # pytorch for M1 : refer to https://www.youtube.com/watch?v=VEDy-c5Sk8Y 
        $ conda install -c pytorch-nightly pytorch 
            # torch mps (not gpu) check: $ python -c "import torch; print(getattr(torch, 'has_mps', False))"
            # usage in code: device= "mps" if getattr(torch, 'has_mps', False) else "gpu" if torch.cuda_is_avilable() else "cpu"
        
        # install brew: https://phoenixnap.com/kb/install-homebrew-on-mac
        
    3) Common    
        $ conda install -c conda-forge ipywidgets
        $ conda install -c anaconda scikit-learn
        $ conda install -c conda-forge nltk
        $ conda install -c conda-forge bs4     
        $ conda install -c conda-forge spacy
        $ conda install -c conda-forge sentence-transformers
        $ conda install -c conda-forge jupyterlab (if needed)
        
            # To register jupyter kernel : $ python -m ipykernel install --user --name pytorch --display-name "Python 3.9 (pytorch)"

        $ pip install transformers datasets nvidia-ml-py3
        $ pip install pyitlib  (for conditional entropy)
    
    4) Trouble shooting
        # Error with dill._dill.stack: $ pip install dill==0.3.4 (not available from 0.3.5)

 


