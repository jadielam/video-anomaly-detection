# To Build:
# docker build -t training -f Dockerfile .

# To run:
# nvidia-docker run -it training

FROM nvidia/cuda:8.0-cudnn7-devel-ubuntu16.04 

RUN echo "deb http://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1604/x86_64 /" > /etc/apt/sources.list.d/nvidia-ml.list

RUN apt-get update && apt-get install -y --allow-downgrades --no-install-recommends \
         build-essential \
         cmake \
         git \
         curl \
         vim \
         ca-certificates \
         libnccl2=2.0.5-2+cuda8.0 \
         libnccl-dev=2.0.5-2+cuda8.0 \
         libjpeg-dev \
         libpng-dev &&\
     rm -rf /var/lib/apt/lists/*


ENV PYTHON_VERSION=3.6
RUN curl -o ~/miniconda.sh -O https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh  && \
     chmod +x ~/miniconda.sh && \
     ~/miniconda.sh -b -p /opt/conda && \     
     rm ~/miniconda.sh && \
#     /opt/conda/bin/conda install conda-build && \
     /opt/conda/bin/conda create -y --name pytorch-py$PYTHON_VERSION python=$PYTHON_VERSION numpy pyyaml scipy ipython notebook mkl&& \
     /opt/conda/bin/conda clean -ya 
ENV PATH /opt/conda/envs/pytorch-py$PYTHON_VERSION/bin:$PATH
RUN /opt/conda/bin/conda install --name pytorch-py$PYTHON_VERSION -c soumith cuda80
RUN pip install mkl-devel

# Installing pytorch
WORKDIR /opt
RUN git clone https://github.com/pytorch/pytorch
WORKDIR /opt/pytorch
RUN git submodule update --init
RUN TORCH_CUDA_ARCH_LIST="3.5 5.2 6.0 6.1+PTX" TORCH_NVCC_FLAGS="-Xfatbin -compress-all" \
    CMAKE_PREFIX_PATH="$(dirname $(which conda))/../" \
    pip install -v .

# Installing pytorch vision
RUN git clone https://github.com/pytorch/vision.git && cd vision && pip install -v .

# Installing pytorch text
RUN pip install torchvision
RUN pip install pandas
RUN pip install scikit-learn
RUN pip install imageio

#WORKDIR /workspace
#RUN chmod -R a+w /workspace

# Add code to folder
RUN mkdir -p /src
ADD code/ /src/deep_learning_text
ENV PYTHONPATH='$PYTHONPATH:/src/deep_learning_text'

#EXPOSE 8888
#ENTRYPOINT ["/bin/bash", "-c", "jupyter notebook --ip='*' --allow-root --no-browser --port=8888"]
ENTRYPOINT ["/bin/bash"]
