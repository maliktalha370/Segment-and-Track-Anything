# Use the base image
FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-devel
ARG DEBIAN_FRONTEND=noninteractive


# Install necessary packages
RUN apt-get update && apt-get install -y ffmpeg libsm6 libxext6 nano git wget

# Clone the repository
RUN git clone https://github.com/maliktalha370/Segment-and-Track-Anything.git

# Install sam package
WORKDIR ./Segment-and-Track-Anything/sam/
RUN pip install -e .
WORKDIR /Segment-and-Track-Anything/

# Install GroundingDINO package
RUN pip install -e git+https://github.com/IDEA-Research/GroundingDINO.git@main#egg=GroundingDINO

# Install additional dependencies
RUN pip install numpy opencv-python pycocotools matplotlib Pillow scikit-image gdown schedule pydrive gradio

# Install Pytorch Correlation
RUN git clone https://github.com/ClementPinard/Pytorch-Correlation-extension.git
WORKDIR ./Pytorch-Correlation-extension
RUN python setup.py install
WORKDIR /Segment-and-Track-Anything

# Remove existing 'ckpt' directory and create a new one
RUN rm -r ./ckpt
RUN mkdir ./ckpt


# Download Service key json
RUN gdown --id '1B3uL0VyM2uT0yufk7KxTLVAf8RuN5ywP' --output genuine-park-393209-c6843b80e8d2.json

# Download aot-ckpt
RUN gdown --id '1QoChMkTVxdYZ_eBlZhK2acq9KMQZccPJ' --output ./ckpt/R50_DeAOTL_PRE_YTB_DAV.pth

# Download sam-ckpt
RUN wget -P ./ckpt https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth

# Download grounding-dino ckpt
RUN wget -P ./ckpt https://huggingface.co/ShilongLiu/GroundingDINO/resolve/main/groundingdino_swint_ogc.pth

# Copy the Python script
WORKDIR /Segment-and-Track-Anything

# Run the Python script
CMD python video_scheduler.py
