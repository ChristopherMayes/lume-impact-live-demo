# Define base image
#FROM centos:centos7
FROM continuumio/miniconda3

RUN mkdir /app/

# Create Conda environment from the YAML filei
COPY . /app/

RUN ls /app/

WORKDIR /app/

RUN conda env create -f lume-live-dev.yml   

RUN export PATH="/opt/conda/bin/:/opt/slurm/slurm-curr/bin/:$PATH"

RUN apt-get update && apt install gfortran -y

# Override default shell and use bash:
SHELL ["conda", "run", "-n", "lume-live-dev", "/bin/bash", "-c"]

RUN source ~/.bashrc \
    && source /opt/conda/etc/profile.d/conda.sh \
    && /opt/conda/bin/activate && conda init bash \ 
    && conda activate lume-live-dev \
    && conda install -c conda-forge impact-t=*=mpi_openmpi* \
    && echo "conda activate lume-live-dev" > ~/.bashrc \
    && echo "echo \"Hello World\"" >>  ~/.bashrc \
    && conda clean -a

RUN apt install make 

RUN git clone https://github.com/impact-lbl/IMPACT-T.git

RUN cd IMPACT-T/src/ && /opt/conda/envs/lume-live-dev/bin/cmake -S . -B build -DUSE_MPI=ON \
     && /opt/conda/envs/lume-live-dev/bin/cmake --build build \
     && /opt/conda/envs/lume-live-dev/bin/cmake --build build --target install

RUN ls /usr/local/bin | grep "Impact"

RUN echo "Installing Impact-T seperately"

RUN ls -a /root/

SHELL ["/bin/bash", "-c"]

RUN ls -a ~

RUN conda info | grep -i 'base environment'

RUN echo "Check if Impactexe and Impactexe-mpi are installed"
RUN ls -ltr /opt/conda/envs/lume-live-dev/bin/ | grep "Impact"

# Activate Conda environment and check if it is working properly
RUN echo "Making sure Key Packages are installed correctly..."
RUN conda run -n lume-live-dev python -c "import impact"

RUN mkdir -p /app/output/{archive,output,plot,snapshot,log,summary}

#Convert Jupyter Notebooks to Python Files and Create Necessary Folders
RUN echo "Convert Jupyter Notebooks to Python Files and Create Necessary Folders"
RUN conda run -n lume-live-dev jupyter nbconvert --to script lume-impact-live-demo.ipynb \
 && conda run -n lume-live-dev jupyter nbconvert --to script make_dashboard.ipynb \
 && conda run -n lume-live-dev jupyter nbconvert --to script get_vcc_image.ipynb

# Setup our EPICS environment to be able to read PROD variables
ENV EPICS_PVA_SERVER_PORT 5075
ENV EPICS_PVA_BROADCAST_PORT 5076
ENV EPICS_PVA_AUTO_ADDR_LIST FALSE
ENV EPICS_PVA_ADDR_LIST "lcls-prod01:5068"
ENV EPICS_PVA_ADDR_LIST "${EPICS_PVA_ADDR_LIST} lcls-prod01:5063"
ENV EPICS_PVA_ADDR_LIST "${EPICS_PVA_ADDR_LIST} mcc-dmz mccas0.slac.stanford.edu"
ENV EPICS_CA_AUTO_ADDR_LIST NO
ENV EPICS_CA_ADDR_LIST "lcls-prod01:5068 lcls-prod01:5063 mcc-dmz"
ENV EPICS_CA_REPEATER_PORT "5069"
ENV EPICS_CA_SERVER_PORT "5068"
ENV EPICS_TS_NTP_INET "134.79.48.11"
ENV EPICS_IOC_LOG_INET "134.79.151.21"

ENV OMPI_ALLOW_RUN_AS_ROOT 1
ENV OMPI_ALLOW_RUN_AS_ROOT_CONFIRM 1



# Python program to run in the container
#ENTRYPOINT ["conda", "run", "-n", "lume-live-dev", "ipython", "/app/lume-impact-live-demo.py", "--", "-t", "'singularity'"]