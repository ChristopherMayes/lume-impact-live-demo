# lume-impact-live-demo
Demonstration of LUME-Impact running a live model fetching data from the LCLS EPICS network



# Setup

Make the `lume-live-dev` environment:
`conda env create --file lume-live-dev.yml`

Convert notebooks to `.py` files:
```bash
./build.bash
```

EPICS:
```bash
ssh -fN -L 24666:lcls-prod01.slac.stanford.edu:5068 centos7.slac.stanford.edu
export EPICS_CA_NAME_SERVERS=localhost:24666
```

Test with: `caget KLYS:LI22:11:KPHR`


# Run

For example, LCLS on SDF with default parameters:
```
ipython lume-impact-live-demo.py
```

LCLS on S3DF with default parameters:
```
ipython lume-impact-live-demo.py -- -t "s3df"
```

All Parameters to run lume-impact-live-demo.py file -
```
-d Debug=True/False
-l Live=True/False
-v USE_VCC=True/False
-m Please pass the model name here (sc_inj/facet/lcls)
-h Please pass the host name here (sdf/singularity)

Defaults -
debug=False
use_vcc=True
live=True
model=sc_inj
host=sdf
```

## TOML properties file

Running the simulation requires definition of the following variables within a toml file:


| Variable                  | Description                          |
|---------------------------|--------------------------------------|
| host                      | Host of Impact sim                   |
| config_file               | Impact configuration file            |
| distgen_input_file        | Input to distgen generation          |
| workdir                   | Working directory of simulation run  |
| summary_output_dir        | Output directory for summary files   |
| plot_output_dir           | Output directory of plot files       |
| archive_dir               | Output directory for archive files   |
| shapshot_dir              | Output directory for snapshot files  |
| distgen_laser_file        | File for generating distgen input    |
| num_procs                 | Number of processes to use           |
| mpi_run                   | command for running mpi              |

Running on SDF additionally requires:

| Variable                  | Description                          |
|---------------------------|--------------------------------------|
| impact_command            | Command for Impact execution         |
| impact_command_mpi        | Command for running MPI              |

Default configurations are given in the example environment files packaged with this repository.
