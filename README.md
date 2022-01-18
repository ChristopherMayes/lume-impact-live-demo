# lume-impact-live-demo
Demonstration of LUME-Impact running a live model fetching data from the LCLS EPICS network



# Setup

Make the `lume-live` environment:
`conda env create --file environment.yml`

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

For example, LCLS on SDF:
```
source example_sdf_lcls.env # source appropriate environment:
`./run.bash`
```

## TOML properties file

Running the simulation requires definition of the following variables within a toml file:

host=sdf
config_file=/sdf/group/ard/impact/facet2-lattice/impact/models/f2e_inj/v0/ImpactT.yaml
distgen_input_file=/sdf/group/ard/impact/facet2-lattice/distgen/models/f2e_inj/vcc_image/distgen.yaml
workdir=/sdf/group/ard/impact/output
summary_output_dir=/sdf/group/ard/impact/output/summary
plot_output_dir=/sdf/group/ard/impact/output/plot
archive_dir=/sdf/group/ard/impact/output/archive
snapshot_dir=/sdf/group/ard/impact/output/snapshot
distgen_laser_file=/sdf/group/ard/impact/output/distgen_laser.txt
num_procs=64
mpi_run_cmd='salloc --partition ard -N 1 -n {n} /sdf/sw/gcc-4.8.5/openmpi-4.0.4/bin/mpirun -n {n} {command_mpi}'
impact_command=/sdf/home/c/cmayes/impact/bin/ImpactTexe
impact_command_mpi='/sdf/home/c/cmayes/impact/bin/ImpactTexe-mpi'


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