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



`./run.bash`

## Environment variables

Running the simulation requires definition of the following variables:

| Variable                  | Description                          |
|---------------------------|--------------------------------------|
| IMPACT_HOST               | Host of Impact sim                   |
| IMPACT_CONFIG_FILE        | Impact configuration file            |
| IMPACT_DISTGEN_INPUT_FILE | Input to distgen generation          |
| IMPACT_WORKDIR            | Working directory of simulation run  |
| IMPACT_SUMMARY_OUTPUT_DIR | Output directory for summary files   |
| IMPACT_PLOT_OUTPUT_DIR    | Output directory of plot files       |
| IMPACT_ARCHIVE_DIR        | Output directory for archive files   |
| IMPACT_SNAPSHOT_DIR       | Output directory for snapshot files  |
| IMPACT_DISTGEN_LASER_FILE | File for generating distgen input    |
| IMPACT_NUM_PROCS          | Number of processes to use           |
| IMPACT_COMMAND            | Command for Impact execution         |
| IMPACT_MPI_RUN_CMD        | Command for running MPI              |

Running on SDF additionally requires:

| Variable                  | Description                          |
|---------------------------|--------------------------------------|
| IMPACT_COMMAND_MPI        | MPI for running impact command
| IMPACT_COMMAND            | Command for Impact execution         |


Default configurations are given in the example environment files packaged with this repository.