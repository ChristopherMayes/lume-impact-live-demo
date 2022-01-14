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