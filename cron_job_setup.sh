#!/bin/bash

mkdir -p "$HOME/tmp"
PIDFILE="$HOME/tmp/myprogram.pid"

if [ -e "${PIDFILE}" ] && (ps -u $(whoami) -opid= |
                           grep -P "^\s*$(cat ${PIDFILE})$" &> /dev/null); then
  echo "Already running."
  #Mail is taking too long to run
  #mailx -s "Healthcheck - CronJob Ran and detected that Lume Impact Live Demo is Still Up" "$EMAIL_ID"
  exit 99
fi

echo $$ > "${PIDFILE}"
chmod 644 "${PIDFILE}"

#mailx -s "CronJob has been started/restarted - " "$EMAIL_ID"

export EPICS_PVA_SERVER_PORT=5075
export EPICS_PVA_BROADCAST_PORT=5076
export EPICS_PVA_AUTO_ADDR_LIST=FALSE
export EPICS_PVA_ADDR_LIST="lcls-prod01:5068"
export EPICS_PVA_ADDR_LIST="${EPICS_PVA_ADDR_LIST} lcls-prod01:5063"
export EPICS_PVA_ADDR_LIST="${EPICS_PVA_ADDR_LIST} mcc-dmz mccas0.slac.stanford.edu"
export EPICS_CA_AUTO_ADDR_LIST=NO
export EPICS_CA_ADDR_LIST="lcls-prod01:5068 lcls-prod01:5063 mcc-dmz"
export EPICS_CA_REPEATER_PORT="5069"
export EPICS_CA_SERVER_PORT="5068"
export EPICS_TS_NTP_INET="134.79.48.11"
export EPICS_IOC_LOG_INET="134.79.151.21"

conda activate lume-live-dev

#These are S3DF Variables
export LCLS_LATTICE=/sdf/group/ad/beamphysics/lcls-lattice
export SCRATCH=/sdf/group/ad/beamphysics/lume-impact-live-demo/SCRATCH

cd /sdf/group/ad/beamphysics/lume-impact-live-demo/

ipython /sdf/group/ad/beamphysics/lume-impact-live-demo/lume-impact-live-demo.py -- -t "s3df"