#!/bin/bash

pip3 install -r utils/requirements3.txt

if [ ! -d "models" ] 
then
  mkdir models
fi

if [ $CHURN_MODEL_NAME ]
then
  CHURN_MODEL_FILE="$CHURN_MODEL_NAME.pkl"

  if [ -f $CHURN_MODEL_FILE ]
  then 
    mkdir models/$CHURN_MODEL_NAME
    mv $CHURN_MODEL_FILE models/$CHURN_MODEL_NAME
  fi
fi

if [[ ! -d /home/cdsw/R ]]
then 
  mkdir -m 755 /home/cdsw/R
fi
