#!/bin/bash

SECONDS=0

mkdir -p logs
ABLATE_PARAM="arrival_rate"

# ablating sigma
if [ "$ABLATE_PARAM" == "sigma" ]; then
    for sigma in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0
    do
        python llm_scheduling_iclr/vllm/core/run_synthetic.py --sigma ${sigma} > logs/log_sigma_${sigma}.txt &
    done
    wait
fi

# ablating token_gen_rate
if [ "$ABLATE_PARAM" == "token_gen_rate" ]; then
    for token_gen_rate in 1 2 5 10 15 20 25 30
    do
        python llm_scheduling_iclr/vllm/core/run_synthetic.py --token_gen_rate ${token_gen_rate} > logs/log_tokengenrate_${token_gen_rate}.txt &
    done
    wait
fi 

# ablating arrival_rate
if [ "$ABLATE_PARAM" == "arrival_rate" ]; then
    for arrival_rate in  0.6 0.7 0.8 0.9 0.95 1.0
    do
        python llm_scheduling_iclr/vllm/core/run_synthetic.py --arrival_rate ${arrival_rate} --token_gen_rate 40 > logs/log_arrival_rate_${arrival_rate}.txt &
    done
    wait
fi

echo "Total runtime: $SECONDS seconds"

