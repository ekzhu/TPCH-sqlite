#!/bin/bash

# Set ~/.sqliterc to use the TPC-H database
echo ".timer on" >> ~/.sqliterc

# Run the estimates for the GPT-4 model with 17 examples
for i in {2..18}
do
    python predict.py --db-path TPC-H.db --query-files tpc-queries/sqlite_tpc/*.sql  --timeout 30 --runtime-cache runtimes.json --output estimates-gpt-4-$i.json --model gpt-4 --num-examples $i
done
