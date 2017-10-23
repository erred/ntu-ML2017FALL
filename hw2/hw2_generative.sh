#!/bin/sh

# $1: raw data (train.csv)
# $2: test data (test.csv)
# $3: provided train feature (X_train)
# $4: provided train label (Y_train)
# $5: provided test feature (X_test)
# $6: prediction.csv
# python3 gen.py "test" "$3" "$5" "$4" "gen.model" "$6"
python3 genta.py --infer --train_data_path "$3" --train_label_path "$4" --test_data_path "$5" --output_file "$6"
