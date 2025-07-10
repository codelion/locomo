# sets necessary environment variables
source scripts/env.sh

# Evaluate Gemini Pro
python3 task_eval/evaluate_qa.py \
    --data-file $DATA_FILE_PATH --out-file $OUT_DIR/$QA_OUTPUT_FILE \
    --model gemini-2.5-flash-lite-preview-06-17 --batch-size 20
