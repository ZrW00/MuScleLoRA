code=$(echo "$1")
config=$(echo "$2")
dataset=$(echo "$3" | tr '[:upper:]' '[:lower:]')
modelName=$(echo "$4" | tr '[:upper:]' '[:lower:]')
log=$(echo "$5")

echo "nohup python -u $code --config_path $config --dataset $dataset --target_model $modelName > $log 2>&1 &"
nohup python -u $code --config_path $config --dataset $dataset --target_model $modelName > $log 2>&1 &
code $log