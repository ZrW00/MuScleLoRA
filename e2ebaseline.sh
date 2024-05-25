dataset=$(echo "$1" | tr '[:upper:]' '[:lower:]')
modelName=$(echo "$2" | tr '[:upper:]' '[:lower:]')
defender=$(echo "$3" | tr '[:upper:]' '[:lower:]')
start=$4
end=$5

onionConfig="./baselineConfigs/Onion.json"
onionLLMConfig="./baselineConfigs/Onionllm.json"
bkiConfig="./baselineConfigs/BKI.json"
cubeConfig="./baselineConfigs/CUBE.json"
stripConfig="./baselineConfigs/STRIP.json"
stripLLMConfig="./baselineConfigs/STRIPllm.json"
rapConfig="./baselineConfigs/RAP.json"

if [ "$defender" == 'onion' ]; then
    config=$onionConfig
elif [ "$defender" == 'onionllm' ]; then
    config=$onionLLMConfig
elif [ "$defender" == 'bki' ]; then
    config=$bkiConfig
elif [ "$defender" == 'cube' ]; then
    config=$cubeConfig  
elif [ "$defender" == 'strip' ]; then
    config=$stripConfig  
elif [ "$defender" == 'stripllm' ]; then
    config=$stripLLMConfig  
elif [ "$defender" == 'rap' ]; then
    config=$rapConfig  
else
    :
fi
poisoners=("badnets4" "addsents" "style" "hiddenkiller")

echo "start = $start, end = $end"
for ((i=$start; i<$end; i++))
do
    poisoner="${poisoners[$i]}"
    log="./baselineLogs/Ablaion+$modelName+$dataset+$defender+$poisoner.log"
    echo "Config Path: $config, Log Path: $log"
    echo "nohup python -u e2ebaselineDefense.py --config_path $config --dataset $dataset --target_model $modelName --poisoner $poisoner > $log 2>&1 &"
    nohup python -u e2ebaselineDefense.py --config_path $config --dataset $dataset --target_model $modelName --poisoner $poisoner > $log 2>&1 &
    echo $log
    sleep 30s
done

