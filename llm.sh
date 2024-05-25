dataset=$(echo "$1" | tr '[:upper:]' '[:lower:]')
modelName=$(echo "$2" | tr '[:upper:]' '[:lower:]')
way=$(echo "$3" | tr '[:upper:]' '[:lower:]')
start=$4
end=$5

if [ $# -ge 6 ]; then
    notation=$6
    echo $notation
else 
    notation=""
fi

vanillaConfigs=("./llmConfigs/Badnets4NoDefense.json" "./llmConfigs/AddsentsNoDefense.json" "./llmConfigs/StyleNoDefense.json" "./llmConfigs/HiddenKillerNoDefense.json")
vanillaLogs=("./llmLogs/Ablation+$modelName+Badnets4+Vanilla+$dataset+$notation.log" "./llmLogs/Ablation+$modelName+Addsents+Vanilla+$dataset+$notation.log" "./llmLogs/Ablation+$modelName+Style+Vanilla+$dataset+$notation.log" "./llmLogs/Ablation+$modelName+HiddenKiller+Vanilla+$dataset+$notation.log")


mslrConfigs=("./llmConfigs/Badnets4OnlyMSLRConfig.json" "./llmConfigs/AddsentsOnlyMSLRConfig.json" "./llmConfigs/StyleOnlyMSLRConfig.json" "./llmConfigs/HiddenKillerOnlyMSLRConfig.json")
mslrLogs=("./llmLogs/Ablation+$modelName+Badnets4+OnlyMSLR+$dataset+$notation.log" "./llmLogs/Ablation+$modelName+Addsents+OnlyMSLR+$dataset+$notation.log" "./llmLogs/Ablation+$modelName+Style+OnlyMSLR+$dataset+$notation.log" "./llmLogs/Ablation+$modelName+HiddenKiller+OnlyMSLR+$dataset+$notation.log")

loraConfigs=("./llmConfigs/Badnets4OnlyLoRAConfig.json" "./llmConfigs/AddsentsOnlyLoRAConfig.json" "./llmConfigs/StyleOnlyLoRAConfig.json" "./llmConfigs/HiddenKillerOnlyLoRAConfig.json")
loraLogs=("./llmLogs/Ablation+$modelName+Badnets4+OnlyLoRA+$dataset+$notation.log" "./llmLogs/Ablation+$modelName+Addsents+OnlyLoRA+$dataset+$notation.log" "./llmLogs/Ablation+$modelName+Style+OnlyLoRA+$dataset+$notation.log" "./llmLogs/Ablation+$modelName+HiddenKiller+OnlyLoRA+$dataset+$notation.log")

prefixConfigs=("./llmConfigs/Badnets4OnlyPrefixConfig.json" "./llmConfigs/AddsentsOnlyPrefixConfig.json" "./llmConfigs/StyleOnlyPrefixConfig.json" "./llmConfigs/HiddenKillerOnlyPrefixConfig.json")
prefixLogs=("./llmLogs/Ablation+$modelName+Badnets4+OnlyPrefix+$dataset+$notation.log" "./llmLogs/Ablation+$modelName+Addsents+OnlyPrefix+$dataset+$notation.log" "./llmLogs/Ablation+$modelName+Style+OnlyPrefix+$dataset+$notation.log" "./llmLogs/Ablation+$modelName+HiddenKiller+OnlyPrefix+$dataset+$notation.log")

gaLoraConfigs=("./llmConfigs/Badnets4GA+LoRAConfig.json" "./llmConfigs/AddsentsGA+LoRAConfig.json" "./llmConfigs/StyleGA+LoRAConfig.json" "./llmConfigs/HiddenKillerGA+LoRAConfig.json")
gaLoraLogs=("./llmLogs/Ablation+$modelName+Badnets4+GA+LoRA+$dataset+RandomRef+$notation.log" "./llmLogs/Ablation+$modelName+Addsents+GA+LoRA+$dataset+RandomRef+$notation.log" "./llmLogs/Ablation$modelName+Style+GA+LoRA+$dataset+RandomRef+$notation.log" "./llmLogs/Ablation+$modelName+HiddenKiller+GA+LoRA+$dataset+RandomRef+$notation.log")

gaLoraMSLRConfigs=("./llmConfigs/Badnets4GA+LoRA+MSLRConfig.json" "./llmConfigs/AddsentsGA+LoRA+MSLRConfig.json" "./llmConfigs/StyleGA+LoRA+MSLRConfig.json" "./llmConfigs/HiddenKillerGA+LoRA+MSLRConfig.json")
gaLoraMSLRLogs=("./llmLogs/Ablation+$modelName+Badnets4+GA+LoRA+MSLR+$dataset+RandomRef+$notation.log" "./llmLogs/Ablation+$modelName+Addsents+GA+LoRA+MSLR+$dataset+RandomRef+$notation.log" "./llmLogs/Ablation+$modelName+Style+GA+LoRA+MSLR+$dataset+RandomRef+$notation.log" "./llmLogs/Ablation+$modelName+HiddenKiller+GA+LoRA+MSLR+$dataset+RandomRef+$notation.log")

if [ "$way" == 'vanilla' ]; then
    configs=("${vanillaConfigs[@]}")
    logs=("${vanillaLogs[@]}")
elif [ "$way" == 'prefix' ]; then
    configs=("${prefixConfigs[@]}")
    logs=("${prefixLogs[@]}")
elif [ "$way" == 'mslr' ]; then
    configs=("${mslrConfigs[@]}")
    logs=("${mslrLogs[@]}")
elif [ "$way" == 'lora' ]; then
    configs=("${loraConfigs[@]}")
    logs=("${loraLogs[@]}")
elif [ "$way" == 'ga+lora' ]; then
    configs=("${gaLoraConfigs[@]}")
    logs=("${gaLoraLogs[@]}")
elif [ "$way" == 'ga+lora+mslr' ]; then
    configs=("${gaLoraMSLRConfigs[@]}")
    logs=("${gaLoraMSLRLogs[@]}")
else
    :
fi
echo $configs
echo $logs

configLength=${#configs[@]}
echo $configLength

echo "start = $start, end = $end"

for ((i=$start; i<$end; i++))
do
    config="${configs[$i]}"
    log="${logs[$i]}"
    echo "Config Path: $config, Log Path: $log"
    echo "nohup python -u llmDefense.py --config_path $config --dataset $dataset --target_model $modelName > $log 2>&1 &"
    nohup python -u llmDefense.py --config_path $config --dataset $dataset --target_model $modelName > $log 2>&1 &
    echo $log
    sleep 90s
done

