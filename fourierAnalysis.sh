dataset=$(echo "$1" | tr '[:upper:]' '[:lower:]')
modelName=$(echo "$2" | tr '[:upper:]' '[:lower:]')
way=$(echo "$3" | tr '[:upper:]' '[:lower:]')
start=$4
end=$5

if [ $# -ge 6 ]; then
    poison_rate=$6
    echo "poison_rate = $poison_rate"
else 
    poison_rate=0.1
fi

if [ $# -ge 7 ]; then
    notation=$7
    echo $notation
else 
    notation=""
fi


vanillaConfigs=("./frequencyConfigs/Badnets4NoDefense.json" "./frequencyConfigs/AddsentsNoDefense.json" "./frequencyConfigs/StyleNoDefense.json" "./frequencyConfigs/HiddenKillerNoDefense.json")
vanillaLogs=("./frequencyLogs/Ablation+$modelName+Badnets4+Vanilla+$dataset+$notation.log" "./frequencyLogs/Ablation+$modelName+Addsents+Vanilla+$dataset+$notation.log" "./frequencyLogs/Ablation+$modelName+Style+Vanilla+$dataset+$notation.log" "./frequencyLogs/Ablation+$modelName+HiddenKiller+Vanilla+$dataset+$notation.log")

mslrConfigs=("./frequencyConfigs/Badnets4OnlyMSLRFrequency.json" "./frequencyConfigs/AddsentsOnlyMSLRFrequency.json" "./frequencyConfigs/StyleOnlyMSLRFrequency.json" "./frequencyConfigs/HiddenKillerOnlyMSLRFrequency.json")
mslrLogs=("./frequencyLogs/Ablation+$modelName+Badnets4+OnlyMSLR+$dataset+$notation.log" "./frequencyLogs/Ablation+$modelName+Addsents+OnlyMSLR+$dataset+$notation.log" "./frequencyLogs/Ablation+$modelName+Style+OnlyMSLR+$dataset+$notation.log" "./frequencyLogs/Ablation+$modelName+HiddenKiller+OnlyMSLR+$dataset+$notation.log")

loraConfigs=("./frequencyConfigs/Badnets4OnlyLoRAFrequency.json" "./frequencyConfigs/AddsentsOnlyLoRAFrequency.json" "./frequencyConfigs/StyleOnlyLoRAFrequency.json" "./frequencyConfigs/HiddenKillerOnlyLoRAFrequency.json")
loraLogs=("./frequencyLogs/Ablation+$modelName+Badnets4+OnlyLoRA+$dataset+$notation.log" "./frequencyLogs/Ablation+$modelName+Addsents+OnlyLoRA+$dataset+$notation.log" "./frequencyLogs/Ablation+$modelName+Style+OnlyLoRA+$dataset+$notation.log" "./frequencyLogs/Ablation+$modelName+HiddenKiller+OnlyLoRA+$dataset+$notation.log")

gaLoraConfigs=("./frequencyConfigs/Badnets4GA+LoRAFrequency.json" "./frequencyConfigs/AddsentsGA+LoRAFrequency.json" "./frequencyConfigs/StyleGA+LoRAFrequency.json" "./frequencyConfigs/HiddenKillerGA+LoRAFrequency.json")
gaLoraLogs=("./frequencyLogs/Ablation+$modelName+Badnets4+GA+LoRA+$dataset+RandomRef+$notation.log" "./frequencyLogs/Ablation+$modelName+Addsents+GA+LoRA+$dataset+RandomRef+$notation.log" "./frequencyLogs/Ablation$modelName+Style+GA+LoRA+$dataset+RandomRef+$notation.log" "./frequencyLogs/Ablation+$modelName+HiddenKiller+GA+LoRA+$dataset+RandomRef+$notation.log")

gaLoraMSLRConfigs=("./frequencyConfigs/Badnets4GA+LoRA+MSLRFrequency.json" "./frequencyConfigs/AddsentsGA+LoRA+MSLRFrequency.json" "./frequencyConfigs/StyleGA+LoRA+MSLRFrequency.json" "./frequencyConfigs/HiddenKillerGA+LoRA+MSLRFrequency.json")
gaLoraMSLRLogs=("./frequencyLogs/Ablation+$modelName+Badnets4+GA+LoRA+MSLR+$dataset+RandomRef+$notation.log" "./frequencyLogs/Ablation+$modelName+Addsents+GA+LoRA+MSLR+$dataset+RandomRef+$notation.log" "./frequencyLogs/Ablation+$modelName+Style+GA+LoRA+MSLR+$dataset+RandomRef+$notation.log" "./frequencyLogs/Ablation+$modelName+HiddenKiller+GA+LoRA+MSLR+$dataset+RandomRef+$notation.log")

adapterConfigs=("./frequencyConfigs/Badnets4OnlyAdapterConfig.json" "./frequencyConfigs/AddsentsOnlyAdapterConfig.json" "./frequencyConfigs/StyleOnlyAdapterConfig.json" "./frequencyConfigs/HiddenKillerOnlyAdapterConfig.json")
adapterLogs=("./frequencyLogs/Ablation+$modelName+Badnets4+OnlyAdapter+$dataset+$notation.log" "./frequencyLogs/Ablation+$modelName+Addsents+OnlyAdapter+$dataset+$notation.log" "./frequencyLogs/Ablation+$modelName+Style+OnlyAdapter+$dataset+$notation.log" "./frequencyLogs/Ablation+$modelName+HiddenKiller+OnlyAdapter+$dataset+$notation.log")

prefixConfigs=("./frequencyConfigs/Badnets4OnlyPrefixConfig.json" "./frequencyConfigs/AddsentsOnlyPrefixConfig.json" "./frequencyConfigs/StyleOnlyPrefixConfig.json" "./frequencyConfigs/HiddenKillerOnlyPrefixConfig.json")
prefixLogs=("./frequencyLogs/Ablation+$modelName+Badnets4+OnlyPrefix+$dataset+$notation.log" "./frequencyLogs/Ablation+$modelName+Addsents+OnlyPrefix+$dataset+$notation.log" "./frequencyLogs/Ablation+$modelName+Style+OnlyPrefix+$dataset+$notation.log" "./frequencyLogs/Ablation+$modelName+HiddenKiller+OnlyPrefix+$dataset+$notation.log")

if [ "$way" == 'vanilla' ]; then
    configs=("${vanillaConfigs[@]}")
    logs=("${vanillaLogs[@]}")
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
elif [ "$way" == 'adapter' ]; then
    configs=("${adapterConfigs[@]}")
    logs=("${adapterLogs[@]}")
elif [ "$way" == 'prefix' ]; then
    configs=("${prefixConfigs[@]}")
    logs=("${prefixLogs[@]}")
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
    echo "nohup python -u fourierAnalysis.py --config_path $config --dataset $dataset --target_model $modelName --poison_rate $poison_rate > $log 2>&1 &"
    nohup python -u fourierAnalysis.py --config_path $config --dataset $dataset --target_model $modelName --poison_rate $poison_rate > $log 2>&1 &
    echo $log
    sleep 60s
done

