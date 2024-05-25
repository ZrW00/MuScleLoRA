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


vanillaConfigs=("./plmConfigs/Badnets4NoDefense.json" "./plmConfigs/AddsentsNoDefense.json" "./plmConfigs/StyleNoDefense.json" "./plmConfigs/HiddenKillerNoDefense.json")
vanillaLogs=("./plmLogs/Ablation+$modelName+Badnets4+Vanilla+$dataset+$notation.log" "./plmLogs/Ablation+$modelName+Addsents+Vanilla+$dataset+$notation.log" "./plmLogs/Ablation+$modelName+Style+Vanilla+$dataset+$notation.log" "./plmLogs/Ablation+$modelName+HiddenKiller+Vanilla+$dataset+$notation.log")

gaConfigs=("./plmConfigs/Badnets4OnlyGAConfig.json" "./plmConfigs/AddsentsOnlyGAConfig.json" "./plmConfigs/StyleOnlyGAConfig.json" "./plmConfigs/HiddenKillerOnlyGAConfig.json")
gaLogs=("./plmLogs/Ablation+$modelName+Badnets4+OnlyGA+$dataset+$notation.log" "./plmLogs/Ablation+$modelName+Addsents+OnlyGA+$dataset+$notation.log" "./plmLogs/Ablation+$modelName+Style+OnlyGA+$dataset+$notation.log" "./plmLogs/Ablation+$modelName+HiddenKiller+OnlyGA+$dataset+$notation.log")

mslrConfigs=("./plmConfigs/Badnets4OnlyMSLRConfig.json" "./plmConfigs/AddsentsOnlyMSLRConfig.json" "./plmConfigs/StyleOnlyMSLRConfig.json" "./plmConfigs/HiddenKillerOnlyMSLRConfig.json")
mslrLogs=("./plmLogs/Ablation+$modelName+Badnets4+OnlyMSLR+$dataset+$notation.log" "./plmLogs/Ablation+$modelName+Addsents+OnlyMSLR+$dataset+$notation.log" "./plmLogs/Ablation+$modelName+Style+OnlyMSLR+$dataset+$notation.log" "./plmLogs/Ablation+$modelName+HiddenKiller+OnlyMSLR+$dataset+$notation.log")

loraConfigs=("./plmConfigs/Badnets4OnlyLoRAConfig.json" "./plmConfigs/AddsentsOnlyLoRAConfig.json" "./plmConfigs/StyleOnlyLoRAConfig.json" "./plmConfigs/HiddenKillerOnlyLoRAConfig.json")
loraLogs=("./plmLogs/Ablation+$modelName+Badnets4+OnlyLoRA+$dataset+$notation.log" "./plmLogs/Ablation+$modelName+Addsents+OnlyLoRA+$dataset+$notation.log" "./plmLogs/Ablation+$modelName+Style+OnlyLoRA+$dataset+$notation.log" "./plmLogs/Ablation+$modelName+HiddenKiller+OnlyLoRA+$dataset+$notation.log")

gaLoraConfigs=("./plmConfigs/Badnets4GA+LoRAConfig.json" "./plmConfigs/AddsentsGA+LoRAConfig.json" "./plmConfigs/StyleGA+LoRAConfig.json" "./plmConfigs/HiddenKillerGA+LoRAConfig.json")
gaLoraLogs=("./plmLogs/Ablation+$modelName+Badnets4+GA+LoRA+$dataset+RandomRef+$notation.log" "./plmLogs/Ablation+$modelName+Addsents+GA+LoRA+$dataset+RandomRef+$notation.log" "./plmLogs/Ablation$modelName+Style+GA+LoRA+$dataset+RandomRef+$notation.log" "./plmLogs/Ablation+$modelName+HiddenKiller+GA+LoRA+$dataset+RandomRef+$notation.log")

gaLoraMSLRConfigs=("./plmConfigs/Badnets4GA+LoRA+MSLRConfig.json" "./plmConfigs/AddsentsGA+LoRA+MSLRConfig.json" "./plmConfigs/StyleGA+LoRA+MSLRConfig.json" "./plmConfigs/HiddenKillerGA+LoRA+MSLRConfig.json")
gaLoraMSLRLogs=("./plmLogs/Ablation+$modelName+Badnets4+GA+LoRA+MSLR+$dataset+RandomRef+$notation.log" "./plmLogs/Ablation+$modelName+Addsents+GA+LoRA+MSLR+$dataset+RandomRef+$notation.log" "./plmLogs/Ablation+$modelName+Style+GA+LoRA+MSLR+$dataset+RandomRef+$notation.log" "./plmLogs/Ablation+$modelName+HiddenKiller+GA+LoRA+MSLR+$dataset+RandomRef+$notation.log")

adapterConfigs=("./plmConfigs/Badnets4OnlyAdapterConfig.json" "./plmConfigs/AddsentsOnlyAdapterConfig.json" "./plmConfigs/StyleOnlyAdapterConfig.json" "./plmConfigs/HiddenKillerOnlyAdapterConfig.json")
adapterLogs=("./plmLogs/Ablation+$modelName+Badnets4+OnlyAdapter+$dataset+$notation.log" "./plmLogs/Ablation+$modelName+Addsents+OnlyAdapter+$dataset+$notation.log" "./plmLogs/Ablation+$modelName+Style+OnlyAdapter+$dataset+$notation.log" "./plmLogs/Ablation+$modelName+HiddenKiller+OnlyAdapter+$dataset+$notation.log")

prefixConfigs=("./plmConfigs/Badnets4OnlyPrefixConfig.json" "./plmConfigs/AddsentsOnlyPrefixConfig.json" "./plmConfigs/StyleOnlyPrefixConfig.json" "./plmConfigs/HiddenKillerOnlyPrefixConfig.json")
prefixLogs=("./plmLogs/Ablation+$modelName+Badnets4+OnlyPrefix+$dataset+$notation.log" "./plmLogs/Ablation+$modelName+Addsents+OnlyPrefix+$dataset+$notation.log" "./plmLogs/Ablation+$modelName+Style+OnlyPrefix+$dataset+$notation.log" "./plmLogs/Ablation+$modelName+HiddenKiller+OnlyPrefix+$dataset+$notation.log")

if [ "$way" == 'vanilla' ]; then
    configs=("${vanillaConfigs[@]}")
    logs=("${vanillaLogs[@]}")
elif [ "$way" == 'ga' ]; then
    configs=("${gaConfigs[@]}")
    logs=("${gaLogs[@]}")
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
    echo "nohup python -u plmDefense.py --config_path $config --dataset $dataset --target_model $modelName --poison_rate $poison_rate > $log 2>&1 &"
    nohup python -u plmDefense.py --config_path $config --dataset $dataset --target_model $modelName --poison_rate $poison_rate > $log 2>&1 &
    echo $log
    sleep 60s
done

