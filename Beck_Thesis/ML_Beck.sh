#!/bin/bash

conda activate py39

declare -a stations=('calais-calais-france-refmar' \         
                'denhelder-hel-nl-rws' \
                'aberdeen-p038-uk-bodc' \
                # 'cuxhaven-cuxhaven-germany-bsh' \
                # 'esbjerg-130121-denmark-dmi' \
                # 'brest-brest-france-refmar' \
                # 'delfzijl-del-nl-rws' \
                # 'hoekvanholla-hvh-nl-rws'
                )


# Loop through each station in the array stations and print the station name
for station in "${!stations[@]}"; do
    echo "Station: ${stations[$station]}"
    # Run the script
    python -W ignore ML_env_Beck.py ${stations[$station]} 'ANN' 'gumbel'
done