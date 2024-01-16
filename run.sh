
TASK=${1}
FLAG=${2}

echo "Subtask: "
read TASK
echo "download the data? y or n"
read DOWN
echo "process the data? y or n "
read FLAG
if [[${TASK}] == "cite"]
then
    if [[${DOWN}] == "y"]
    then
        python ./utils/download_data.py --subtask ${TASK}
    fi
    if [[${FLAG}] == "y"]
    then 
        python ./utils/feature_selection.py --subtask ${TASK}
    fi
    python train.py
fi
    