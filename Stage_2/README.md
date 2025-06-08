# Stage 2 - VaeDiff-DocRE
Implementation of Stage 2 in [VaeDiff-DocRE](https://aclanthology.org/2025.coling-main.488/).

## Environment setup
```bash
>> conda create --name stage2-vaediff-docre python=3.11
>> conda activate stage2-vaediff-docre
>> pip install -r VaeDiff-DocRE/Stage_2/requirements.txt
```

## Training
```sh
>> bash VaeDiff-DocRE/Stage_2/scripts/train.sh
```

## Evaluation
```sh
>> bash VaeDiff-DocRE/Stage_2/scripts/eval.sh
```