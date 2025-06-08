# Stage 1 - VaeDiff-DocRE
Implementation of Stage 1 in [VaeDiff-DocRE](https://aclanthology.org/2025.coling-main.488/).

## Environment setup
```bash
>> conda create --name stage1-vaediff-docre python=3.11
>> conda activate stage1-vaediff-docre
>> pip install -r VaeDiff-DocRE/Stage_1/requirements.txt
```

## Training
```sh
>> bash VaeDiff-DocRE/Stage_1/scripts/train.sh
```

## Testing
```sh
>> bash VaeDiff-DocRE/Stage_1/scripts/eval.sh
```
