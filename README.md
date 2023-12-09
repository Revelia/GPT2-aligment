### GPT-2 aligment using BERT classifier

## Введение

Данный репозиторий содержит код и результаты серии экспериментов по GPT-2 aligment

Под aligment имеется ввиду дообучение генеративной модели с целью корректировки результатов генерации в соответствии с нашими ожиданиями. 
Например, мы будем использовать [GPT-2 обученную на imbd датасете](https://huggingface.co/lvwerra/gpt2-imdb), которая генерирует отзывы к фильмам, причем она была обучена как на негативных так и на положительных отзывах. Мы хотим, чтобы модель генерировала положительные отзывы.

Мы использовали подход [SLiC-HF](https://arxiv.org/pdf/2305.10425.pdf)

##pipeline

Как было ранее отмечено, в качестве генеративной SFT(supervised fine tuned) модели была использована [GPT-2 обученную на imbd датасете](https://huggingface.co/lvwerra/gpt2-imdb) для генерации отзывов на фильмы. В качестве reward функции мы взяли [BERT](https://huggingface.co/lvwerra/distilbert-imdb) модель, которая классифицирует отзывы на положительные/негативные.

При помощи GPT-2 было сгенерированно 1000 отзывов и посчитан reward для каждого (generate_text.py)

Далее было составлено 2000 пар winner-loser, которые в дальнейшем будут использоваться как датасет для обучения DPO (prepare_dataset.py)

При помощи dpo Trainer зафайнтьюнили GPT-2 (main.py) и посчитали распределние reward по отзывам(calculate_metrics.py)

Итоговое распределение выглядит следующим образом:

![Image alt](https://github.com/Revelia/GPT2-aligment/blob/master/images/result.jpg)
