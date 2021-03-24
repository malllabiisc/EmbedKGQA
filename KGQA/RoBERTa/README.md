# Relation matching module

UPDATE: Please see https://github.com/malllabiisc/EmbedKGQA/issues/69 for an issue related to a missing file.

As mentioned in Section 4.4.1 of paper, a model is trained for detecting relations associated to the question, and this model is further used in relation matching.

## Model

File `pruning_main.py` trains this model. A pre-trained version is available in the downloadable pretrained_models.zip

## Relation matching scoring

Please see the ipython notebook `relation_matching_eval.ipynb` for scoring QA model outputs with relation matching. This notebook assumes that the scores of the
base QA model are stored in the file `webqsp_scores_full_kg.pkl`. This can be done by setting `writeCandidatesToFile = True` in the validate function in `main.py`. 
Then using these scores as well as the above mentioned model the final answer scores are calculated.
