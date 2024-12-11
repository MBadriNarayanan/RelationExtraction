# RelationExtraction

Extracting complex relationships between entities from unstructured text data, by fine-tuning Meta Llama3.2 Large Language Model (LLM).

## Contributors

* Badri Narayanan Murali Krishnan [LinkedIn](https://www.linkedin.com/in/mbadrinarayanan/) [GitHub](https://github.com/MBadriNarayanan)
* Son Dam [LinkedIn](https://www.linkedin.com/in/son-dam-856853209/) [GitHub](https://github.com/damchauson)
* Neha Chadaga [LinkedIn](https://www.linkedin.com/in/neha-chadaga/) [GitHub](https://github.com/neha-chadaga)

## Base LLM used

- [Meta Llama 3.2-1B](https://huggingface.co/meta-llama/Llama-3.2-1B)

## Replicating Baseline Results

Replicated the following scripts and trained a smaller variant to get baseline results
- `generate_samples.py`
- `pl_data_modules.py`
- `pl_modules.py`
- `utils.py`

## Steps to run the code

### Download dataset

- Download zip file (don't use wget): https://drive.google.com/file/d/1489PwEFe9S6_OFDQE-ovIgoma7zCsQ6T/view?usp=sharing
- Put zip file into this directory
- `unzip prepared_data.zip`

### Install requirements

- `pip3 install --upgrade pip`
- `pip3 install -r requirements.txt`

### Fine-tune

- `python3 src/train.py`

### Generate Samples from fine-tuned model

- `python3 src/generate_test_samples.py`

### Generate metrics based on prediction

- `python3 src/metrics.py`

## Baseline

* Model & Dataset used: [REBEL: Relation Extraction By End-to-end Language generation
](https://github.com/Babelscape/rebel)

```
@inproceedings{huguet-cabot-navigli-2021-rebel-relation,
    title = "{REBEL}: Relation Extraction By End-to-end Language generation",
    author = "Huguet Cabot, Pere-Llu{\'\i}s  and
      Navigli, Roberto",
    booktitle = "Findings of the Association for Computational Linguistics: EMNLP 2021",
    month = nov,
    year = "2021",
    address = "Punta Cana, Dominican Republic",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2021.findings-emnlp.204",
    pages = "2370--2381",
    abstract = "Extracting relation triplets from raw text is a crucial task in Information Extraction, enabling multiple applications such as populating or validating knowledge bases, factchecking, and other downstream tasks. However, it usually involves multiple-step pipelines that propagate errors or are limited to a small number of relation types. To overcome these issues, we propose the use of autoregressive seq2seq models. Such models have previously been shown to perform well not only in language generation, but also in NLU tasks such as Entity Linking, thanks to their framing as seq2seq tasks. In this paper, we show how Relation Extraction can be simplified by expressing triplets as a sequence of text and we present REBEL, a seq2seq model based on BART that performs end-to-end relation extraction for more than 200 different relation types. We show our model{'}s flexibility by fine-tuning it on an array of Relation Extraction and Relation Classification benchmarks, with it attaining state-of-the-art performance in most of them.",
}
```
