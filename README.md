# RelationExtraction

A machine learning system for extracting complex relationships between entities from unstructured text data, leveraging the latest advancements in large language models.

## Contributors

* Badri Narayanan Murali Krishnan [LinkedIn](https://www.linkedin.com/in/mbadrinarayanan/) [GitHub](https://github.com/MBadriNarayanan)
* Devaki Kulkarni [LinkedIn](https://www.linkedin.com/in/devaki-kulkarni-53252318b/) [GitHub](https://github.com/devaki9)
* Sree Sanjeev Kandasamy Gokulrajan [LinkedIn](https://www.linkedin.com/in/sreesanjeevkg/) [GitHub](https://github.com/sreesanjeevkg)

## Citations

* Model used: [REBEL](https://huggingface.co/Babelscape/rebel-large)
* Dataset used: [WebNLG](https://huggingface.co/datasets/web_nlg)

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

```
@inproceedings{web_nlg,
  author    = {Claire Gardent and
               Anastasia Shimorina and
               Shashi Narayan and
               Laura Perez{-}Beltrachini},
  editor    = {Regina Barzilay and
               Min{-}Yen Kan},
  title     = {Creating Training Corpora for {NLG} Micro-Planners},
  booktitle = {Proceedings of the 55th Annual Meeting of the Association for Computational
               Linguistics, {ACL} 2017, Vancouver, Canada, July 30 - August 4, Volume
               1: Long Papers},
  pages     = {179--188},
  publisher = {Association for Computational Linguistics},
  year      = {2017},
  url       = {https://doi.org/10.18653/v1/P17-1017},
  doi       = {10.18653/v1/P17-1017}
}
```