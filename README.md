# From Form to Meaning: Interlingua Sense-Alignment of Offensive Language with LLMs

This repository contains the code, prompts, and supporting materials for the paper: **From Form to Meaning: Interlingua Sense-Alignment of Offensive Language with LLMs**

The project develops and evaluates a methodology for aligning multilingual offensive-language lexicons at the **sense level**. The alignment covers five languages: **Arabic, Bulgarian, Modern Greek, French, and Italian**. Instead of aligning offensive words only at the lemma level or through English pivot translation, the method compares **lemma–definition–example triples** in the original languages using an LLM-as-a-judge approach.

The final resource aims to support downstream applications such as **Machine Translation**, **interlingual offensive-language detection**, and **multilingual hate-speech research**.

---

## Content warning

This repository contains lexical material related to offensive language, including insults, slurs, and derogatory expressions. The material is provided strictly for academic and research purposes.

---

## Repository structure
```text
├── README.md
├── LICENSE
├── LICENSE-DATA.md
│
├── Dataset_Construction/
│   ├── Definition_Prompt/
│   │   ├── English_Translation_Definition_Prompt/
│   │   └── Modern_Greek_Original_Definition_Prompt/
│   ├── Final_Dataset/
│   └── Code/
│
└── Sense_Alignment_using_LLMs/
    ├── Judging_Prompt/
    ├── Code/
    ├── Jsonl_files_from_alignment_results/
    └── Final_alignment_tuples/
```
---

## Data sources

The final dataset is based on the HurtNet branch of HurtLex, especially the multilingual offensive lexicons for Arabic, Bulgarian, French, and Italian:

https://github.com/valeriobasile/hurtlex/tree/hurtnet/hurtnet_data

For the Modern Greek lexicon, the work also used resources and examples from:

Eris: https://osf.io/hbu4d
AIKIA corpus: https://osf.io/vae2u/?view_only=d21e6fdc5ffc4ac794d4b2c5972d2742

---

## Large alignment outputs

Some LLM output files are too large to store directly in this GitHub repository. They are available separately in the following Google Drive folder:

https://drive.google.com/drive/folders/17DnaYXecgySE4HY7sqoSo0XAVAQT7-l8?usp=sharing

---

## Licenses

The repository uses separate licenses for code and research materials.

**Code**

The source code in this repository is licensed under the MIT License.

See [`LICENSE`](LICENSE)

**Data and research materials**

The lexicon files, prompts, CSV/JSONL files, alignment outputs, and other non-code research materials are licensed under the Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License (CC BY-NC-SA 4.0).

See [`LICENSE-DATA.md`](LICENSE-DATA.md).

**Third-party materials**

This license applies only to materials created by the authors of this repository. Third-party resources, including HurtNet/HurtLex, Eris, AIKIA, and any external corpus examples, remain subject to their original licenses and terms of use.

---

## How to cite

If you use this repository, please cite the associated paper:
```bibtex
@inproceedings{roussopoulou-markantonatou-2026-interlingua,
  title = {From Form to Meaning: Interlingua Sense-Alignment of Offensive Language with LLMs},
  author = {Roussopoulou, Maria Alexandra and Markantonatou, Stella},
  booktitle = {Proceedings of the Sixth Workshop on Language Technology for Equality, Diversity and Inclusion (LT-EDI 2026)},
  month = {july},
  year = {2026},
  address = {San Diego, California},
  publisher = {Association for Computational Linguistics},
  note = {To appear}
}
```
