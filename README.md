# Code and data for "Causal evidence of identity-based and institutional biases in accessing paywalled articles and scientific data"

Ibrahim, Liu, Mengal, Alshaibi, Kaufman, Zaki & Rahwan.

This repository contains the code and data needed to reproduce the quantitative results of the paper:

| Study | What it is | Folder |
|---|---|---|
| Study 2 | Observational analysis of citations to paywalled vs. open-access papers (OpenAlex) | `observational_analyses/` |
| Study 3 | Randomized audit experiment: requests for paywalled papers (18 conditions) | `audit_experiments/paywalled_papers_audit/` |
| Study 4 | Observational analysis of data citations to papers whose data are "available upon request" (Semantic Scholar + OpenAlex) | `observational_analyses/` |
| Study 5 | Randomized audit experiment: requests for "available upon request" datasets (16 conditions) | `audit_experiments/dataset_audit/` |

Study 1 (survey and interviews) is not included: aggregate survey statistics are reported in Supplementary Tables 1–5 and 26, and interview transcripts cannot be shared under the study's consent protocol (see the paper's Data Availability statement).

## Repository structure

```
README.md                  this file
LICENSE                    MIT licence
requirements.txt           pinned Python dependencies
run_audit_demo.sh          runs all audit-experiment analyses (the demo)
expected_output/           output of run_audit_demo.sh on our machine, for comparison
audit_experiments/
  paywalled_papers_audit/  Study 3: anonymized data, regression models, Figure 3
  dataset_audit/           Study 5: anonymized data, regression models, Figure 4
observational_analyses/
  notebooks/               Jupyter notebooks for Studies 2 and 4
  src/                     Doc2Vec training and knowledge-breadth scripts (Study 2)
  data/                    small input files; large files are downloaded separately (see below)
  figures/                 figures produced by the notebooks
```

## 1. System requirements

**Software**

- Python 3.12 (tested with 3.12.2 and 3.12.8; Python 3.10 or later should work)
- The Python packages listed with exact versions in `requirements.txt`: numpy 1.26.4, pandas 2.2.3, scipy 1.13.1, statsmodels 0.14.4, matplotlib 3.9.4, seaborn 0.13.2, geopandas 1.0.1, gensim 4.3.3, tqdm 4.67.1, ipywidgets 8.1.5, jupyterlab 4.3.5
- Any operating system that runs Python (macOS, Linux, Windows). `run_audit_demo.sh` needs a bash shell; on Windows, run the four Python scripts directly (see section 3).

**Tested on**

- macOS 26.7.1 (Apple M1 Pro, 16 GB RAM), Python 3.12.2: audit-experiment analyses (Studies 3 and 5), including the demo
- Linux high-performance computing cluster (NYU Abu Dhabi), Python 3.12.8: observational notebooks (Studies 2 and 4)

**Hardware**

- The demo and the audit-experiment analyses run on a normal desktop or laptop computer. No non-standard hardware is needed.
- The observational notebooks read input files of up to about 3 GB each, plus local snapshots of OpenAlex (several hundred GB of disk), Semantic Scholar and SciSciNet.
- Fine-tuning ModernBERT (`Study_4_classification.ipynb`) needs a CUDA-capable GPU.
- Training the Doc2Vec model from scratch (Study 2, `observational_analyses/src/doc2vec.py`) is optional, because its outputs are provided as downloadable files. On 42 million abstracts it needed about 430 GB of RAM and 4 days on a high-performance computing cluster.

## 2. Installation guide

```bash
git clone https://github.com/comnetsAD/bias_academia.git
cd bias_academia
python3 -m venv venv
source venv/bin/activate        # on Windows: venv\Scripts\activate
pip install -r requirements.txt
```

**Typical install time** on a normal desktop computer: about 2 minutes (1 min 42 s measured on the test machine above, on a fresh virtual environment, broadband connection).

## 3. Demo

The demo runs every analysis for both audit experiments (Studies 3 and 5) on the anonymized experimental data included in this repository.

**Instructions**

```bash
source venv/bin/activate
./run_audit_demo.sh
```

On Windows, or to run a single analysis, go to the study folder and run the scripts directly, for example `cd audit_experiments/dataset_audit` and then `python regression_and_figures.py` and `python analysis_bonferroni.py`.

**Expected output**

Each script prints its regression tables to a `.log` file in its own folder and saves figures as PDFs:

| Folder | Files written | Contents |
|---|---|---|
| `paywalled_papers_audit/` | `regression_and_figures.log`, `figure_2a.pdf`, `figure_2b.pdf` | Study 3 logistic regressions of reply and of positive reply (base, controls, interaction models); reply rate by condition (manuscript Fig. 3A) and log-odds coefficients (Fig. 3B) |
| `paywalled_papers_audit/` | `analysis_bonferroni.log`, `figure_2a_bonferroni.pdf`, `figure_2b_bonferroni.pdf`, `log_odds_model_2_bonferroni.csv` | the same models with Bonferroni-adjusted p-values and confidence intervals |
| `dataset_audit/` | `regression_and_figures.log`, `figure_4a.pdf`, `figure_4b.pdf` | Study 5 logistic regressions of reply (base, controls, interaction models); reply rate by condition (Fig. 4A) and log-odds coefficients (Fig. 4B) |
| `dataset_audit/` | `analysis_bonferroni.log`, `figure_4b_bonferroni.pdf`, `logistic_regression_reply_rate_bonferroni.csv` | the same models with Bonferroni-adjusted p-values and confidence intervals |

The script file names (`figure_2*`, `figure_4*`) follow an earlier figure numbering; they correspond to Figures 3 and 4 of the manuscript.

Reference copies of all of these outputs are in `expected_output/`. Your coefficients should match them to the printed precision. Matplotlib may warn that the Helvetica font is missing; this only changes the figure font.

For example, the Study 5 model with controls (`dataset_audit/regression_and_figures.log`, N = 9,971) should report these log-odds coefficients:

| Term | Coefficient | p |
|---|---|---|
| South African sender (vs. English) | −0.253 | < 0.001 |
| South African university (vs. English) | −0.226 | < 0.001 |
| Low-ranked university (vs. high) | −0.171 | 0.005 |

These correspond to the 22.3%, 20.2% and 15.7% lower odds of reply reported in the paper.

**Expected run time** on a normal desktop computer: about 20 seconds for the whole demo (19 s measured on the test machine above).

## 4. Instructions for use

### Running the audit analyses on your own data

The scripts read one CSV per experiment, with one row per email. To analyse a new audit experiment, prepare a CSV with the same columns and replace the input file, or change the `pd.read_csv(...)` line at the top of the script.

Study 3 (`paywalled_papers_audit_reply_data.csv`):

| Column | Values |
|---|---|
| `sender_name` | sender condition (`Karl Muller`, `Olu Adeyemi`, `Faisal Khan`) |
| `sender_status` | request purpose (`Journal Article`, `Class project`) |
| `sender_university` | `New York University`, `Louisiana State University`, `San Diego State University` |
| `paper_domain` | `Health Sciences`, `Life Sciences`, `Physical Sciences`, `Social Sciences` |
| `h_index`, `academic_age` | recipient h-index and years since first publication (from OpenAlex) |
| `institution_continent` | region of the recipient's institution |
| `matching_continent` | whether the sender's region matches the recipient's (`No Match`, `Match German`, `Match Nigerian`, `Match Asian`) |
| `replied` | 1 if a substantive (non-automated) reply was received, else 0 |
| `is_positive_reply` | 1 if the reply shared the requested paper, else 0 |

Study 5 (`dataset_audit_data.csv`, one row per delivered email):

| Column | Values |
|---|---|
| `sender`, `sender_country` | sender identity (`James Whitfield` / `England`, `Kabelo Molefe` / `South Africa`) |
| `university_country` | `England`, `South Africa` |
| `university_ranking` | `High` (QS 150–155), `Low` (QS 1001–1200) |
| `request_purpose` | `journal article`, `final course project` |
| `paper_domain` | as above |
| `h_index`, `academic_age` | as above |
| `replied` | 1 if a substantive reply was received, else 0 |
| `reply_type` | for replies: `positive`, `conditional`, `negative` (codebook: Supplementary Table 25) |

Reference levels are set in the model formulas, for example `Treatment(reference='England')`. If your factor levels have different names, change them there.

### Reproducing the observational results (Studies 2 and 4)

1. **Get the input data.**
   - Small inputs are already in `observational_analyses/data/`.
   - Large inputs (files over 100 MB) must be downloaded from [Figshare](https://figshare.com/articles/dataset/Causal_evidence_of_racial_and_institutional_biases_in_accessing_paywalled_articles_and_scientific_data/30126952) and [Google Drive](https://drive.google.com/drive/folders/1uL3Kn0TNFbF25agYG9d_Ozenk3wiOE0k?usp=sharing), and placed in `observational_analyses/data/large/`.
   - Some notebooks also need a local copy of the [OpenAlex](https://openalex.org/) snapshot and of [SciSciNet](https://github.com/kellogg-cssi/SciSciNet) v1. SciSciNet v2 also works but may give slightly different numbers.
2. **Set the paths.** At the top of each notebook, set `OPENALEX` and `SCISCINET` to your local copies. `DIR` (`../data/study_2_data` or `../data/study_4_data`) and `LARGEFILE` (`../data/large`) work as they are.
3. **Run the notebooks.** Start `jupyter lab` and run each notebook from top to bottom. Apart from the Study 4 order noted below the table, the notebooks are independent of each other. Figures are written to `observational_analyses/figures/`.

| Notebook | Analyses | Figures written | Needs OpenAlex / SciSciNet? |
|---|---|---|---|
| `Study_2AB.ipynb` | share of references to paywalled papers by country; Global North vs. South t-tests and Cohen's d (Fig. 2A–B; Supplementary Fig. 2) | `OA_map_percentageNonOA.pdf`, `map_legend.pdf`, `nonOA_distribution.pdf`, `nonOA_distribution_Q1.pdf`, `nonOA_distribution_T3.pdf` | no (large files only) |
| `Study_2CD.ipynb` | knowledge breadth of reference lists (Fig. 2D; Supplementary Fig. 3) | `Knowledge_breadth_all.pdf`, `Knowledge_breadth_all_50dim.pdf` | no (large files only) |
| `Study_2EFGH.ipynb` | OA type and citation share, impact, novelty and disruptiveness regressions, GVIFs, regional back-of-the-envelope estimate | `Impact_novelty_disruptiveness_regression.pdf`, `Impact_paywalled_subregional_estimation.pdf` | yes |
| `Study_4_merge_OpenAlex.ipynb` | matches Semantic Scholar papers to OpenAlex (by MAG ID, DOI, then the OpenAlex API) and assigns paper regions | none (writes intermediate data files) | Semantic Scholar snapshot, SciSciNet v2; queries the OpenAlex API |
| `Study_4_classification.ipynb` | citation-intent classification: stage 1 labels with GPT-5.4-mini (`gpt-5.4-mini-2026-03-17`), stage 2 fine-tunes and applies ModernBERT (`answerdotai/ModernBERT-base`); data-availability status | none | Semantic Scholar snapshot; an OpenAI API key; a CUDA GPU |
| `Study_4.ipynb` | data-citation logit models and GVIFs (Table 1; Supplementary Tables 17–18) | none (tables printed in the notebook) | Semantic Scholar snapshot, SciSciNet v2 |

For Study 4, run `Study_4_merge_OpenAlex.ipynb` and `Study_4_classification.ipynb` before `Study_4.ipynb`. The classification notebook needs extra packages: `pip install torch transformers datasets evaluate openai`.

The Doc2Vec embeddings used by `Study_2CD.ipynb` can be regenerated with `src/doc2vec.py` (training), `src/ComputePaperVecDoc2vec.py` (inference) and `src/refListDiameter.py` (knowledge breadth). These steps need a high-performance computing cluster (see Hardware).

## Data

The audit-experiment files contain no names, email addresses or other identifiers of the researchers who were emailed; recipients are described only by h-index, academic age, field and institutional region. The sender names are those of the fictional requesters.

## Licence

Released under the MIT licence; see `LICENSE`.

## Citation

If you use this code or data, please cite the paper (citation to be added on publication).
