# Corpus sources

<!-- Generated from corpus/manifest.toml by `uv run python -m medquery.ingest sources`. Do not edit by hand. -->

Every document is public-domain, openly licensed, or synthetic. No real patient data.
Raw PDFs are downloaded to the git-ignored `data/raw/` and are not committed.

| ID | Document | Publisher | Year | Topic | As of | URL | Licence | Evidence |
|---|---|---|---|---|---|---|---|---|
| `dm-stats-2020` | National Diabetes Statistics Report, 2020 | CDC | 2020 | diabetes | 2020 | https://stacks.cdc.gov/view/cdc/85309/cdc_85309_DS1.pdf | Public domain | CDC Stacks record https://stacks.cdc.gov/view/cdc/85309 marked Public Domain; https://www.cdc.gov/other/agencymaterials.html |
| `dm-4steps` | 4 Steps to Manage Your Diabetes for Life | National Diabetes Education Program (NIH/CDC) | 2016 | diabetes | 2016 | https://stacks.cdc.gov/view/cdc/55506/cdc_55506_DS1.pdf | Public domain | CDC Stacks record https://stacks.cdc.gov/view/cdc/55506 marked Public Domain |
| `dm-vadod-cpg` | VA/DoD Clinical Practice Guideline for the Management of Type 2 Diabetes Mellitus, v6.0 | US Department of Veterans Affairs / Department of Defense | 2023 | diabetes | 2023 | https://www.healthquality.va.gov/guidelines/CD/diabetes/VADoD-Diabetes-CPG_Final_508.pdf | Public domain | US federal work, 17 U.S.C. §105 (the publisher page states no licence) |
| `htn-dash` | Your Guide to Lowering Your Blood Pressure with DASH | NHLBI (NIH) | 2006 | hypertension | 2006 | https://www.nhlbi.nih.gov/files/docs/public/heart/new_dash.pdf | Public domain | NHLBI statement https://www.nhlbi.nih.gov/es/node/94288 |
| `htn-jnc7-express` | JNC 7 Express: The Seventh Report of the Joint National Committee on Prevention, Detection, Evaluation, and Treatment of High Blood Pressure | NHLBI (NIH) | 2003 | hypertension | 2003 | https://www.nhlbi.nih.gov/files/docs/guidelines/express.pdf | Public domain | NHLBI statement https://www.nhlbi.nih.gov/es/node/94288 |
| `htn-millionhearts` | Hypertension Control Change Package, 2nd Edition | CDC Million Hearts | 2020 | hypertension | 2020 | https://millionhearts.hhs.gov/files/HTN_Change_Package.pdf | Public domain | https://www.cdc.gov/other/agencymaterials.html |
| `tb-qa-2021` | Questions and Answers About Tuberculosis, 2021 | CDC | 2021 | tuberculosis | 2021 | https://stacks.cdc.gov/view/cdc/108893/cdc_108893_DS1.pdf | Public domain | CDC Stacks record https://stacks.cdc.gov/view/cdc/108893 marked Public Domain |
| `tb-ltbi-2020` | Guidelines for the Treatment of Latent Tuberculosis Infection: Recommendations from the NTCA and CDC, 2020 (MMWR Recomm Rep 69(1)) | CDC (MMWR) | 2020 | tuberculosis | 2020 | https://www.cdc.gov/mmwr/volumes/69/rr/pdfs/rr6901a1-H.pdf | Public domain | Stated in the PDF: all material in the MMWR Series is in the public domain |
| `tb-core-curriculum` | Core Curriculum on Tuberculosis: What the Clinician Should Know, 7th Edition | CDC | 2021 | tuberculosis | 2021 | https://stacks.cdc.gov/view/cdc/104503/cdc_104503_DS1.pdf | Public domain | CDC Stacks record https://stacks.cdc.gov/view/cdc/104503 marked Public Domain |
| `tb-reported-1964` | Reported Tuberculosis Data 1964 (PHS Publication No. 638) | US Public Health Service / CDC | 1966 | tuberculosis | 1964 | https://stacks.cdc.gov/view/cdc/114638/cdc_114638_DS1.pdf | Public domain | CDC Stacks record https://stacks.cdc.gov/view/cdc/114638 marked Public Domain |

## Superseded guidance

- `htn-jnc7-express` (as of 2003): Superseded by later hypertension guidance, notably the 2017 ACC/AHA guideline, which lowered the hypertension threshold from 140/90 to 130/80 mmHg.

## Test fixtures

Single pages cut from the documents above and committed under `tests/fixtures/`. Fixtures contain no photos or third-party images.

| File | From | Page | Purpose |
|---|---|---|---|
| `jnc7-express-p19-table.pdf` | `htn-jnc7-express` | 19 | Table page: Table 1, BP classification (open-sided ruled table) |
| `tb-ltbi-2020-p8-booktabs.pdf` | `tb-ltbi-2020` | 8 | Table page: Tables 3-4, regimens and dosages (horizontal rules only) |
| `tb-reported-1964-p5-scan.pdf` | `tb-reported-1964` | 5 | Scanned page: image-only, running text plus a chart in the scan bitmap (OCR test) |
