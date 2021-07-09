---
annotations_creators:
- no-annotation
language_creators:
- found
languages:
- id
licenses:
- unknown
multilinguality:
- monolingual
size_categories:
- 100K<n<1M
source_datasets:
- original
task_categories:
- conditional-text-generation
task_ids:
- summarization
paperswithcode_id: null
---

# Dataset Card for ID-Collection

## Table of Contents
- [Dataset Description](#dataset-description)
  - [Dataset Summary](#dataset-summary)
  - [Supported Tasks and Leaderboards](#supported-tasks-and-leaderboards)
  - [Languages](#languages)
- [Dataset Structure](#dataset-structure)
  - [Data Instances](#data-instances)
  - [Data Fields](#data-fields)
  - [Data Splits](#data-splits)
- [Dataset Creation](#dataset-creation)
  - [Curation Rationale](#curation-rationale)
  - [Source Data](#source-data)
  - [Annotations](#annotations)
  - [Personal and Sensitive Information](#personal-and-sensitive-information)
- [Considerations for Using the Data](#considerations-for-using-the-data)
  - [Social Impact of Dataset](#social-impact-of-dataset)
  - [Discussion of Biases](#discussion-of-biases)
  - [Other Known Limitations](#other-known-limitations)
- [Additional Information](#additional-information)
  - [Dataset Curators](#dataset-curators)
  - [Licensing Information](#licensing-information)
  - [Citation Information](#citation-information)
  - [Contributions](#contributions)

## Dataset Description

- **Homepage:** 
- **Repository:** 
- **Paper:** 
- **Leaderboard:**
- **Point of Contact:** 

### Dataset Summary

This module load text dataset from local directory. The text dataset should have the format like Oscar dataset
where each new entry is separated by empty lines.

You need to manually collect text datasets in a directory.  The text dataset can then be loaded 
using the following command:
`datasets.load_dataset("id_collection", data_dir="<path/to/dataset>")`.

### Supported Tasks and Leaderboards

[More Information Needed]

### Languages
Indonesian

## Dataset Structure
```
{
  'id': 'int64',
  'text': 'string',
}
```
### Data Instances

An example of the dataset:
```
{
  'id': '1',
  'text': 'sultan agung dan dokternya bilang supaya adeknya diberi kacamata khusus'
}

```

### Data Fields
- `id`: id of the sample
- `text`: content of the article

### Data Splits

The dataset contains only train set.

## Dataset Creation

### Curation Rationale

[More Information Needed]

### Source Data

#### Initial Data Collection and Normalization

[More Information Needed]

#### Who are the source language producers?

[More Information Needed]

### Annotations

#### Annotation process

[More Information Needed]

#### Who are the annotators?
[More Information Needed]

### Personal and Sensitive Information

[More Information Needed]

## Considerations for Using the Data

### Social Impact of Dataset

[More Information Needed]

### Discussion of Biases

[More Information Needed]

### Other Known Limitations

[More Information Needed]

## Additional Information

### Dataset Curators

[More Information Needed]

### Licensing Information

[More Information Needed]

### Citation Information
```

```
### Contributions
