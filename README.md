
<div class="sponsor-logos">
  <a href="http://inab.certh.gr/" title="INAB/CERTH">
    <img style="margin-right:50px" alt="INAB/CERTH" src="static/images/INAB-logo.png" width="70"/>
  </a>
  <a href="https://www.elixir-europe.org/" title="ELIXIR">
    <img style="margin-right:50px" alt="ELIXIR" src="static/images/ELIXIR-logo.png" width="50"/>
  </a>  
  <a href="https://biodata.pt/" title="ELIXIR">
    <img style="margin-right:50px" alt="ELIXIR" src="static/images/Biodata_ELIXIR.png" width="200"/>
  </a>
  <a href="https://www.igc.gulbenkian.pt/" title="ELIXIR">
    <img style="margin-right:50px" alt="ELIXIR" src="static/images/IGC_Black.png" width="200"/>
  </a>
</div>  

## Overview of the course material for the ELIXIR-PT "Introduction to Machine Learning Using R" course

**When**: 11-18 November 2021, 09:00 - 12:00 UTC

**Where**: F2F

**Registration**: CLOSED


### Instructors and helpers

**Instructors**:

- Wandrille Duchemin (ELXIIR-CH, Basel University>
- Crhistian Cardona (ELIXIR-UK, University of Tuebingen)

**Helpers**:

- to be updated

### Overview
With the rise in high-throughput sequencing technologies, the volume of omics data has grown exponentially in recent times and a major issue is to mine useful knowledge from these data which are also heterogeneous in nature. Machine learning (ML) is a discipline in which computers perform automated learning without being programmed explicitly and assist humans to make sense of large and complex data sets. The analysis of complex high-volume data is not trivial and classical tools cannot be used to explore their full potential. Machine learning can thus be very useful in mining large omics datasets to uncover new insights that can advance the field of bioinformatics.

This 4-half-day course will introduce participants to the machine learning taxonomy and the applications of common machine learning algorithms to omics data. The course will cover the common methods being used to analyse different omics data sets by providing a practical context through the use of basic but widely used R libraries. The course will comprise a number of hands-on exercises and challenges where the participants will acquire a first understanding of the standard ML processes, as well as the practical skills in applying them on familiar problems and publicly available real-world data sets.

### Learning objectives

At the end of the course, the participants will be able to:
- Understand the ML taxonomy and the commonly used machine learning algorithms for analysing “omics” data
- Understand differences between ML algorithms categories and to which kind of problem they can be applied
- Understand different applications of ML in different -omics studies
- Use some basic, widely used R packages for ML
- Interpret and visualize the results obtained from ML analyses on omics datasets
- Apply the ML techniques to analyse their own datasets

### Audience and requirements

This course is intended for master and PhD students, post-docs and staff scientists familiar with different omics data technologies who are interested in applying machine learning to analyse these data. No prior knowledge of Machine Learning concepts and methods is expected nor required.

### Prerequisites

#### Knowledge / competencies

Familiarity with any programming language will be required (familiarity with R will be preferable).

#### Technical
This course will be streamed, you are thus required to have your own computer with an internet connection. In order to ensure clear communication between Instructors and participants, we will be using collaborative tools, such as [Google Drive](https://www.google.com/drive/) and/or Google Docs.

_Maximum participants_: 30

### Schedule

**Day 1**

| Time  |  Details |
|--------|----------|
| 09:00 - 09:30 | **Course Introduction**. <br /> <br /> - Welcome. <br /> - Introduction and CoC. <br /> - Way to interact <br /> - Practicalities (agenda, breaks, etc). <br />- Setup <br /> [_Link to material_](episodes/setup.md) |
| 09:30 - 10:00 | **Introduction to Machine Learning** (_theory_) |
| 10:00 - 11:30 | **What is Exploratory Data Analysis (EDA) and why is it useful?** (_hands-on_) <br /><br /> - Loading omics data <br /> - PCA <br /> [_Link to material_](episodes/03-eda-unsupervised-learning.md) |
| 11:30 - 11:45 | _Coffee Break_ |
| 11:45 - 12:15 | **Introduction to Unsupervised Learning** (_theory_) |
| 12:15 - 13:00 | **Agglomerative Clustering: k-means** (_practical_) [_Link to material_](episodes/03-eda-unsupervised-learning.md) |
| 13:00 - 14:00 | _Lunch break_ |
| 14:00 - 14:45 | **Agglomerative Clustering: k-means** (_practical_) (cont'd) |
| 14:45 - 15:30 | **Divisive Clustering: hierarchical clustering** (_practical_) [_Link to material_](episodes/03-eda-unsupervised-learning.md) |
| 15:30 - 15:45 | _Coffee Break_ |
| 15:45 - 16:30 | **Divisive Clustering: hierarchical clustering** (_practical_) (cont'd) |
| 16:30         | _Closing of Day 1_ |


**Day 2**

| Time  |  Details |
|--------|----------|
| 09:00 - 09:30 | **Welcome Day 2**. <br /> <br /> - Questions from Day 1 <br /> - Agenda <br /> |
| 09:30 - 10:00 | **Introduction to Supervised Learning** (_theory_) <br /> <br /> - Overview of multiple algorithms <br /> - Advantages and Disadvantages <br /> |
| 10:00 - 10:30 | **Classification Metrics** (_theory_) <br /> <br /> - F1 Score, Precision, Recall <br /> - Confusion Matrix, ROC-AUC <br /> |
| 10:30 - 11:30 | **Classification** (_practical_) <br /> <br /> - Decision trees <br /> - Random Forests <br /> [_Link to material_](episodes/04-supervised-learning.md) |
| 11:30 - 11:45 | _Coffee Break_ |
| 11:45 - 12:30 | **Classification** (_practical_) (cont'd)
| 12:30 - 13:30 | _Lunch break_ |
| 13:30 - 14:00 | **Regression** (_theory_) |
| 14:00 - 15:15 | **Regression** (_practical_) <br /> <br /> - Linear regression <br /> - Generalized Linear Model (GLM) <br />  [_Link to material_](episodes/04-supervised-learning.md) |
| 15:15 - 15:30 | _Coffee Break_ |
| 15:30 - 16:00 | **Regression** (_practical_) (cont'd) |
| 16:00 - 16:30 | _Closing questions, Discussion_ |

## Other examples

If you finish all the exercises and wish to practice on more examples, here are a couple of good examples to help you get more familiar with the different ML techniques and packages.
1. [RNASeq Analysis in R](https://combine-australia.github.io/RNAseq-R/06-rnaseq-day1.html)
2. [Use the Iris R built-in data set](https://github.com/fpsom/CODATA-RDA-Advanced-Bioinformatics-2019/blob/master/3.Day3.md) to run clustering and also some supervised classification and compare results obtained by different methods.

## Sources / References

The material in the workshop has been based on the following resources:

1. [ELIXIR CODATA Advanced Bioinformatics Workshop](https://codata-rda-advanced-bioinformatics-2019.readthedocs.io)
2. [Machine Learning in R](https://hugobowne.github.io/machine-learning-r/), by [Hugo Bowne-Anderson](https://twitter.com/hugobowne) and [Jorge Perez de Acha Chavez](https://twitter.com/jorge_pda)
3. [Practical Machine Learning in R](https://leanpub.com/practical-machine-learning-r), by [Kyriakos Chatzidimitriou](https://leanpub.com/u/kyrcha), [Themistoklis Diamantopoulos](https://leanpub.com/u/thdiaman), [Michail Papamichail](https://leanpub.com/u/mpapamic), and [Andreas Symeonidis](https://leanpub.com/u/symeonid).
4. [Linear models in R](https://monashbioinformaticsplatform.github.io/r-linear/topics/linear_models.html), by the [Monash Bioinformatics Platform](https://www.monash.edu/researchinfrastructure/bioinformatics)
5. Relevant blog posts from the [R-Bloggers](https://www.r-bloggers.com/) website.
6. [Predicting the breast cancer by characteristics of the cell nuclei present in the image](https://rstudio-pubs-static.s3.amazonaws.com/411600_3185f5d17d104cc5beb4587094b905e9.html#comparison-with-scientific-paper)



Relevant literature includes:

1. [Pattern Recognition and Machine Learning](http://users.isr.ist.utl.pt/~wurmd/Livros/school/Bishop%20-%20Pattern%20Recognition%20And%20Machine%20Learning%20-%20Springer%20%202006.pdf) by Christopher M. Bishop.
2. [Machine learning in bioinformatics](https://academic.oup.com/bib/article/7/1/86/264025), by Pedro Larrañaga et al.
3. [Ten quick tips for machine learning in computational biology](https://biodatamining.biomedcentral.com/articles/10.1186/s13040-017-0155-3), by Davide Chicco
3. [Statistics versus machine learning](https://www.nature.com/articles/nmeth.4642)
4. [Machine learning and systems genomics approaches for multi-omics data](https://biomarkerres.biomedcentral.com/articles/10.1186/s40364-017-0082-y)
5. [A review on machine learning principles for multi-view biological data integration](https://academic.oup.com/bib/article/19/2/325/2664338)
6. [Generalized Linear Model](https://www.sciencedirect.com/topics/mathematics/generalized-linear-model)

## Additional information

**Coordination**: Pedro L. Fernandes, ELIXIR-PT, [IGC Gulbenkian](https://gulbenkian.pt/)

ELIXIR-UK abides by the [ELIXIR Code of Conduct](https://elixir-europe.org/events/code-of-conduct). Participants of ELIXIR-UK courses are also required to abide by the same code.

## License

[![License: CC BY 4.0](https://licensebuttons.net/l/by/4.0/88x31.png)](https://creativecommons.org/licenses/by/4.0/)

This material is made available under the [Creative Commons Attribution 4.0 International license](https://creativecommons.org/licenses/by/4.0). Please see [LICENSE](LICENSE.md) for more details.

## Citation

Shakuntala Baichoo, Wandrille Duchemin, Geert van Geest, Thuong Van Du Tran, Fotis E. Psomopoulos, & Monique Zahn. (2020, July 23). Introduction to Machine Learning (Version v1.0.0). Zenodo. http://doi.org/10.5281/zenodo.3958880
