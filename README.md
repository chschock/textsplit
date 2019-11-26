# Introduction
This library contains simple functionality to tackle the problem of segmenting
documents into coherent parts. Imagine you don't have a good paragraph
annotation in your documents, as it is often the case for scraped pdfs or html
documents. For NLP tasks you want to split them at points where the topic
changes. Good results have been achieved using topic representations, but they
involve a further step of topic modeling which is quite domain dependent. This
approach uses only word embeddings which are assumed to be less domain specific.
See [https://arxiv.org/pdf/1503.05543.pdf] for an overview and an approach very
similar to the one presented here.


The algorithm uses word embeddings to find a segmentation where the splits are
chosen such that the segments are coherent. This coherence can be described as
accumulated weighted cosine similarity of the words of a segment to the mean
vector of that segment.  More formally segments are chosen as to maximize the
quantity |v|, where v is a segment vector and |.| denotes the l2-norm. The
accumulated weighted cosine similarity turns up by a simple transformation:
|v| = 1/|v| <v, v> = <v, v/|v|> = \sum_i <w_i, v/|v|> = \sum_i |w_i| <w_i/|w_i|, v/|v|>,
where v = \sum_i w_i is the definition of the segment vector from word vectors
w_i. The expansion gives a good intuition of what we try to achieve. As we
usually compare word embeddings with cosine similarity, the last scalar product
<w_i/|w_i|, v/|v|> is just the cosine similarity of a word w_i to the segment
vector v. The weighting with the length of w_i suppresses frequent noise words,
that typically have a shorter length.

This leads to the interpretation that coherence corresponds to segment vector
length, in the sense that two segment vectors of same length contain the same
amount of information. This interpretation is of course only capturing
information that we are given as input by means of the word embeddings, but it
serves as an abstraction.

# Formalization

To optimize for segment vector length |v|, we look for a sequence of split
positions such that the sum of l2-norms of the segment vectors formed by summing
the words between the splits is maximal. Given this objective without
constraints, the optimal solution is to split the document between every two
subsequent words (triangle inequality). We have to impose some limit on the
granularity of the segmentation to get useful results. This is done by a penalty
for every split made, that counts against the vector norms, i.e. is subtracted
from the sum of vector norms.

Let Seg := {(0 = t_0 < t_i < ... < t_n = L) | s_i natural number} where L is a
documents length. A segment [a, b[ comprises the words at positions a, a+1, ...,
b-1. Let l(j, k) := |\sum_i=j^{k-1} w_i| denote the vector of segment [i, j[. We
optimize the function f mapping elements of Seg to the real numbers with
f: (t_0, ..., t_n) \mapsto \sum_{i=0}^{n-1} (l(t_{i-1}, t_i) + l(t_i, t_{i+1}) - penalty).

# Algorithms

There are two variants, a greedy that is fast and a dynamic programming approach
that computes the optimal segmentation. Both depend on a penalty hyperparameter,
that defined the granularity of the split.

## Greedy
Split the text iteratively at the position where the gain is highest until this
gain would be below a given penalty threshold. The gain is the sum of norms of
the left and right segments minus the norm of the segment that is to be split.

## Optimal (Dynamic Programming)
Iteratively construct a data structure storing the results of optimally
splitting a prefix of the document. This results in a matrix storing a score
for making a segment from position i to j, given a optimal segmentation up to i.

# Tools

## Penalty hyperparameter choice
The greedy implementation does not need the penalty parameter, but can also be
run by limiting the number of segments. This is leveraged by the `get_penalty`
function to approximately determine a penalty parameter for a desired average
segment length computed over a set of documents.

## Measure accuracy of segmentation against reference
To measure the accuracy of an algorithm against a given reference segmentation
`P_k` is a commonly used metric described e.g. in above paper.

## Apply segmentation definition to document
The function `get_segments` simply applies a segmentation determined by one of
the algorithms to e.g. the sentences of a text used when generating the
segmentation.

# Usage

## Input
The algorithms are fed a matrix `docmat` containing vectors representing the
content of a text. These vectors are supposed to have cosine similarity as a
natural similarity measure and length roughly corresponding to the content
length of a text particle. Particles could be words in which case word2vec
embeddings are a good choice as vectors. The width of `docmat` is the embedding
dimension and the height the number of particles.

## Split along sentence borders
If you want to split text into paragraphs, you most likely already have a good
idea of what potential sentence borders are. It makes sense not to give the word
vectors as input but sentence vectors formed by e.g. the sum of word vectors, as
it is usual practice.


# Installation
Clone the repo, then install it from within the root folder with
```bash
pip install -e .
```

# Getting Started
In the Jupyter notebook HowTo.ipynb you find code that demonstrates the use of
the module. It downloads a corpus to trains word2vec vectors on and an example
text for segmentation. You achieve better results if you compute word vectors on
a larger corpus.
