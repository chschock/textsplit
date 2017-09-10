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

The algorithm uses word embeddings to form segment representations where segment
splits are chosen such that the segments are coherent. This coherence is based
on the idea of accumulated weighted cosine distance of segment words to the mean
vector of a segment. The mean vector in cosine land is the normalized vector.
More formally segments are chosen to maximize the quantity:
<v, v/|v|> = 1/|v| <v, v> = |v|
where v = \sum_i w_i is a segment vector, with word vectors w_i and |.| denotes
the l2-norm.
The accumulated weighted cosine distance turns up by a simple transformation:
<v, v/|v|> = \sum_i <w_i, v/|v|> = \sum_i |w_i| <w_i/|w_i|, v/|v|>.
Given the fact this is nothing else than the segment vector length |v|,
we look for a sequence of split positions such that the sum of l2-norms of the
segments induced by the splits is maximal.
The optimal solution to this unconstrained problem is splitting a documents into
as many segments as it has words. Therefore one has to introduce a penalty for
the split operation, chosen such that the resulting number of segments seems
appropriate.

# Algorithms
There are two variants, a greedy and a dynamic programming approach that
computes the optimal segmentation.

## Greedy
Split the text iteratively at the position where the gain is highest until this
gain would be below a given penalty threshold.
The gain is the sum of norms of the left and right segments minus the norm of
the segment that is to be split.
\sum_i=b^c \sum_i <w_i, w> + \sum_i=c^e \sum_i <w_i, w> - \sum_i=b^e <w_i, w>
with b < c < e.

## Optimal (Dynamic Programming)
Construct a matrix of accumulated scores with maximal scores at position (i, j)
for making a segment (i, j) and given an already optimal segmentation up to i.
The score is as described and the introduction of a split is demotivated by
subtracting a penalty. This penalty has the same effect as the threshold of the
greedy algorithm and for equal choice the splits are done similarly or for
particular choices of texts results are identical.

# Tools

## Penalty hyperparameter choice
The greedy implementation does not need the penalty parameter, but can also be
run by limiting the number of segments. This is leveraged by the `get_penalty`
function to approximately determine a penalty parameter for a desired average
segment length computed over a set of documents.

## Stream processing of texts on sentence level
In `tools.py` you find functions that can tokenize a stream of texts into
sentences and run the segmentation algorithms on them.

# Usage

## Input
The algorithms are fed a matrix `docmat` containing vectors representing the
content of a text. These vectors are supposed to have cosine similarity as their
natural distance and length roughly corresponding to the content length of a
text particle. Particles could be words in which case word2vec embeddings are a
good choice as vectors. The width of `docmat` is the embedding dimension and the
height the number of particles.

## Split along sentence borders
If you want to split text into paragraphs, you most likely already have a good
idea of what potential sentence borders are. It makes sense not to give the word
vectors as input but sentence vectors formed by e.g. the sum of word vectors, as
it is usual practice.

## Getting Started
In the Jupyter notebook HowTo.ipynb you find code that demonstrates the use of
the module. It downloads a corpus to trains word2vec vectors on and an example
text for segmentation. You achieve better results if you compute word vectors on
a larger corpus.
