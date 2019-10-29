#!/usr/bin/env bash

exec_file="bin/run_hnsw_search"

base_vectors_file="datasets/siftsmall/siftsmall_base.fvecs"
query_vectors_file="datasets/siftsmall/siftsmall_query.fvecs"
ground_truth_file="datasets/siftsmall/siftsmall_groundtruth.ivecs"

$exec_file -b $base_vectors_file -q $query_vectors_file -g $ground_truth_file
