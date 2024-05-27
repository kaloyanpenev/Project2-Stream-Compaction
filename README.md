CUDA Stream Compaction

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 2**

This is an implementation of Parallel Prefix Sum (Scan) with CUDA, using the UPenn 565 project template, and [GPU Gems 3, Chapter 39](https://developer.nvidia.com/gpugems/gpugems3/part-vi-gpu-computing/chapter-39-parallel-prefix-sum-scan-cuda).

The aim of this repo is for me to get a better understanding of GPU architecture in the context of a common GPGPU algorithm and exercise how it works in practice.

NOTE: There is an exorbitant amount of mistakes in the GPU Gems chapter ([with even the author coming forward and admitting the code should not be used](https://stackoverflow.com/a/9694697)) so I would highly advise against using it. This should be a bit of a better reference for some points, but I have still only played around with it for educational purposes and definitely wouldn't say it is a good example. 

The scan implementation works similar to `thrust:exclusive_scan` and in my testing on an RTX 2060, it achieves similar performance
The application was profiled with NSight compute and very few significant performance bottlenecks were identified.

Stream Compaction is not currently implemented, as I felt like I have a decent understanding of the work and process necessary and I'd like to move on with more interesting stuff that depends on these algorithms. I may implement compaction in the future if I need a small refresher of CUDA. It is advised that `thrust` should be used for any production applications of these algorithms.
