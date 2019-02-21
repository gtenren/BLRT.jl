# Bag-level Randomized Trees (BLRT.jl)
[![Build Status](https://travis-ci.org/komartom/BLRT.jl.svg?branch=master)](https://travis-ci.org/komartom/BLRT.jl)
[![codecov.io](http://codecov.io/github/komartom/BLRT.jl/coverage.svg?branch=master)](http://codecov.io/github/komartom/BLRT.jl?branch=master)

Multiple Instance Classifier in Julia

[Komárek T., Somol P. (2019) Multiple Instance Learning with Bag-Level Randomized Trees. In: Machine Learning and Knowledge Discovery in Databases. ECML PKDD 2018. Lecture Notes in Computer Science, vol 11051. Springer, Cham](https://doi.org/10.1007/978-3-030-10925-7_16)

## Installation
You can install the classifier using Julia's package manager
```julia
] add https://github.com/komartom/BLRT.jl.git
```

## Classification example
5-times repeated 10-fold cross-validation on Musk1 dataset
```julia
using BLRT, ROCAnalysis, Printf, Statistics

# https://github.com/komartom/MIDatasets.jl.git
using MIDatasets

# Load Musk1 dataset with cross-validation indexes
X, y, folds = midataset("Musk1", folds=true)

AUCs = Matrix{AbstractFloat}(undef, 10, 5)

for rr in 1:5, ff in 1:10

    Xtrain = X[folds[rr][ff]]
    ytrain = y[folds[rr][ff]]

    Xtest = X[.!folds[rr][ff]]
    ytest = y[.!folds[rr][ff]]

    model = train(Xtrain, ytrain)
    scores = classify(model, Xtest)
    
    AUCs[ff, rr] = auc(roc(scores[.!ytest], scores[ytest]))

    println("Repetition: ", rr, " Fold: ", ff)

end

println(@sprintf("AUC: %0.2f (%0.2f)", mean(mean(AUCs, dims=1)), std(mean(AUCs, dims=1))))
```
