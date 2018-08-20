# Bag-level Randomized Trees (BLRT.jl)
[![Build Status](https://travis-ci.org/komartom/BLRT.jl.svg?branch=master)](https://travis-ci.org/komartom/BLRT.jl)
[![codecov.io](http://codecov.io/github/komartom/BLRT.jl/coverage.svg?branch=master)](http://codecov.io/github/komartom/BLRT.jl?branch=master)

Multiple Instance Classifier in Julia

## Installation
You can install the classifier using Julia's package manager
```julia
Pkg.clone("https://github.com/komartom/BLRT.jl.git")
```

## Classification example
5-times repeated 10-fold cross-validation on Musk1 dataset
```julia
using BLRT

# Pkg.clone("https://github.com/komartom/MIDatasets.jl.git")
using MIDatasets

# Pkg.clone("https://github.com/bcbi/AUC.jl.git")
using AUC

# Load Musk1 dataset with cross-validation indexes
X, y, folds = midataset("Musk1", folds=true)

AUCs = Matrix{AbstractFloat}(10, 5)

for rr in 1:5, ff in 1:10

    Xtrain = X[folds[rr][ff]]
    ytrain = y[folds[rr][ff]]

    Xtest = X[.!folds[rr][ff]]
    ytest = y[.!folds[rr][ff]]

    model = train(Xtrain, ytrain)
    scores = classify(model, Xtest)

    AUCs[ff, rr] = auc(ytest, scores)

    println("R: ", rr, " F: ", ff)

end

println(@sprintf("AUC: %0.2f (%0.2f)", mean(mean(AUCs, 1)), std(mean(AUCs, 1))))
```
