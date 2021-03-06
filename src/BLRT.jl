module BLRT

using Random, Distributed, Statistics

export train, classify
export Options, Rule, Leaf, Split, Node, Model


struct Options

    ntrees::Int
    nsubfeat::Int
    nrules::Int
    minsamples::Int
    maxdepth::Int

    function Options(ntrees::Int, nsubfeat::Int, nrules::Int, minsamples::Int, maxdepth::Int)

        if ntrees < 1
            error("The minimum number of trees to grow should be greater than 0.")
        end

        if nsubfeat < 1
            error("The minimum number of randomly selected features at each split should be greater than 0.")
        end

        if nrules < 1
            error("The minimum number of generated splitting rules should be greater than 0.")
        end

        if minsamples < 1
            error("The minimum number of samples in nodes to stop further splitting should be greater than 0.")
        end

        if maxdepth < -1
            error("The maximum tree-depth should be greater than or equal to -1. Setting the value to -1 corresponds to unlimited tree-depth.")
        end

        new(ntrees, nsubfeat, nrules, minsamples, maxdepth)

    end

end


struct Rule
    featid::Int
    featval::AbstractFloat
    instratio::AbstractFloat
end


struct Leaf
    probability::AbstractFloat
end


struct Split
    rule::Rule
    left::Union{Split, Leaf}
    right::Union{Split, Leaf}
end


const Node = Union{Split, Leaf}


struct Model
    options::Options
    trees::Vector{Node}
    trainingtime::AbstractFloat
    description::AbstractString
end


function (phi::Rule)(bag)

    ninst = 0
    for ii in 1:size(bag, 1)
        if bag[ii, phi.featid] <= phi.featval
            ninst += 1
        end
    end

    (ninst / size(bag, 1)) > phi.instratio

end


function entropyloss(X, y, rule)

    nleftpos = 0
    nleftneg = 0
    nrightpos = 0
    nrightneg = 0

    for bb in 1:length(X)
        if rule(X[bb])
            if y[bb]
                nleftpos += 1
            else
                nleftneg += 1
            end
        else
            if y[bb]
                nrightpos += 1
            else
                nrightneg += 1
            end
        end
    end

    wleft = (nleftpos + nleftneg) / length(X)
    wright = 1.0 - wleft

    nleft = nleftpos + nleftneg
    p1 = (nleft == 0) ? 0.0 : nleftpos / nleft
    p0 = 1.0 - p1
    entropyleft = (p0 == 0.0) ? 0.0 : -p0*log2(p0)
    entropyleft += (p1 == 0.0) ? 0.0 : -p1*log2(p1)

    nright = nrightpos + nrightneg
    p1 = (nright == 0) ? 0.0 : nrightpos / nright
    p0 = 1.0 - p1
    entropyright = (p0 == 0.0) ? 0.0 : -p0*log2(p0)
    entropyright += (p1 == 0.0) ? 0.0 : -p1*log2(p1)

    wleft*entropyleft + wright*entropyright

end


function selectrule(X, y, opt)

    bestloss = Inf
    bestrule = Rule(-1, 0.0, 0.0)

    for ff in shuffle(1:size(X[1], 2))[1:opt.nsubfeat]

        minfeat = typemax(typeof(X[1][1, 1]))
        maxfeat = typemin(typeof(X[1][1, 1]))

        for bag in X
            for ii in 1:size(bag, 1)
                if bag[ii, ff] < minfeat
                    minfeat = bag[ii, ff]
                end
                if bag[ii, ff] > maxfeat
                    maxfeat = bag[ii, ff]
                end
            end
        end

        if maxfeat > minfeat
            range = maxfeat - minfeat
            for rr in 1:opt.nrules
                rule = Rule(ff, rand() * range + minfeat, rand())
                loss = entropyloss(X, y, rule)
                if loss < bestloss
                    bestloss = loss
                    bestrule = rule
                end
            end
        end

    end

    bestrule

end


function divide(X, y, opt, depth=0)

    probability = sum(y) / length(y)

    if opt.maxdepth == depth || probability == 1.0 || probability == 0.0
        return Leaf(probability)
    end

    rule = selectrule(X, y, opt)

    if rule.featid == -1
        return Leaf(probability)
    end

    leftsamples = [rule(bag) for bag in X]
    leftsamplessum = sum(leftsamples)

    if leftsamplessum < opt.minsamples || (length(X) - leftsamplessum) < opt.minsamples
        return Leaf(probability)
    end

    return Split(rule,
        divide(X[leftsamples], y[leftsamples], opt, depth+1),
        divide(X[.!leftsamples], y[.!leftsamples], opt, depth+1)
    )

end


function train(X::Vector{Matrix{T}}, y::AbstractArray{Bool}, opt::Options, description::AbstractString) where T <: AbstractFloat

    @assert length(X) == length(y)

    starttime = time_ns()
    Model(opt, pmap((arg)->divide(X, y, opt), 1:opt.ntrees), (time_ns()-starttime)/10^9, description)

end


function train(X::Vector{Matrix{T}}, y::AbstractArray{Bool}; ntrees::Int=100, nsubfeat::Int=-1, nrules::Int=16, minsamples::Int=1, maxdepth::Int=-1, description::AbstractString="none") where T <: AbstractFloat

    if nsubfeat < 0
        nsubfeat = round(Int, sqrt(size(X[1], 2)))
    end

    train(X, y, Options(ntrees, nsubfeat, nrules, minsamples, maxdepth), description)

end


function classify(node::Node, bag::Matrix{T}) where T <: AbstractFloat

    while isa(node, Split)
        node = node.rule(bag) ? node.left : node.right
    end

    node.probability

end


function classify(model::Model, bag::Matrix{T}) where T <: AbstractFloat

    mean(map((arg)->classify(arg...), [(tree, bag) for tree in model.trees]))

end


function classify(model::Model, bags::Vector{Matrix{T}}) where T <: AbstractFloat

    [classify(model, bag) for bag in bags]

end

end # module
