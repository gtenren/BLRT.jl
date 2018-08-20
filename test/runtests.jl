using BLRT
using Base.Test


# Load Mutagenesis2 dataset (X,y)
const LABEL = 1
const BAGID = 2
const FEATURES = 3:9

D = readdlm("./Mutagenesis2.csv", ',', Float32)

numofbags = Int(maximum(D[:, BAGID]))

X = [Vector{Vector{Float32}}() for bb in 1:numofbags]
y = Vector{Bool}(numofbags)

for ii in 1:size(D, 1)
    bagid = Int(D[ii, BAGID])
    y[bagid] = D[ii, LABEL] == 1.0
    push!(X[bagid], D[ii, FEATURES])
end

X = map(bag -> hcat(bag...)', X)


# Fit model
model = train(X, y)
scores = classify(model, X)
acc = mean(y .== (scores .> 0.5))


# Test accuracy
@test acc > 0.9
