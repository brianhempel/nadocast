module PCA


# X = [-.1 -.2; 0 0; .1 .2]
# 3×2 Matrix{Float64}:
#  -0.1  -0.2
#   0.0   0.0
#   0.1   0.2

# X * transpose(svd(X).Vt)
# 3×2 Matrix{Float64}:
#  -0.223607   4.9374e-17
#   0.0        0.0
#   0.223607  -4.9374e-17

# X * transpose(svd(X).Vt)[:,1:1]
# 3×1 Matrix{Float64}:
#  -0.22360679774997896
#   0.0
#   0.22360679774997896

# X has one datapoint in each row
function pca(X; components = size(X,2))
  # X .- mapslices(sum, X, dims = [1]) ./ size(X,1)
  transpose(svd(X).Vt)[:, 1:components]
end

function apply_pca(X, pca)
  X * pca
end

# Subtract so the mean of each row is 0.0
function recenter_rows(X)
  X .- mapslices(sum, X, dims = [1]) ./ size(X,1)
end


end # module PCA