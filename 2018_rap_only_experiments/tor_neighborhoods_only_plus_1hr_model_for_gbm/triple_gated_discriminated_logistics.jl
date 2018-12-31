println("Loading Flux...")
using Flux
using BSON: @load

USE_NORMALIZING_FACTORS = true
DO_BACKPROP             = true

# Best (non-gated) seemed to be 3 splits 0.05916296851657632 bad dev loss, 0.0627 corrected, on epoch 5
# To beat is still the 4factor logistic with 0.05993 corrected

# Dev loss: (0.07002153384332845, 0.2521149542468677)
# 400 HLCY:3000-0 m above ground:storm path max  Float32[-167.302, 337.279, 1497.21]
# 360 HGT:convective cloud top level:storm path 50mi mean  Float32[-8740.75, 3332.81, 15295.1]
# 526 REFD:4000 m above ground:storm path max  Float32[-10.0, 34.5, 46.375]

const DIM1                    = 400 # 400 HLCY:3000-0 m above ground:storm path max  Float32[-167.302, 337.279, 1497.21]
const DIM1_NORMALIZING_FACTOR = 1566.5924f0
const DIM1_SPLIT              = 337.279f0 / DIM1_NORMALIZING_FACTOR
const DIM2                    = 360 # 360 HGT:convective cloud top level:storm path 50mi mean  Float32[-8740.75, 3332.81, 15295.1]
const DIM2_NORMALIZING_FACTOR = 13084.946f0
const DIM2_SPLIT              = 3332.81f0 / DIM2_NORMALIZING_FACTOR
const DIM3                    = 526 # 526 REFD:4000 m above ground:storm path max  Float32[-10.0, 34.5, 46.375]
const DIM3_NORMALIZING_FACTOR = 52.124996f0
const DIM3_SPLIT              = 34.5f0 / DIM3_NORMALIZING_FACTOR
# const DIM4                    = 415 # 415 LFTX:500-1000 mb:point
# const DIM4_NORMALIZING_FACTOR = 30.053402f0
# const DIM4_SPLIT              = -4.80626f0 / DIM4_NORMALIZING_FACTOR
# const DIM5                    = 1278 # VVEL:60-30 mb above ground:storm path 50mi mean
# const DIM5_NORMALIZING_FACTOR = 1.8872524f0
# const DIM5_SPLIT              = -0.186422f0 / DIM5_NORMALIZING_FACTOR
# const DIM6                    = 1168 # 1168 VVEL:180-150 mb above ground:storm path max
# const DIM6_NORMALIZING_FACTOR = 17.768608f0
# const DIM6_SPLIT              = 1.19261f0 / DIM6_NORMALIZING_FACTOR


# Dev loss: (0.0676323137018803, 0.2599611631487407)
# VVEL:60-30 mb above ground:storm path 50mi mean  Float32[-0.186422]
# LFTX:500-1000 mb:point  Float32[-4.80626]
# HLCY:3000-0 m above ground:storm path max  Float32[337.279]
# HGT:convective cloud top level:storm path 50mi mean  Float32[3332.81]
# PRATE:surface:storm path max  Float32[0.002]

# (1168, 1278, 415, 400, 360, 526)
# Dev loss: (0.0675831967318204, 0.17930936899021016)
# VVEL:180-150 mb above ground:storm path max  Float32[1.19261]
# VVEL:60-30 mb above ground:storm path 50mi mean  Float32[-0.186422]
# LFTX:500-1000 mb:point  Float32[-4.80626]
# HLCY:3000-0 m above ground:storm path max  Float32[337.279]
# HGT:convective cloud top level:storm path 50mi mean  Float32[3332.81]
# REFD:4000 m above ground:storm path max  Float32[34.5]


# const DIM1                    = 400 # 400 HLCY:3000-0 m above ground:storm path max  Float32[-167.302, 337.279, 1497.21]
# const DIM1_NORMALIZING_FACTOR = 1566.5924f0
# const DIM1_SPLIT              = 337.279f0 / DIM1_NORMALIZING_FACTOR
# const DIM2                    = 360 # 360 HGT:convective cloud top level:storm path 50mi mean  Float32[-8740.75, 3332.81, 15295.1]
# const DIM2_NORMALIZING_FACTOR = 13084.946f0
# const DIM2_SPLIT              = 3332.81f0 / DIM2_NORMALIZING_FACTOR
# const DIM3                    = 526 # 526 REFD:4000 m above ground:storm path max  Float32[-10.0, 34.5, 46.375]
# const DIM3_NORMALIZING_FACTOR = 52.124996f0
# const DIM3_SPLIT              = 34.5f0 / DIM3_NORMALIZING_FACTOR
# const DIM4                    = 415 # 415 LFTX:500-1000 mb:point
# const DIM4_NORMALIZING_FACTOR = 30.053402f0
# const DIM4_SPLIT              = -4.80626f0 / DIM4_NORMALIZING_FACTOR
# const DIM5                    = 1278 # VVEL:60-30 mb above ground:storm path 50mi mean
# const DIM5_NORMALIZING_FACTOR = 1.8872524f0
# const DIM5_SPLIT              = -0.186422f0 / DIM5_NORMALIZING_FACTOR
# const DIM6                    = 1168 # 1168 VVEL:180-150 mb above ground:storm path max
# const DIM6_NORMALIZING_FACTOR = 17.768608f0
# const DIM6_SPLIT              = 1.19261f0 / DIM6_NORMALIZING_FACTOR

struct Logistics{L}
  gate1::L
  gate2::L
  gate3::L
  logistic000::L
  logistic001::L
  logistic010::L
  logistic011::L
  logistic100::L
  logistic101::L
  logistic110::L
  logistic111::L
end

Flux.treelike(Logistics)

function (a::Logistics)(x)
  # global normalizing_factors
  # sbcape = x[32] * normalizing_factors[32] # storm path mean

  dim1 = x[DIM1]
  dim2 = x[DIM2]
  dim3 = x[DIM3]

  y =
    a.gate1(x) .*
    a.gate2(x) .*
    a.gate3(x) .*
      if dim1 < DIM1_SPLIT
        if dim2 < DIM2_SPLIT
          if dim3 < DIM3_SPLIT
            a.logistic000(x)
          else
            a.logistic001(x)
          end
        else
          if dim3 < DIM3_SPLIT
            a.logistic010(x)
          else
            a.logistic011(x)
          end
        end
      else
        if dim2 < DIM2_SPLIT
          if dim3 < DIM3_SPLIT
            a.logistic100(x)
          else
            a.logistic101(x)
          end
        else
          if dim3 < DIM3_SPLIT
            a.logistic110(x)
          else
            a.logistic111(x)
          end
        end
      end

  y[1]
end

model =
  Logistics(
    Dense(FEATURE_COUNT, 1, σ),
    Dense(FEATURE_COUNT, 1, σ),
    Dense(FEATURE_COUNT, 1, σ),
    Dense(FEATURE_COUNT, 1, σ),
    Dense(FEATURE_COUNT, 1, σ),
    Dense(FEATURE_COUNT, 1, σ),
    Dense(FEATURE_COUNT, 1, σ),
    Dense(FEATURE_COUNT, 1, σ),
    Dense(FEATURE_COUNT, 1, σ),
    Dense(FEATURE_COUNT, 1, σ),
    Dense(FEATURE_COUNT, 1, σ)
  )

points_per_epoch = 5156235 # Somewhat rough since we are droping some examples randomly.

# β2 is the memory of the variance.
optimizer = ADAM(params(model), 10.0 / points_per_epoch; β1 = 0.9, β2 = 0.99999, ϵ = 1e-07)
# optimizer = SGD(params(model), 1.0 / 40000)

loss_func = Flux.binarycrossentropy

function model_prediction(x)
  global model
  Flux.Tracker.data(model(x))
end


function show_extra_training_info()
  nothing
end

function model_load(saved_bson_path)
  global model
  global optimizer
  @load saved_bson_path model optimizer
end
