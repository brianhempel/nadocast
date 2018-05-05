println("Loading Flux...")
using Flux
using BSON: @load

USE_NORMALIZING_FACTORS = true
DO_BACKPROP             = true

# Best seemed to be 3 splits 0.05916296851657632 bad dev loss, 0.0627 corrected, on epoch 5

# Dev loss: (0.07002153384332845, 0.2521149542468677)
# 400 HLCY:3000-0 m above ground:storm path max  Float32[-167.302, 337.279, 1497.21]
# 360 HGT:convective cloud top level:storm path 50mi mean  Float32[-8740.75, 3332.81, 15295.1]
# 526 REFD:4000 m above ground:storm path max  Float32[-10.0, 34.5, 46.375]

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


const DIM1                    = 400 # 400 HLCY:3000-0 m above ground:storm path max  Float32[-167.302, 337.279, 1497.21]
const DIM1_NORMALIZING_FACTOR = 1566.5924f0
const DIM1_SPLIT              = 337.279f0 / DIM1_NORMALIZING_FACTOR
const DIM2                    = 360 # 360 HGT:convective cloud top level:storm path 50mi mean  Float32[-8740.75, 3332.81, 15295.1]
const DIM2_NORMALIZING_FACTOR = 13084.946f0
const DIM2_SPLIT              = 3332.81f0 / DIM2_NORMALIZING_FACTOR
const DIM3                    = 526 # 526 REFD:4000 m above ground:storm path max  Float32[-10.0, 34.5, 46.375]
const DIM3_NORMALIZING_FACTOR = 52.124996f0
const DIM3_SPLIT              = 34.5f0 / DIM3_NORMALIZING_FACTOR
const DIM4                    = 415 # 415 LFTX:500-1000 mb:point
const DIM4_NORMALIZING_FACTOR = 30.053402f0
const DIM4_SPLIT              = -4.80626f0 / DIM4_NORMALIZING_FACTOR
const DIM5                    = 1278 # VVEL:60-30 mb above ground:storm path 50mi mean
const DIM5_NORMALIZING_FACTOR = 1.8872524f0
const DIM5_SPLIT              = -0.186422f0 / DIM5_NORMALIZING_FACTOR
const DIM6                    = 1168 # 1168 VVEL:180-150 mb above ground:storm path max
const DIM6_NORMALIZING_FACTOR = 17.768608f0
const DIM6_SPLIT              = 1.19261f0 / DIM6_NORMALIZING_FACTOR

struct Logistics{L}
  logistic000000::L
  logistic000001::L
  logistic000010::L
  logistic000011::L
  logistic000100::L
  logistic000101::L
  logistic000110::L
  logistic000111::L
  logistic001000::L
  logistic001001::L
  logistic001010::L
  logistic001011::L
  logistic001100::L
  logistic001101::L
  logistic001110::L
  logistic001111::L
  logistic010000::L
  logistic010001::L
  logistic010010::L
  logistic010011::L
  logistic010100::L
  logistic010101::L
  logistic010110::L
  logistic010111::L
  logistic011000::L
  logistic011001::L
  logistic011010::L
  logistic011011::L
  logistic011100::L
  logistic011101::L
  logistic011110::L
  logistic011111::L
  logistic100000::L
  logistic100001::L
  logistic100010::L
  logistic100011::L
  logistic100100::L
  logistic100101::L
  logistic100110::L
  logistic100111::L
  logistic101000::L
  logistic101001::L
  logistic101010::L
  logistic101011::L
  logistic101100::L
  logistic101101::L
  logistic101110::L
  logistic101111::L
  logistic110000::L
  logistic110001::L
  logistic110010::L
  logistic110011::L
  logistic110100::L
  logistic110101::L
  logistic110110::L
  logistic110111::L
  logistic111000::L
  logistic111001::L
  logistic111010::L
  logistic111011::L
  logistic111100::L
  logistic111101::L
  logistic111110::L
  logistic111111::L
end

Flux.treelike(Logistics)

function (a::Logistics)(x)
  # global normalizing_factors
  # sbcape = x[32] * normalizing_factors[32] # storm path mean

  dim1 = x[DIM1]
  dim2 = x[DIM2]
  dim3 = x[DIM3]
  dim4 = x[DIM4]
  dim5 = x[DIM5]
  dim6 = x[DIM6]

  y =
    if dim1 < DIM1_SPLIT
      if dim2 < DIM2_SPLIT
        if dim3 < DIM3_SPLIT
          if dim4 < DIM4_SPLIT
            if dim5 < DIM5_SPLIT
              if dim6 < DIM6_SPLIT
                a.logistic000000(x)
              else
                a.logistic000001(x)
              end
            else
              if dim6 < DIM6_SPLIT
                a.logistic000010(x)
              else
                a.logistic000011(x)
              end
            end
          else
            if dim5 < DIM5_SPLIT
              if dim6 < DIM6_SPLIT
                a.logistic000100(x)
              else
                a.logistic000101(x)
              end
            else
              if dim6 < DIM6_SPLIT
                a.logistic000110(x)
              else
                a.logistic000111(x)
              end
            end
          end
        else
          if dim4 < DIM4_SPLIT
            if dim5 < DIM5_SPLIT
              if dim6 < DIM6_SPLIT
                a.logistic001000(x)
              else
                a.logistic001001(x)
              end
            else
              if dim6 < DIM6_SPLIT
                a.logistic001010(x)
              else
                a.logistic001011(x)
              end
            end
          else
            if dim5 < DIM5_SPLIT
              if dim6 < DIM6_SPLIT
                a.logistic001100(x)
              else
                a.logistic001101(x)
              end
            else
              if dim6 < DIM6_SPLIT
                a.logistic001110(x)
              else
                a.logistic001111(x)
              end
            end
          end
        end
      else
        if dim3 < DIM3_SPLIT
          if dim4 < DIM4_SPLIT
            if dim5 < DIM5_SPLIT
              if dim6 < DIM6_SPLIT
                a.logistic010000(x)
              else
                a.logistic010001(x)
              end
            else
              if dim6 < DIM6_SPLIT
                a.logistic010010(x)
              else
                a.logistic010011(x)
              end
            end
          else
            if dim5 < DIM5_SPLIT
              if dim6 < DIM6_SPLIT
                a.logistic010100(x)
              else
                a.logistic010101(x)
              end
            else
              if dim6 < DIM6_SPLIT
                a.logistic010110(x)
              else
                a.logistic010111(x)
              end
            end
          end
        else
          if dim4 < DIM4_SPLIT
            if dim5 < DIM5_SPLIT
              if dim6 < DIM6_SPLIT
                a.logistic011000(x)
              else
                a.logistic011001(x)
              end
            else
              if dim6 < DIM6_SPLIT
                a.logistic011010(x)
              else
                a.logistic011011(x)
              end
            end
          else
            if dim5 < DIM5_SPLIT
              if dim6 < DIM6_SPLIT
                a.logistic011100(x)
              else
                a.logistic011101(x)
              end
            else
              if dim6 < DIM6_SPLIT
                a.logistic011110(x)
              else
                a.logistic011111(x)
              end
            end
          end
        end
      end
    else
      if dim2 < DIM2_SPLIT
        if dim3 < DIM3_SPLIT
          if dim4 < DIM4_SPLIT
            if dim5 < DIM5_SPLIT
              if dim6 < DIM6_SPLIT
                a.logistic100000(x)
              else
                a.logistic100001(x)
              end
            else
              if dim6 < DIM6_SPLIT
                a.logistic100010(x)
              else
                a.logistic100011(x)
              end
            end
          else
            if dim5 < DIM5_SPLIT
              if dim6 < DIM6_SPLIT
                a.logistic100100(x)
              else
                a.logistic100101(x)
              end
            else
              if dim6 < DIM6_SPLIT
                a.logistic100110(x)
              else
                a.logistic100111(x)
              end
            end
          end
        else
          if dim4 < DIM4_SPLIT
            if dim5 < DIM5_SPLIT
              if dim6 < DIM6_SPLIT
                a.logistic101000(x)
              else
                a.logistic101001(x)
              end
            else
              if dim6 < DIM6_SPLIT
                a.logistic101010(x)
              else
                a.logistic101011(x)
              end
            end
          else
            if dim5 < DIM5_SPLIT
              if dim6 < DIM6_SPLIT
                a.logistic101100(x)
              else
                a.logistic101101(x)
              end
            else
              if dim6 < DIM6_SPLIT
                a.logistic101110(x)
              else
                a.logistic101111(x)
              end
            end
          end
        end
      else
        if dim3 < DIM3_SPLIT
          if dim4 < DIM4_SPLIT
            if dim5 < DIM5_SPLIT
              if dim6 < DIM6_SPLIT
                a.logistic110000(x)
              else
                a.logistic110001(x)
              end
            else
              if dim6 < DIM6_SPLIT
                a.logistic110010(x)
              else
                a.logistic110011(x)
              end
            end
          else
            if dim5 < DIM5_SPLIT
              if dim6 < DIM6_SPLIT
                a.logistic110100(x)
              else
                a.logistic110101(x)
              end
            else
              if dim6 < DIM6_SPLIT
                a.logistic110110(x)
              else
                a.logistic110111(x)
              end
            end
          end
        else
          if dim4 < DIM4_SPLIT
            if dim5 < DIM5_SPLIT
              if dim6 < DIM6_SPLIT
                a.logistic111000(x)
              else
                a.logistic111001(x)
              end
            else
              if dim6 < DIM6_SPLIT
                a.logistic111010(x)
              else
                a.logistic111011(x)
              end
            end
          else
            if dim5 < DIM5_SPLIT
              if dim6 < DIM6_SPLIT
                a.logistic111100(x)
              else
                a.logistic111101(x)
              end
            else
              if dim6 < DIM6_SPLIT
                a.logistic111110(x)
              else
                a.logistic111111(x)
              end
            end
          end
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
