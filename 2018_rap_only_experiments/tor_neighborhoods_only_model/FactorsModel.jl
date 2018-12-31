struct Factors{L,M}
  f1::L
  fs::Array{L}
  bs::M
end

MakeFactors(f1, fs, bs) = Factors(f1, fs, param(bs))

Flux.treelike(Factors)

function (a::Factors)(x)
  f1, fs, bs = a.f1, a.fs, a.bs

  y1 = f1(x)

  ys = map(i -> bs[i] + (1.0 - bs[i]) * fs[i](x)[1], 1:length(bs))
  # ys = bs .+ (1.0 - bs) .* map(f -> , fs)

  y1 * prod(ys)
end
