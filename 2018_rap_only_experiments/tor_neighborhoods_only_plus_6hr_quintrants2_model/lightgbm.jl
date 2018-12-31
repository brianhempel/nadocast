ENV["LIGHTGBM_PATH"] = "/Users/brian/Documents/open_source/LightGBM"

import LightGBM


USE_NORMALIZING_FACTORS = false


# LGBMBinary(; num_iterations = 10,
#                     learning_rate = .1,
#                     num_leaves = 127,
#                     max_depth = -1,
#                     tree_learner = "serial",
#                     num_threads = Sys.CPU_CORES,
#                     histogram_pool_size = -1.,
#                     min_data_in_leaf = 100,
#                     min_sum_hessian_in_leaf = 10.,
#                     lambda_l1 = 0.,
#                     lambda_l2 = 0.,
#                     min_gain_to_split = 0.,
#                     feature_fraction = 1.,
#                     feature_fraction_seed = 2,
#                     bagging_fraction = 1.,
#                     bagging_freq = 0,
#                     bagging_seed = 3,
#                     early_stopping_round = 0,
#                     max_bin = 255,
#                     data_random_seed = 1,
#                     init_score = "",
#                     is_sparse = true,
#                     save_binary = false,
#                     categorical_feature = Int[],
#                     sigmoid = 1.,
#                     is_unbalance = false,
#                     metric = ["binary_logloss"],
#                     metric_freq = 1,
#                     is_training_metric = false,
#                     ndcg_at = Int[],
#                     num_machines = 1,
#                     local_listen_port = 12400,
#                     time_out = 120,
#                     machine_list_file = "")

estimator =
  LightGBM.LGBMBinary(
    num_iterations = 2000,
    min_data_in_leaf = 1000, # 1000 did better than 2000 or 5000, 500 is too small
    # min_sum_hessian_in_leaf = 1000,
    learning_rate = .02, #
    early_stopping_round = 50,
    feature_fraction = .8, # .9 – 0.06654782621229002; .8 – 0.06607885426312127; .7 — 0.06638090709783184
    bagging_fraction = .8, #
    bagging_freq = 10,
    num_leaves = 25, # 15 is too small, 63 is too much; 21 - 0.06660153, 25 - 0.06607885426312127, 28 - 0.06638649215921182, 31 - 0.0661989, 40 - 0.066334
    is_sparse = false,
    max_bin = 255,
    num_threads = 4,
    is_unbalance = false, # if true, changes the label weights so pos/neg classes have same total weight across dataset when computing loss/gradient; causes WAAY too much probability all over
    metric = ["binary_logloss"]
  )


function model_predict_all(X)
  global estimator
  LightGBM.predict(estimator, X; is_row_major = true)
end

function model_prediction(x)
  global estimator
  LightGBM.predict(estimator, collect(reshape(x, (length(x),1))))[1]
end

function model_save(model_path)
  global estimator
  LightGBM.savemodel(estimator, model_path)
end


function model_load(model_path)
  global estimator
  LightGBM.loadmodel(estimator, model_path)
end
