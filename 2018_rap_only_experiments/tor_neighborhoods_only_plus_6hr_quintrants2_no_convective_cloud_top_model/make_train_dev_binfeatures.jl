# $ julia tor_neighborhoods_only_plus_6hr_quintrants2_no_convective_cloud_top_model/make_train_dev_binfeatures.jl

MODEL_DIR     = "tor_neighborhoods_only_plus_6hr_quintrants2_model"
NEW_MODEL_DIR = "tor_neighborhoods_only_plus_6hr_quintrants2_no_convective_cloud_top_model"

non_convective_cloud_top_features_is = [collect(1:289); collect(299:1711)]

include("../ReadFeatureData.jl")


dev_file_path      = "$MODEL_DIR/dev.binfeatures"
train_file_path    = "$MODEL_DIR/train.binfeatures"

dev_data,   dev_data_file,   dev_point_count   = open_data_file(dev_file_path)
println("$dev_point_count dev points.")
new_dev_file = open("$NEW_MODEL_DIR/dev.binfeatures", "w")
for i = 1:dev_point_count
  write(new_dev_file, dev_data[non_convective_cloud_top_features_is,i])
end
close(new_dev_file)
close(dev_data_file)

train_data, train_data_file, train_point_count = open_data_file(train_file_path)
println("$train_point_count training points.")
new_train_file = open("$NEW_MODEL_DIR/train.binfeatures", "w")
for i = 1:train_point_count
  write(new_train_file, train_data[non_convective_cloud_top_features_is,i])
end
close(new_train_file)
close(train_data_file)
