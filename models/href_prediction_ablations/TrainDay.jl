import Dates
import Printf

push!(LOAD_PATH, (@__DIR__) * "/../shared")
# import TrainGBDTShared
import TrainingShared
import LogisticRegression
using Metrics

push!(LOAD_PATH, @__DIR__)
import HREFPredictionAblations

push!(LOAD_PATH, (@__DIR__) * "/../../lib")
import Forecasts
import Inventories
import StormEvents

MINUTE = 60 # seconds
HOUR   = 60*MINUTE

(_, validation_forecasts, _) = TrainingShared.forecasts_train_validation_test(HREFPredictionAblations.forecasts_day_accumulators(); just_hours_near_storm_events = false);

# We don't have storm events past this time.
cutoff = Dates.DateTime(2022, 6, 1, 12)
validation_forecasts = filter(forecast -> Forecasts.valid_utc_datetime(forecast) < cutoff, validation_forecasts);

length(validation_forecasts) # 716

validation_forecasts_0z_12z = filter(forecast -> forecast.run_hour == 0 || forecast.run_hour == 12, validation_forecasts);
length(validation_forecasts_0z_12z) # 358

@time Forecasts.data(validation_forecasts[10]); # Check if a forecast loads


# const ε = 1e-15 # Smallest Float64 power of 10 you can add to 1.0 and not round off to 1.0
const ε = 1f-7 # Smallest Float32 power of 10 you can add to 1.0 and not round off to 1.0
logloss(y, ŷ) = -y*log(ŷ + ε) - (1.0f0 - y)*log(1.0f0 - ŷ + ε)

σ(x) = 1.0f0 / (1.0f0 + exp(-x))

logit(p) = log(p / (one(p) - p))

event_name_to_day_labeler = Dict(
  "tornado" => TrainingShared.event_name_to_day_labeler["tornado"]
)

# rm("day_accumulators_validation_forecasts_0z_12z"; recursive = true)

X, Ys, weights =
  TrainingShared.get_data_labels_weights(
    validation_forecasts_0z_12z;
    event_name_to_labeler = event_name_to_day_labeler,
    save_dir = "day_accumulators_validation_forecasts_0z_12z",
  );



# should do some checks here.
import PlotMap

dec11 = filter(f -> Forecasts.time_title(f) == "2021-12-11 00Z +35", validation_forecasts_0z_12z)[1];
dec11_data = Forecasts.data(dec11);
for i in 1:size(dec11_data,2)
  prediction_i = div(i - 1, 2) + 1
  model_name, _ = HREFPredictionAblations.models[prediction_i]
  PlotMap.plot_debug_map("dec11_0z_12z_day_accs_$(i)_$model_name", dec11.grid, dec11_data[:,i]);
end
PlotMap.plot_debug_map("dec11_0z_12z_day_tornado", dec11.grid, event_name_to_day_labeler["tornado"](dec11));

# scp nadocaster2:/home/brian/nadocast_dev/models/href_prediction_ablations/dec11_0z_12z_day_accs_1.pdf ./
# scp nadocaster2:/home/brian/nadocast_dev/models/href_prediction_ablations/dec11_0z_12z_day_tornado.pdf ./


# Confirm that the accs are better than the maxes
function test_predictive_power(forecasts, X, Ys, weights)
  inventory = Forecasts.inventory(forecasts[1])

  for feature_i in 1:length(inventory)
    prediction_i = div(feature_i - 1, 2) + 1
    model_name, _ = HREFPredictionAblations.models[prediction_i]
    y = Ys["tornado"]
    x = @view X[:,feature_i]
    au_pr_curve = Metrics.area_under_pr_curve(x, y, weights)
    println("$model_name ($(sum(y))) feature $feature_i $(Inventories.inventory_line_description(inventory[feature_i]))\tAU-PR-curve: $au_pr_curve")
  end
end
test_predictive_power(validation_forecasts_0z_12z, X, Ys, weights)

# tornado_mean_58                                           (20606.0) feature 1 independent events total  TORPROB:calculated:day fcst:: AU-PR-curve: 0.13162558944528638
# tornado_mean_58                                           (20606.0) feature 2 highest hourly            TORPROB:calculated:day fcst:: AU-PR-curve: 0.12449276362093863
# tornado_prob_80                                           (20606.0) feature 3 independent events total  TORPROB:calculated:day fcst:: AU-PR-curve: 0.13009477393713162
# tornado_prob_80                                           (20606.0) feature 4 highest hourly            TORPROB:calculated:day fcst:: AU-PR-curve: 0.11937844108793211
# tornado_mean_prob_138                                     (20606.0) feature 5 independent events total  TORPROB:calculated:day fcst:: AU-PR-curve: 0.13792107012674604
# tornado_mean_prob_138                                     (20606.0) feature 6 highest hourly            TORPROB:calculated:day fcst:: AU-PR-curve: 0.13220794055741025
# tornado_mean_prob_computed_no_sv_219                      (20606.0) feature 7 independent events total  TORPROB:calculated:day fcst:: AU-PR-curve: 0.1369357573174645
# tornado_mean_prob_computed_no_sv_219                      (20606.0) feature 8 highest hourly            TORPROB:calculated:day fcst:: AU-PR-curve: 0.13028198157841184
# tornado_mean_prob_computed_220                            (20606.0) feature 9 independent events total  TORPROB:calculated:day fcst:: AU-PR-curve: 0.1358151410361583
# tornado_mean_prob_computed_220                            (20606.0) feature 10 highest hourly           TORPROB:calculated:day fcst:: AU-PR-curve: 0.13025419582526923
# tornado_mean_prob_computed_partial_climatology_227        (20606.0) feature 11 independent events total TORPROB:calculated:day fcst:: AU-PR-curve: 0.14589679961936527
# tornado_mean_prob_computed_partial_climatology_227        (20606.0) feature 12 highest hourly           TORPROB:calculated:day fcst:: AU-PR-curve: 0.13918545049521447
# tornado_mean_prob_computed_climatology_253                (20606.0) feature 13 independent events total TORPROB:calculated:day fcst:: AU-PR-curve: 0.1476615184683567
# tornado_mean_prob_computed_climatology_253                (20606.0) feature 14 highest hourly           TORPROB:calculated:day fcst:: AU-PR-curve: 0.13952586413567675
# tornado_mean_prob_computed_climatology_blurs_910          (20606.0) feature 15 independent events total TORPROB:calculated:day fcst:: AU-PR-curve: 0.14386786713427835
# tornado_mean_prob_computed_climatology_blurs_910          (20606.0) feature 16 highest hourly           TORPROB:calculated:day fcst:: AU-PR-curve: 0.1358966592876078
# tornado_mean_prob_computed_climatology_grads_1348         (20606.0) feature 17 independent events total TORPROB:calculated:day fcst:: AU-PR-curve: 0.14783769477377198
# tornado_mean_prob_computed_climatology_grads_1348         (20606.0) feature 18 highest hourly           TORPROB:calculated:day fcst:: AU-PR-curve: 0.1396541560734088
# tornado_mean_prob_computed_climatology_blurs_grads_2005   (20606.0) feature 19 independent events total TORPROB:calculated:day fcst:: AU-PR-curve: 0.15169494866457822
# tornado_mean_prob_computed_climatology_blurs_grads_2005   (20606.0) feature 20 highest hourly           TORPROB:calculated:day fcst:: AU-PR-curve: 0.1420868134833007
# tornado_mean_prob_computed_climatology_prior_next_hrs_691 (20606.0) feature 21 independent events total TORPROB:calculated:day fcst:: AU-PR-curve: 0.15109026383083607
# tornado_mean_prob_computed_climatology_prior_next_hrs_691 (20606.0) feature 22 highest hourly           TORPROB:calculated:day fcst:: AU-PR-curve: 0.14853730824111055
# tornado_mean_prob_computed_climatology_3hr_1567           (20606.0) feature 23 independent events total TORPROB:calculated:day fcst:: AU-PR-curve: 0.15952592485504877 ***best**
# tornado_mean_prob_computed_climatology_3hr_1567           (20606.0) feature 24 highest hourly           TORPROB:calculated:day fcst:: AU-PR-curve: 0.15389499088128772 (second best)
# tornado_full_13831                                        (20606.0) feature 25 independent events total TORPROB:calculated:day fcst:: AU-PR-curve: 0.15398281422151097 (third best)
# tornado_full_13831                                        (20606.0) feature 26 highest hourly           TORPROB:calculated:day fcst:: AU-PR-curve: 0.14738984560906757


# event_names is list of keys of Ys, one per column in Ŷ
Metrics.reliability_curves_midpoints(20, X, Ys, fill("tornado", size(X,2)), weights, map(i -> HREFPredictionAblations.models[div(i - 1, 2) + 1][1] * (isodd(i) ? "_tot" : "_max"), 1:size(X,2)))
# ŷ_tornado_mean_58_tot	y_tornado_mean_58_tot	ŷ_tornado_mean_58_max	y_tornado_mean_58_max	ŷ_tornado_prob_80_tot	y_tornado_prob_80_tot	ŷ_tornado_prob_80_max	y_tornado_prob_80_max	ŷ_tornado_mean_prob_138_tot	y_tornado_mean_prob_138_tot	ŷ_tornado_mean_prob_138_max	y_tornado_mean_prob_138_max	ŷ_tornado_mean_prob_computed_no_sv_219_tot	y_tornado_mean_prob_computed_no_sv_219_tot	ŷ_tornado_mean_prob_computed_no_sv_219_max	y_tornado_mean_prob_computed_no_sv_219_max	ŷ_tornado_mean_prob_computed_220_tot	y_tornado_mean_prob_computed_220_tot	ŷ_tornado_mean_prob_computed_220_max	y_tornado_mean_prob_computed_220_max	ŷ_tornado_mean_prob_computed_partial_climatology_227_tot	y_tornado_mean_prob_computed_partial_climatology_227_tot	ŷ_tornado_mean_prob_computed_partial_climatology_227_max	y_tornado_mean_prob_computed_partial_climatology_227_max	ŷ_tornado_mean_prob_computed_climatology_253_tot	y_tornado_mean_prob_computed_climatology_253_tot	ŷ_tornado_mean_prob_computed_climatology_253_max	y_tornado_mean_prob_computed_climatology_253_max	ŷ_tornado_mean_prob_computed_climatology_blurs_910_tot	y_tornado_mean_prob_computed_climatology_blurs_910_tot	ŷ_tornado_mean_prob_computed_climatology_blurs_910_max	y_tornado_mean_prob_computed_climatology_blurs_910_max	ŷ_tornado_mean_prob_computed_climatology_grads_1348_tot	y_tornado_mean_prob_computed_climatology_grads_1348_tot	ŷ_tornado_mean_prob_computed_climatology_grads_1348_max	y_tornado_mean_prob_computed_climatology_grads_1348_max	ŷ_tornado_mean_prob_computed_climatology_blurs_grads_2005_tot	y_tornado_mean_prob_computed_climatology_blurs_grads_2005_tot	ŷ_tornado_mean_prob_computed_climatology_blurs_grads_2005_max	y_tornado_mean_prob_computed_climatology_blurs_grads_2005_max	ŷ_tornado_mean_prob_computed_climatology_prior_next_hrs_691_tot	y_tornado_mean_prob_computed_climatology_prior_next_hrs_691_tot	ŷ_tornado_mean_prob_computed_climatology_prior_next_hrs_691_max	y_tornado_mean_prob_computed_climatology_prior_next_hrs_691_max	ŷ_tornado_mean_prob_computed_climatology_3hr_1567_tot	y_tornado_mean_prob_computed_climatology_3hr_1567_tot	ŷ_tornado_mean_prob_computed_climatology_3hr_1567_max	y_tornado_mean_prob_computed_climatology_3hr_1567_max	ŷ_tornado_full_13831_tot	y_tornado_full_13831_tot	ŷ_tornado_full_13831_max	y_tornado_full_13831_max
# 0.00014624637	0.000121510224	2.7186386E-05	0.00012168855	0.0001854611	0.00012027176	3.9926257E-05	0.00012038185	0.00015254003	0.00012020319	3.0640716E-05	0.00012050523	0.00015608326	0.000120115066	3.0643445E-05	0.000120245146	0.00015045531	0.00012007886	3.0579577E-05	0.000120202996	0.00015448309	0.00011949108	3.035024E-05	0.00011974352	0.00015304616	0.00011934258	2.9291072E-05	0.00011971917	0.00014371765	0.00011854989	2.5485231E-05	0.00011913049	0.000148418	0.00011869596	3.1260675E-05	0.00011843727	0.00013809305	0.00011872367	2.6440206E-05	0.00011897855	0.00014019222	0.00011942354	2.8021563E-05	0.00011909958	0.00013396324	0.00011953378	2.5121799E-05	0.00011935066	0.00014385104	0.00011868292	2.4220833E-05	0.00011878335
# 0.0025690068	0.002513773	0.000612657	0.0023642068	0.0032676146	0.002687461	0.0007942304	0.0029658393	0.002946649	0.002658651	0.0007289576	0.0025547617	0.0029469398	0.0025400578	0.0007424388	0.0025481451	0.0029568602	0.002567729	0.0007330381	0.0025656708	0.002968483	0.0027645158	0.00071951986	0.00276423	0.0031251032	0.0027019784	0.0007504149	0.0026374848	0.0032212206	0.0028949024	0.0007328072	0.0026828013	0.0034377754	0.0027473113	0.0008557611	0.0031647068	0.0032848923	0.0026932429	0.00075781863	0.0027464207	0.0029718166	0.0026530088	0.0006936368	0.0029370673	0.0028937482	0.0025162608	0.0006810936	0.0026607355	0.003032044	0.0029303133	0.0006838064	0.002877402
# 0.005836331	0.0042689373	0.0014674369	0.004737947	0.0066076377	0.0059429626	0.0016415126	0.0051884754	0.0067575355	0.00468134	0.0016954963	0.0051581454	0.006877937	0.004984713	0.0017374164	0.0052513853	0.0068914043	0.0047865435	0.0016874971	0.005164184	0.0067281113	0.0054089744	0.0016909376	0.005509096	0.0071413023	0.0056899437	0.0017968718	0.0057726535	0.0077111647	0.005475896	0.0018568172	0.005817962	0.007882103	0.006321952	0.0019216515	0.0057159695	0.007992553	0.0058817435	0.0018980955	0.005685286	0.0066526867	0.0060049207	0.0015715428	0.005888402	0.007064182	0.005892667	0.0016966867	0.0060858387	0.007449875	0.005211285	0.0017348687	0.005909616
# 0.010264396	0.009129297	0.0025260765	0.008756157	0.010838163	0.007465536	0.0027777876	0.007878702	0.011963857	0.009419353	0.0029179743	0.009133253	0.011803998	0.009869428	0.002918976	0.009971051	0.011912201	0.010478558	0.0028793884	0.009955756	0.0116313705	0.009292392	0.002878574	0.009611364	0.01223536	0.009702802	0.0030191995	0.010276552	0.013373986	0.011416487	0.0032142824	0.010724233	0.012512046	0.01201318	0.0031927435	0.010950479	0.013337369	0.012087287	0.0032648724	0.011597867	0.011578926	0.008537608	0.002672526	0.010145501	0.0121617345	0.009889136	0.0029153496	0.010392551	0.012997985	0.0117155295	0.0030777645	0.01015733
# 0.014831658	0.013279054	0.003608682	0.013371101	0.016251251	0.012345411	0.0041773245	0.011492117	0.017237533	0.0152924135	0.004267941	0.015039607	0.016872814	0.015496567	0.0041037714	0.015891947	0.01707377	0.014878927	0.0041679894	0.0156129645	0.017027661	0.014809536	0.00404627	0.014288565	0.018053873	0.014434817	0.004311673	0.013188092	0.018792802	0.017973358	0.004466764	0.016642686	0.01756907	0.015041025	0.0044218265	0.015616823	0.018609403	0.018492315	0.0046468657	0.015477248	0.017528778	0.015090845	0.0039050218	0.012790497	0.017974852	0.0148236975	0.0041412464	0.01501946	0.018202595	0.017957337	0.004373065	0.015643895
# 0.019840324	0.017320778	0.004881386	0.016645001	0.022262413	0.01771635	0.005906561	0.016146472	0.022753522	0.020083949	0.0057634623	0.018740853	0.022342687	0.020805229	0.005488179	0.018578421	0.022789834	0.021709716	0.0057223174	0.01825227	0.022555757	0.022611465	0.0053889193	0.01880293	0.024116734	0.024111265	0.0058390577	0.020892631	0.024892326	0.021170935	0.0059697037	0.019250356	0.024214983	0.019153621	0.0059760134	0.018717652	0.025213411	0.019388678	0.0063291	0.020286936	0.023349054	0.023531305	0.005431647	0.018395064	0.024459185	0.02085987	0.0056067496	0.018951269	0.024484172	0.020132925	0.005865778	0.020169087
# 0.025598902	0.02250539	0.006432862	0.022608805	0.02927581	0.022879321	0.008017878	0.023147026	0.02902082	0.025661351	0.0076035783	0.022108484	0.028763207	0.025604516	0.0073636775	0.021721914	0.029206753	0.026383482	0.007736835	0.022594793	0.028536633	0.026266607	0.0070453286	0.023908675	0.030386169	0.028257115	0.007513664	0.026391916	0.032443617	0.026713712	0.007961658	0.025637694	0.03235648	0.026085952	0.008034674	0.02677037	0.033195708	0.030231504	0.008404074	0.028248426	0.029717559	0.027620966	0.007371774	0.024032116	0.03167293	0.027533462	0.0075037526	0.024356915	0.032620717	0.026062839	0.007866154	0.028508145
# 0.032090202	0.029488442	0.008214499	0.029701224	0.03718282	0.03297733	0.010411761	0.030456921	0.036251348	0.0319514	0.009820452	0.028140256	0.036471054	0.030848553	0.009749063	0.028748862	0.03703163	0.03184684	0.010125763	0.029960636	0.035834	0.032700278	0.009173213	0.028129205	0.03767226	0.03645704	0.009527661	0.03150849	0.040920567	0.039634444	0.010382925	0.03680284	0.041580115	0.03416709	0.010633687	0.033716135	0.04210937	0.035047226	0.010802502	0.03811775	0.037465326	0.035475135	0.009807035	0.033581402	0.039493043	0.037185673	0.009698761	0.036097262	0.042019937	0.038459085	0.010403755	0.036606018
# 0.039688703	0.033623043	0.010348398	0.032724388	0.04563075	0.044054493	0.013043903	0.037120614	0.045116086	0.03672115	0.012376866	0.03593605	0.045678716	0.03956464	0.0123846205	0.039837334	0.046227124	0.040226206	0.012878141	0.03717701	0.044880185	0.038987976	0.0117759025	0.035756994	0.046253335	0.041070778	0.011927878	0.037780106	0.050288744	0.048154525	0.013005414	0.048522636	0.05217162	0.046483282	0.013601708	0.04399893	0.052503224	0.048792746	0.013325816	0.04860056	0.04652447	0.042600952	0.01239849	0.044604048	0.048573952	0.043049645	0.012031775	0.042973533	0.053018544	0.043945048	0.013496416	0.048868727
# 0.04942779	0.04035747	0.0129870055	0.039374925	0.054850012	0.053945694	0.015902529	0.046867907	0.056370977	0.044651866	0.015265102	0.048314057	0.056801014	0.047837637	0.015343881	0.047835093	0.057184555	0.04873454	0.015855504	0.052651912	0.055673674	0.04895243	0.014930786	0.04655201	0.05687877	0.04797183	0.014989158	0.046809506	0.061468825	0.054680757	0.015799599	0.057108276	0.063700035	0.060081683	0.016734855	0.059621044	0.06447297	0.053272936	0.016067501	0.060099445	0.05774458	0.047973804	0.015333511	0.051838003	0.060591277	0.04796966	0.0147505095	0.052384555	0.06642444	0.052665576	0.016637186	0.05934798
# 0.06090485	0.054473475	0.016073938	0.051171873	0.06483105	0.06623576	0.018776255	0.061995845	0.069388345	0.059018254	0.018484922	0.057266966	0.06880493	0.06758889	0.018738016	0.06194416	0.068883136	0.06618578	0.019012168	0.06076043	0.06890996	0.058136925	0.01884966	0.058552206	0.07117501	0.053998232	0.01905671	0.05602566	0.075385734	0.061245397	0.019083753	0.063516416	0.07707731	0.066066615	0.019907467	0.07201907	0.07881475	0.06811687	0.01919567	0.06779488	0.07200683	0.058254097	0.019120825	0.060631733	0.07581228	0.062449895	0.018349146	0.055323496	0.08107264	0.070676826	0.019927874	0.064522885
# 0.07360052	0.06349219	0.019482283	0.06518396	0.07591828	0.07808929	0.021832382	0.07536242	0.08327696	0.07686855	0.022336604	0.064633936	0.081481	0.08124175	0.022775047	0.06971473	0.08130896	0.07634336	0.022672417	0.07012831	0.084597506	0.07579488	0.023859732	0.071146816	0.08846634	0.07725968	0.024212275	0.07045176	0.0919406	0.076363705	0.022989828	0.07343763	0.09291605	0.08400913	0.023398595	0.078253046	0.09361613	0.09383884	0.02315039	0.074288316	0.08849408	0.077062935	0.023907585	0.075748116	0.093199275	0.084276944	0.023399496	0.06647031	0.096126005	0.09132529	0.023415359	0.06965363
# 0.08749362	0.08608245	0.023306161	0.079191245	0.08858926	0.086041346	0.025383715	0.08133082	0.09762267	0.10301779	0.026814014	0.08913279	0.09587488	0.093608946	0.027597226	0.08101065	0.09571726	0.09299544	0.027005155	0.08373388	0.10274587	0.09184477	0.029725354	0.087012894	0.10739961	0.09218276	0.030176273	0.087279946	0.11079618	0.09383401	0.02756095	0.08445129	0.11002255	0.105545886	0.027990604	0.08206621	0.10958462	0.10832969	0.028505996	0.07769532	0.10701553	0.098247424	0.028996617	0.097459964	0.1124243	0.10600966	0.029301148	0.08865561	0.1127624	0.10586217	0.02733126	0.08638261
# 0.10271754	0.09751232	0.027733775	0.09734223	0.10310045	0.10752117	0.029701574	0.10001655	0.11258997	0.121183015	0.031765483	0.10203953	0.11346878	0.10151955	0.033413567	0.09202033	0.112922005	0.10606202	0.032178376	0.094451174	0.123856716	0.12028083	0.036671944	0.10757553	0.1297485	0.11876885	0.03709903	0.11009281	0.13126391	0.1301481	0.033945978	0.08529338	0.129546	0.1204112	0.034350988	0.08773514	0.12947042	0.11644151	0.035353303	0.09683211	0.12728506	0.12833014	0.03519525	0.108794175	0.13521872	0.12742694	0.036874644	0.10375514	0.13335794	0.11975571	0.03282819	0.09087089
# 0.1225541	0.104959026	0.033338595	0.099885546	0.120238885	0.11336331	0.03510993	0.11117825	0.13183063	0.11561295	0.038089342	0.1107609	0.13524434	0.12038638	0.03996163	0.124557264	0.13464014	0.12437303	0.038460776	0.118401006	0.1474565	0.15292227	0.044856507	0.13126971	0.15493515	0.14362174	0.045129832	0.1373424	0.15298903	0.14467295	0.042055145	0.12449045	0.15378648	0.12570779	0.042758476	0.107554786	0.15430327	0.13796127	0.043572027	0.11320392	0.15170674	0.14128786	0.042903647	0.14149514	0.16243768	0.15255095	0.046919074	0.142838	0.1576756	0.15105228	0.042469647	0.09731387
# 0.15000111	0.119122095	0.040555973	0.12903923	0.14622204	0.10368341	0.042036876	0.12293411	0.15921828	0.1307167	0.046287443	0.13348193	0.16252716	0.14369076	0.04759594	0.14628908	0.16121367	0.14719522	0.04581369	0.14147724	0.17643784	0.16204095	0.05446312	0.1587893	0.18455306	0.17982812	0.055160463	0.150373	0.1797352	0.1731783	0.05083011	0.1610262	0.18371344	0.1534251	0.05379073	0.13321397	0.18580434	0.14890015	0.05352324	0.1421056	0.18393923	0.15254149	0.052377846	0.15562242	0.19554982	0.16975246	0.058814064	0.17522262	0.18533225	0.18059108	0.05621599	0.15946102
# 0.18657596	0.1445826	0.049585383	0.15256731	0.18485893	0.12772161	0.05091773	0.14635989	0.20037313	0.1556123	0.056887873	0.15865548	0.1992302	0.18266271	0.057598006	0.18328838	0.19752264	0.16471815	0.054880474	0.18134463	0.2184469	0.16567732	0.066486664	0.17614539	0.22136284	0.18957324	0.06833286	0.18361352	0.21614125	0.18266566	0.060906503	0.18306604	0.2201844	0.1768627	0.06739702	0.17332336	0.22723961	0.16199541	0.06624643	0.16858932	0.2276395	0.17983684	0.06509628	0.17285398	0.23759344	0.18720542	0.072743416	0.20430036	0.22203691	0.18012376	0.06950047	0.21052974
# 0.2384303	0.19100018	0.06308951	0.1831422	0.23805106	0.16389696	0.06308935	0.16232772	0.25895923	0.18260816	0.0725996	0.19158241	0.25390914	0.18947363	0.07040666	0.21206689	0.2510452	0.18522689	0.06765922	0.18739206	0.28135416	0.1975457	0.082812086	0.20328513	0.27824548	0.20199022	0.08499505	0.22087179	0.2689792	0.20031914	0.07410705	0.22378543	0.2756756	0.20417473	0.0837414	0.23488356	0.28718904	0.2054992	0.082907006	0.22993068	0.28401515	0.21769945	0.08366679	0.19520573	0.29371005	0.23092309	0.09124888	0.23845504	0.2796896	0.19657871	0.084017456	0.2872454
# 0.32949427	0.21014242	0.08523896	0.23712213	0.32639024	0.22707373	0.081770666	0.21700136	0.35567406	0.25990376	0.09468396	0.28107935	0.35014418	0.22755279	0.09097835	0.24447857	0.34269962	0.2302976	0.089754805	0.23336138	0.3689121	0.2743946	0.106606066	0.28496248	0.36354342	0.2774113	0.110145524	0.24026863	0.34296295	0.2776258	0.09374712	0.26049155	0.36606568	0.30709785	0.106860295	0.28150374	0.37742984	0.32255098	0.10581174	0.26810327	0.36695704	0.27734518	0.112063244	0.263147	0.37258536	0.3026054	0.12212039	0.25095928	0.3760596	0.31925818	0.106317304	0.26658607
# 0.5327235	0.35482723	0.13865821	0.29063419	0.4561464	0.3453891	0.12798557	0.25082517	0.5252476	0.33951762	0.15305229	0.2709261	0.5373784	0.3209198	0.1401067	0.28093067	0.52684474	0.31208244	0.13977243	0.30067918	0.54012346	0.3446827	0.16808215	0.3126107	0.5348997	0.33342165	0.1874735	0.30665067	0.52142614	0.2960098	0.15566358	0.26858437	0.534871	0.32855514	0.17857444	0.30579054	0.55056304	0.34258875	0.1822829	0.30637315	0.55658317	0.3563684	0.18857855	0.3373401	0.55551714	0.36285776	0.21925487	0.34093624	0.54714584	0.3338828	0.18405898	0.31421718


# 3. bin predictions into 4 bins of equal weight of positive labels

const bin_count = 4

function find_ŷ_bin_splits(event_name, model_name, ŷ, Ys, weights)
  y = Ys[event_name]

  total_positive_weight = sum(Float64.(y .* weights))
  per_bin_pos_weight = total_positive_weight / bin_count

  sort_perm      = Metrics.parallel_sort_perm(ŷ);
  y_sorted       = Metrics.parallel_apply_sort_perm(y, sort_perm);
  ŷ_sorted       = Metrics.parallel_apply_sort_perm(ŷ, sort_perm);
  weights_sorted = Metrics.parallel_apply_sort_perm(weights, sort_perm);

  bins_Σŷ      = zeros(Float64, bin_count)
  bins_Σy      = zeros(Float64, bin_count)
  bins_Σweight = zeros(Float64, bin_count)
  bins_max     = ones(Float32, bin_count)

  bin_i = 1
  for i in 1:length(y_sorted)
    if ŷ_sorted[i] > bins_max[bin_i]
      bin_i += 1
    end

    bins_Σŷ[bin_i]      += Float64(ŷ_sorted[i] * weights_sorted[i])
    bins_Σy[bin_i]      += Float64(y_sorted[i] * weights_sorted[i])
    bins_Σweight[bin_i] += Float64(weights_sorted[i])

    if bins_Σy[bin_i] >= per_bin_pos_weight
      bins_max[bin_i] = ŷ_sorted[i]
    end
  end

  for bin_i in 1:bin_count
    Σŷ      = bins_Σŷ[bin_i]
    Σy      = bins_Σy[bin_i]
    Σweight = bins_Σweight[bin_i]

    mean_ŷ = Σŷ / Σweight
    mean_y = Σy / Σweight

    println("$model_name\t$(Float32(mean_y))\t$(Float32(mean_ŷ))\t$(Float32(Σweight))\t$(bins_max[bin_i])")
  end

  bins_max
end

nmodels = length(HREFPredictionAblations.models)
event_to_day_bins = Dict{String,Vector{Float32}}()
println("model_name\tmean_y\tmean_ŷ\tΣweight\tbin_max")
for prediction_i in 1:nmodels
  model_name, _, _, _, _ = HREFPredictionAblations.models[prediction_i]

  ŷ = @view X[:,(prediction_i*2) - 1]; # total prob acc

  event_to_day_bins[model_name] = find_ŷ_bin_splits("tornado", model_name, ŷ, Ys, weights)

  # println("event_to_day_bins[\"$event_name\"] = $(event_to_day_bins[event_name])")
end

# model_name                                                mean_y        mean_ŷ        Σweight    bin_max
# tornado_mean_58                                           0.0005525441  0.00064435945 8.74989e6  0.017304773
# tornado_mean_58                                           0.026186043   0.030275125   184641.98  0.0553223
# tornado_mean_58                                           0.07631268    0.084108755   63355.395  0.13471735
# tornado_mean_58                                           0.1773909     0.24363838    27244.582  1.0
# tornado_prob_80                                           0.00055105187 0.0007312233  8.773089e6 0.019274753
# tornado_prob_80                                           0.029058345   0.033250812   166377.2   0.059863195
# tornado_prob_80                                           0.08667382    0.086673744   55785.69   0.13051617
# tornado_prob_80                                           0.16175802    0.22859141    29879.643  1.0
# tornado_mean_prob_138                                     0.0005505699  0.0006848417  8.780765e6 0.019996958
# tornado_mean_prob_138                                     0.029467909   0.034681678   164072.67  0.06293694
# tornado_mean_prob_138                                     0.08824085    0.093168594   54791.113  0.14385402
# tornado_mean_prob_138                                     0.18952441    0.25917518    25502.709  1.0
# tornado_mean_prob_computed_no_sv_219                      0.0005502851  0.0006740746  8.786296e6 0.019619932
# tornado_mean_prob_computed_no_sv_219                      0.030165914   0.034514353   160281.48  0.06317351
# tornado_mean_prob_computed_no_sv_219                      0.0892953     0.09441178    54143.47   0.14767814
# tornado_mean_prob_computed_no_sv_219                      0.19797844    0.2680879     24410.443  1.0
# tornado_mean_prob_computed_220                            0.0005500478  0.00067421497 8.790041e6 0.020045973
# tornado_mean_prob_computed_220                            0.031090373   0.035085127   155508.97  0.06332173
# tornado_mean_prob_computed_220                            0.088539734   0.09354061    54611.535  0.14715679
# tornado_mean_prob_computed_220                            0.19352877    0.26402542    24970.133  1.0
# tornado_mean_prob_computed_partial_climatology_227        0.0005498372  0.00066107186 8.793964e6 0.019988786
# tornado_mean_prob_computed_partial_climatology_227        0.03144185    0.03438385    153782.98  0.061916392
# tornado_mean_prob_computed_partial_climatology_227        0.08897645    0.09635408    54334.35   0.16038308
# tornado_mean_prob_computed_partial_climatology_227        0.20965223    0.2826052     23050.35   1.0
# tornado_mean_prob_computed_climatology_253                0.0005492325  0.00068179093 8.803066e6 0.02144064
# tornado_mean_prob_computed_climatology_253                0.033442535   0.03624927    144564.94  0.063346006
# tornado_mean_prob_computed_climatology_253                0.08659002    0.100448      55831.27   0.16949469
# tornado_mean_prob_computed_climatology_253                0.22305384    0.28914496    21669.77   1.0
# tornado_mean_prob_computed_climatology_blurs_910          0.0005490907  0.00065349595 8.805662e6 0.02166239
# tornado_mean_prob_computed_climatology_blurs_910          0.03356391    0.037512593   144040.5   0.06790343
# tornado_mean_prob_computed_climatology_blurs_910          0.091271184   0.10380857    52967.86   0.16562855
# tornado_mean_prob_computed_climatology_blurs_910          0.21518865    0.28201032    22461.203  1.0
# tornado_mean_prob_computed_climatology_grads_1348         0.0005495181  0.00065491704 8.798646e6 0.020726241
# tornado_mean_prob_computed_climatology_grads_1348         0.0316612     0.03734225    152710.67  0.06985498
# tornado_mean_prob_computed_climatology_grads_1348         0.09449947    0.106237754   51163.77   0.16851029
# tornado_mean_prob_computed_climatology_grads_1348         0.2137203     0.28244874    22611.502  1.0
# tornado_mean_prob_computed_climatology_blurs_grads_2005   0.00054904306 0.00064249634 8.805825e6 0.021495968
# tornado_mean_prob_computed_climatology_blurs_grads_2005   0.032774426   0.038583875   147511.77  0.071746476
# tornado_mean_prob_computed_climatology_blurs_grads_2005   0.09908231    0.10698437    48797.91   0.16894186
# tornado_mean_prob_computed_climatology_blurs_grads_2005   0.21016328    0.28666922    22996.9    1.0
# tornado_mean_prob_computed_climatology_prior_next_hrs_691 0.0005491873  0.0006503939  8.803094e6 0.020654399
# tornado_mean_prob_computed_climatology_prior_next_hrs_691 0.033055868   0.035830382   146264.4   0.06460478
# tornado_mean_prob_computed_climatology_prior_next_hrs_691 0.0905178     0.10031668    53417.71   0.16610569
# tornado_mean_prob_computed_climatology_prior_next_hrs_691 0.2161817     0.2877253     22355.297  1.0
# tornado_mean_prob_computed_climatology_3hr_1567           0.0005491144  0.0006493895  8.80407e6  0.021308538
# tornado_mean_prob_computed_climatology_3hr_1567           0.032205112   0.037208974   150132.1   0.067999676
# tornado_mean_prob_computed_climatology_3hr_1567           0.096681885   0.10643049    50004.45   0.17814146
# tornado_mean_prob_computed_climatology_3hr_1567           0.23098622    0.2981269     20925.564  1.0
# tornado_full_13831                                        0.000549145   0.0006369488  8.804865e6 0.021043906
# tornado_full_13831                                        0.0320742     0.038657922   150732.98  0.074019335
# tornado_full_13831                                        0.100802325   0.10940487    47960.055  0.17095083
# tornado_full_13831                                        0.22403128    0.29084346    21573.916  1.0


println("event_to_day_bins = $event_to_day_bins")
# event_to_day_bins = Dict{String, Vector{Float32}}("tornado_mean_prob_computed_partial_climatology_227" => [0.019988786, 0.061916392, 0.16038308, 1.0], "tornado_mean_58" => [0.017304773, 0.0553223, 0.13471735, 1.0], "tornado_mean_prob_computed_climatology_prior_next_hrs_691" => [0.020654399, 0.06460478, 0.16610569, 1.0], "tornado_prob_80" => [0.019274753, 0.059863195, 0.13051617, 1.0], "tornado_full_13831" => [0.021043906, 0.074019335, 0.17095083, 1.0], "tornado_mean_prob_computed_no_sv_219" => [0.019619932, 0.06317351, 0.14767814, 1.0], "tornado_mean_prob_computed_220" => [0.020045973, 0.06332173, 0.14715679, 1.0], "tornado_mean_prob_computed_climatology_253" => [0.02144064, 0.063346006, 0.16949469, 1.0], "tornado_mean_prob_138" => [0.019996958, 0.06293694, 0.14385402, 1.0], "tornado_mean_prob_computed_climatology_grads_1348" => [0.020726241, 0.06985498, 0.16851029, 1.0], "tornado_mean_prob_computed_climatology_blurs_910" => [0.02166239, 0.06790343, 0.16562855, 1.0], "tornado_mean_prob_computed_climatology_blurs_grads_2005" => [0.021495968, 0.071746476, 0.16894186, 1.0], "tornado_mean_prob_computed_climatology_3hr_1567" => [0.021308538, 0.067999676, 0.17814146, 1.0])







# 4. combine bin-pairs (overlapping, 3 bins total)
# 5. train a logistic regression for each bin, σ(a1*logit(HREF) + a2*logit(SREF) + a3*logit(HREF)*logit(SREF) + a4*max(logit(HREF),logit(SREF)) + a5*min(logit(HREF),logit(SREF)) + b)
# was producing dangerously large coeffs even for simple 4-param models like σ(a1*logit(HREF) + a2*logit(SREF) + a3*logit(HREF*SREF) + b) so avoiding all interaction terms


function find_logistic_coeffs(event_name, model_name, prediction_i, X, Ys, weights)
  y = Ys[event_name]
  ŷ = @view X[:,(prediction_i*2) - 1]; # total prob acc

  bins_max = event_to_day_bins[model_name]
  bins_logistic_coeffs = []

  # Paired, overlapping bins
  for bin_i in 1:(bin_count - 1)
    bin_min = bin_i >= 2 ? bins_max[bin_i-1] : -1f0
    bin_max = bins_max[bin_i+1]

    bin_members = (ŷ .> bin_min) .* (ŷ .<= bin_max)

    bin_total_prob_x  = X[bin_members, prediction_i*2 - 1]
    bin_max_hourly_x  = X[bin_members, prediction_i*2]
    # bin_ŷ       = ŷ[bin_members]
    bin_y       = y[bin_members]
    bin_weights = weights[bin_members]
    bin_weight  = sum(bin_weights)

    # logit(HREF), logit(SREF)
    bin_X_features = Array{Float32}(undef, (length(bin_y), 2))

    Threads.@threads for i in 1:length(bin_y)
      logit_total_prob = logit(bin_total_prob_x[i])
      logit_max_hourly = logit(bin_max_hourly_x[i])

      bin_X_features[i,1] = logit_total_prob
      bin_X_features[i,2] = logit_max_hourly
    end

    coeffs = LogisticRegression.fit(bin_X_features, bin_y, bin_weights; iteration_count = 300)

    # println("Fit logistic coefficients: $(coeffs)")

    logistic_ŷ = LogisticRegression.predict(bin_X_features, coeffs)

    stuff = [
      ("model_name", model_name),
      ("bin", "$bin_i-$(bin_i+1)"),
      ("total_prob_ŷ_min", bin_min),
      ("total_prob_ŷ_max", bin_max),
      ("count", length(bin_y)),
      ("pos_count", sum(bin_y)),
      ("weight", bin_weight),
      ("mean_total_prob_ŷ", sum(bin_total_prob_x .* bin_weights) / bin_weight),
      ("mean_max_hourly_ŷ", sum(bin_max_hourly_x .* bin_weights) / bin_weight),
      ("mean_y", sum(bin_y .* bin_weights) / bin_weight),
      ("total_prob_logloss", sum(logloss.(bin_y, bin_total_prob_x) .* bin_weights) / bin_weight),
      ("max_hourly_logloss", sum(logloss.(bin_y, bin_max_hourly_x) .* bin_weights) / bin_weight),
      ("total_prob_au_pr", Float32(Metrics.area_under_pr_curve(bin_total_prob_x, bin_y, bin_weights))),
      ("max_hourly_au_pr", Float32(Metrics.area_under_pr_curve(bin_max_hourly_x, bin_y, bin_weights))),
      ("mean_logistic_ŷ", sum(logistic_ŷ .* bin_weights) / bin_weight),
      ("logistic_logloss", sum(logloss.(bin_y, logistic_ŷ) .* bin_weights) / bin_weight),
      ("logistic_au_pr", Float32(Metrics.area_under_pr_curve(logistic_ŷ, bin_y, bin_weights))),
      ("logistic_coeffs", coeffs)
    ]

    headers = map(first, stuff)
    row     = map(last, stuff)

    bin_i == 1 && println(join(headers, "\t"))
    println(join(row, "\t"))

    push!(bins_logistic_coeffs, coeffs)
  end

  bins_logistic_coeffs
end

event_to_day_bins_logistic_coeffs = Dict{String,Vector{Vector{Float32}}}()
for prediction_i in 1:nmodels
  model_name, _, _, _, _ = HREFPredictionAblations.models[prediction_i]

  event_to_day_bins_logistic_coeffs[model_name] = find_logistic_coeffs("tornado", model_name, prediction_i, X, Ys, weights)
end

# model_name                                                bin total_prob_ŷ_min total_prob_ŷ_max count   pos_count weight     mean_total_prob_ŷ mean_max_hourly_ŷ mean_y       total_prob_logloss max_hourly_logloss total_prob_au_pr max_hourly_au_pr mean_logistic_ŷ logistic_logloss logistic_au_pr logistic_coeffs
# tornado_mean_58                                           1-2 -1.0             0.0553223        9696254 10423.0   8.934532e6 0.0012567121      0.00032494482     0.0010822886 0.0060854536       0.0066588083       0.02568117       0.025274718      0.0010822888    0.0060613793     0.026141258    Float32[0.6559432,  0.33183938,   0.2690549]
# tornado_mean_58                                           2-3 0.017304773      0.13471735       265695  10312.0   247997.38  0.044027895       0.011814468       0.038991798  0.15783298         0.1784407          0.07641898       0.073937856      0.038991798     0.15708669       0.077856414    Float32[0.6666154,  0.36046487,   0.4614278]
# tornado_mean_58                                           3-4 0.0553223        1.0              95762   10183.0   90599.98   0.13208137        0.034078375       0.10670821   0.3278625          0.3775572          0.20934382       0.19714081       0.106708206     0.32114118       0.21226463     Float32[0.42497587, 0.44417486,   0.20855717]
# model_name                                                bin total_prob_ŷ_min total_prob_ŷ_max count   pos_count weight     mean_total_prob_ŷ mean_max_hourly_ŷ mean_y       total_prob_logloss max_hourly_logloss total_prob_au_pr max_hourly_au_pr mean_logistic_ŷ logistic_logloss logistic_au_pr logistic_coeffs
# tornado_prob_80                                           1-2 -1.0             0.059863195      9702047 10437.0   8.939466e6 0.0013364629      0.00038399108     0.0010816164 0.0060175555       0.0064752693       0.030860346      0.02929963       0.0010816163    0.005984786      0.031192083    Float32[0.8796285,  0.16921686,   0.22449368]
# tornado_prob_80                                           2-3 0.019274753      0.13051617       236468  10324.0   222162.9   0.04666545        0.014008047       0.04352574   0.17031741         0.19181994         0.084860384      0.08604295       0.043525744     0.16960227       0.08983052     Float32[0.8071765,  0.3629977,    0.88375133]
# tornado_prob_80                                           3-4 0.059863195      1.0              89969   10169.0   85665.336  0.13617392        0.037335          0.11286281   0.34404176         0.39736992         0.20111248       0.18362024       0.1128628       0.33861682       0.20195282     Float32[0.51591927, 0.29846162,   -0.097147524]
# model_name                                                bin total_prob_ŷ_min total_prob_ŷ_max count   pos_count weight     mean_total_prob_ŷ mean_max_hourly_ŷ mean_y       total_prob_logloss max_hourly_logloss total_prob_au_pr max_hourly_au_pr mean_logistic_ŷ logistic_logloss logistic_au_pr logistic_coeffs
# tornado_mean_prob_138                                     1-2 -1.0             0.06293694       9707455 10417.0   8.944838e6 0.0013084363      0.00036521038     0.0010809927 0.0059776995       0.006496915        0.028291194      0.026154425      0.0010809926    0.0059552435     0.028430644    Float32[0.93436664, 0.07926717,   -0.026851058]
# tornado_mean_prob_138                                     2-3 0.019996958      0.14385402       233659  10320.0   218863.8   0.049323488       0.014423123       0.04418133   0.17289616         0.19545229         0.087084584      0.08254773       0.04418133      0.17245421       0.08872876     Float32[0.9042307,  0.17261904,   0.32986057]
# tornado_mean_prob_138                                     3-4 0.06293694       1.0              84561   10189.0   80293.81   0.14589518        0.04049197        0.12041028   0.35680637         0.41135052         0.215481         0.209414         0.12041028      0.35006297       0.22002642     Float32[0.46952277, 0.3558928,    0.015089741]
# model_name                                                bin total_prob_ŷ_min total_prob_ŷ_max count   pos_count weight     mean_total_prob_ŷ mean_max_hourly_ŷ mean_y       total_prob_logloss max_hourly_logloss total_prob_au_pr max_hourly_au_pr mean_logistic_ŷ logistic_logloss logistic_au_pr logistic_coeffs
# tornado_mean_prob_computed_no_sv_219                      1-2 -1.0             0.06317351       9709203 10415.0   8.946578e6 0.0012803365      0.0003509757      0.0010808608 0.005960811        0.0065027666       0.029830607      0.02745438       0.0010808606    0.005941358      0.029989375    Float32[0.8970471,  0.12673378,   0.10686361]
# tornado_mean_prob_computed_no_sv_219                      2-3 0.019619932      0.14767814       229041  10315.0   214424.97  0.049638774       0.0143582625      0.045096405  0.17584834         0.19896401         0.08727907       0.08571333       0.045096405     0.17540172       0.089979455    Float32[0.77730256, 0.24956441,   0.30919433]
# tornado_mean_prob_computed_no_sv_219                      3-4 0.06317351       1.0              82813   10191.0   78553.914  0.1483812         0.040372815       0.12306833   0.36492777         0.4197913          0.21194836       0.20339726       0.12306832      0.3567272        0.21477623     Float32[0.38325134, 0.41652393,   0.074524485]
# model_name                                                bin total_prob_ŷ_min total_prob_ŷ_max count   pos_count weight     mean_total_prob_ŷ mean_max_hourly_ŷ mean_y       total_prob_logloss max_hourly_logloss total_prob_au_pr max_hourly_au_pr mean_logistic_ŷ logistic_logloss logistic_au_pr logistic_coeffs
# tornado_mean_prob_computed_220                            1-2 -1.0             0.06332173       9708100 10412.0   8.94555e6  0.0012724124      0.00035469467     0.0010809592 0.0059478595       0.0064870096       0.029999185      0.028186163      0.0010809592    0.005930681      0.030236766    Float32[0.9081216,  0.107832894,  0.050864797]
# tornado_mean_prob_computed_220                            2-3 0.020045973      0.14715679       224454  10314.0   210120.52  0.05027804        0.014654327       0.046021793  0.17881362         0.20252019         0.089375         0.08498586       0.046021793     0.1784264        0.090924874    Float32[0.800826,   0.23256853,   0.31022668]
# tornado_mean_prob_computed_220                            3-4 0.06332173       1.0              83916   10194.0   79581.67   0.14703318        0.03939797        0.12148187   0.36165562         0.4170215          0.20962848       0.20314537       0.12148187      0.35378066       0.21338512     Float32[0.40040252, 0.4055984,    0.067884356]
# model_name                                                bin total_prob_ŷ_min total_prob_ŷ_max count   pos_count weight     mean_total_prob_ŷ mean_max_hourly_ŷ mean_y       total_prob_logloss max_hourly_logloss total_prob_au_pr max_hourly_au_pr mean_logistic_ŷ logistic_logloss logistic_au_pr logistic_coeffs
# tornado_mean_prob_computed_partial_climatology_227        1-2 -1.0             0.061916392      9710511 10405.0   8.947747e6 0.001240658       0.0003388053      0.0010807714 0.0059278994       0.0065114894       0.03022069       0.028243082      0.0010807713    0.005915461      0.03050957     Float32[0.9746325,  0.05395953,   0.059440296]
# tornado_mean_prob_computed_partial_climatology_227        2-3 0.019988786      0.16038308       221807  10317.0   208117.34  0.05056276        0.014658076       0.046462726  0.17946826         0.204413           0.096210934      0.0893217        0.046462722     0.179183         0.0984764      Float32[0.8418499,  0.16475138,   0.1517096]
# tornado_mean_prob_computed_partial_climatology_227        3-4 0.061916392      1.0              81505   10201.0   77384.7    0.15183213        0.044288047       0.12492177   0.36279795         0.4149026          0.22880438       0.21963352       0.124921784     0.35611352       0.23174424     Float32[0.48570332, 0.34678283,   -0.001083027]
# model_name                                                bin total_prob_ŷ_min total_prob_ŷ_max count   pos_count weight     mean_total_prob_ŷ mean_max_hourly_ŷ mean_y       total_prob_logloss max_hourly_logloss total_prob_au_pr max_hourly_au_pr mean_logistic_ŷ logistic_logloss logistic_au_pr logistic_coeffs
# tornado_mean_prob_computed_climatology_253                1-2 -1.0             0.063346006      9710126 10383.0   8.947631e6 0.0012564469      0.00034074739     0.0010806825 0.005891865        0.006469967        0.031067852      0.02872078       0.0010806825    0.005876925      0.030992843    Float32[0.982298,   0.05099982,   0.06179544]
# tornado_mean_prob_computed_climatology_253                2-3 0.02144064       0.16949469       213944  10316.0   200396.22  0.054135315       0.01560994        0.04824966   0.18579943         0.21097252         0.09405763       0.08667607       0.04824966      0.18534118       0.09519578     Float32[0.80724055, 0.13905986,   -0.0778435]
# tornado_mean_prob_computed_climatology_253                3-4 0.063346006      1.0              81890   10223.0   77501.03   0.15320885        0.04611948        0.12474617   0.3605683          0.41195595         0.2311477        0.21925905       0.124746144     0.3547717        0.23343967     Float32[0.58624,    0.255771,     -0.13458002]
# model_name                                                bin total_prob_ŷ_min total_prob_ŷ_max count   pos_count weight     mean_total_prob_ŷ mean_max_hourly_ŷ mean_y       total_prob_logloss max_hourly_logloss total_prob_au_pr max_hourly_au_pr mean_logistic_ŷ logistic_logloss logistic_au_pr logistic_coeffs
# tornado_mean_prob_computed_climatology_blurs_910          1-2 -1.0             0.06790343       9712335 10378.0   8.949703e6 0.0012467225      0.00032986148     0.0010804458 0.0058266623       0.00642974         0.033769023      0.031561345      0.0010804457    0.0058138226     0.033726607    Float32[0.98352575, 0.03981221,   0.010008344]
# tornado_mean_prob_computed_climatology_blurs_910          2-3 0.02166239       0.16562855       210172  10314.0   197008.36  0.055336993       0.015332372       0.04907914   0.18692967         0.21396725         0.097825475      0.089538544      0.049079135     0.18652098       0.09890015     Float32[0.9775675,  0.045042068,  -0.0055927183]
# tornado_mean_prob_computed_climatology_blurs_910          3-4 0.06790343       1.0              79681   10228.0   75429.06   0.15687333        0.043077637       0.12817124   0.3720117          0.43263972         0.22026254       0.20930332       0.12817124      0.36554968       0.22139385     Float32[0.57631963, 0.20392773,   -0.28303716]
# model_name                                                bin total_prob_ŷ_min total_prob_ŷ_max count   pos_count weight     mean_total_prob_ŷ mean_max_hourly_ŷ mean_y       total_prob_logloss max_hourly_logloss total_prob_au_pr max_hourly_au_pr mean_logistic_ŷ logistic_logloss logistic_au_pr logistic_coeffs
# tornado_mean_prob_computed_climatology_grads_1348         1-2 -1.0             0.06985498       9714340 10397.0   8.951356e6 0.0012808052      0.00034945284     0.0010802852 0.005839013        0.006387385        0.033812318      0.029529627      0.0010802853    0.0058203507     0.033697013    Float32[0.94020444, 0.0850467,    0.051697694]
# tornado_mean_prob_computed_climatology_grads_1348         2-3 0.020726241      0.16851029       217332  10320.0   203874.44  0.05463208        0.01564935        0.047430914  0.18148142         0.20685415         0.09502037       0.08189804       0.047430906     0.18089765       0.09562675     Float32[1.1303655,  -0.08044045,  -0.13138889]
# tornado_mean_prob_computed_climatology_grads_1348         3-4 0.06985498       1.0              77676   10209.0   73775.266  0.16024496        0.04653408        0.13103966   0.37649903         0.43550298         0.22815737       0.21831112       0.13103966      0.37059173       0.23022527     Float32[0.63580066, 0.1437665,    -0.37667832]
# model_name                                                bin total_prob_ŷ_min total_prob_ŷ_max count   pos_count weight     mean_total_prob_ŷ mean_max_hourly_ŷ mean_y       total_prob_logloss max_hourly_logloss total_prob_au_pr max_hourly_au_pr mean_logistic_ŷ logistic_logloss logistic_au_pr logistic_coeffs
# tornado_mean_prob_computed_climatology_blurs_grads_2005   1-2 -1.0             0.071746476      9716472 10399.0   8.953336e6 0.0012676042      0.00034184597     0.0010799763 0.00582184         0.006394267        0.03330716       0.029734429      0.0010799761    0.005806496      0.033193134    Float32[0.95560277, 0.05098174,   -0.06451891]
# tornado_mean_prob_computed_climatology_blurs_grads_2005   2-3 0.021495968      0.16894186       209192  10321.0   196309.67  0.05558661        0.015846616       0.049256988  0.1866256          0.21373942         0.097892374      0.08648004       0.049257        0.18617484       0.0975887      Float32[1.1089742,  -0.043572877, -0.017801635]
# tornado_mean_prob_computed_climatology_blurs_grads_2005   3-4 0.071746476      1.0              75544   10207.0   71794.81   0.16453998        0.04728531        0.13466312   0.38415837         0.447142           0.23444557       0.2223931        0.13466312      0.3776991        0.23589237     Float32[0.64432305, 0.10341309,   -0.47638172]
# model_name                                                bin total_prob_ŷ_min total_prob_ŷ_max count   pos_count weight     mean_total_prob_ŷ mean_max_hourly_ŷ mean_y       total_prob_logloss max_hourly_logloss total_prob_au_pr max_hourly_au_pr mean_logistic_ŷ logistic_logloss logistic_au_pr logistic_coeffs
# tornado_mean_prob_computed_climatology_prior_next_hrs_691 1-2 -1.0             0.06460478       9712035 10394.0   8.949358e6 0.0012253602      0.00031497504     0.0010804625 0.0058915494       0.0065076104       0.031272214      0.030206645      0.0010804624    0.005879357      0.031604104    Float32[0.81858176, 0.18513452,   0.15438604]
# tornado_mean_prob_computed_climatology_prior_next_hrs_691 2-3 0.020654399      0.16610569       213084  10320.0   199682.1   0.053081363       0.014722611       0.04842773   0.18551108         0.2118019          0.09631815       0.090434216      0.048427735     0.1850077        0.09710584     Float32[0.6977342,  0.27534917,   0.21286853]
# tornado_mean_prob_computed_climatology_prior_next_hrs_691 3-4 0.06460478       1.0              79981   10212.0   75773.01   0.15560782        0.045160126       0.1275924    0.36658564         0.42145917         0.23740172       0.23477384       0.12759241      0.36017954       0.24182731     Float32[0.5196121,  0.31482816,   -0.040216677]
# model_name                                                bin total_prob_ŷ_min total_prob_ŷ_max count   pos_count weight     mean_total_prob_ŷ mean_max_hourly_ŷ mean_y       total_prob_logloss max_hourly_logloss total_prob_au_pr max_hourly_au_pr mean_logistic_ŷ logistic_logloss logistic_au_pr logistic_coeffs
# tornado_mean_prob_computed_climatology_3hr_1567           1-2 -1.0             0.067999676      9717107 10376.0   8.954202e6 0.0012623717      0.0003241106      0.0010798799 0.0058977003       0.006511817        0.031293306      0.027593525      0.0010798799    0.0058833743     0.031110743    Float32[0.96216094, 0.034555722,  -0.12364115]
# tornado_mean_prob_computed_climatology_3hr_1567           2-3 0.021308538      0.17814146       213541  10303.0   200136.53  0.054504093       0.015043081       0.048314743  0.18396774         0.2112294          0.10122298       0.09050967       0.04831475      0.18355963       0.1007994      Float32[1.0485787,  -0.011567661, -0.047783367]
# tornado_mean_prob_computed_climatology_3hr_1567           3-4 0.067999676      1.0              74909   10230.0   70930.016  0.1629842         0.04952672        0.13630395   0.38097858         0.44215336         0.25189593       0.24691355       0.13630396      0.3757996        0.25742134     Float32[0.6019041,  0.21778609,   -0.17962365]
# model_name                                                bin total_prob_ŷ_min total_prob_ŷ_max count   pos_count weight     mean_total_prob_ŷ mean_max_hourly_ŷ mean_y       total_prob_logloss max_hourly_logloss total_prob_au_pr max_hourly_au_pr mean_logistic_ŷ logistic_logloss logistic_au_pr logistic_coeffs
# tornado_full_13831                                        1-2 -1.0             0.074019335      9718800 10390.0   8.955598e6 0.0012768853      0.00033077784     0.0010797478 0.0058558607       0.0064523746       0.032501902      0.029135438      0.0010797479    0.0058391923     0.032301545    Float32[0.95958227, 0.04161413,   -0.10651286]
# tornado_full_13831                                        2-3 0.021043906      0.17095083       211661  10323.0   198693.03  0.055734657       0.015497775       0.048663635  0.1845327          0.2118171          0.10270019       0.089645624      0.048663624     0.18389481       0.10116889     Float32[1.2272763,  -0.15624464,  -0.18067063]
# tornado_full_13831                                        3-4 0.074019335      1.0              73216   10216.0   69533.97   0.1656988         0.04697025        0.13903588   0.39089894         0.45915875         0.23874679       0.23317282       0.13903588      0.38513187       0.24201544     Float32[0.5964124,  0.17200926,   -0.3083448]

println("event_to_day_bins_logistic_coeffs = $event_to_day_bins_logistic_coeffs")
# event_to_day_bins_logistic_coeffs = Dict{String, Vector{Vector{Float32}}}("tornado_mean_prob_computed_partial_climatology_227" => [[0.9746325, 0.05395953, 0.059440296], [0.8418499, 0.16475138, 0.1517096], [0.48570332, 0.34678283, -0.001083027]], "tornado_mean_58" => [[0.6559432, 0.33183938, 0.2690549], [0.6666154, 0.36046487, 0.4614278], [0.42497587, 0.44417486, 0.20855717]], "tornado_mean_prob_computed_climatology_prior_next_hrs_691" => [[0.81858176, 0.18513452, 0.15438604], [0.6977342, 0.27534917, 0.21286853], [0.5196121, 0.31482816, -0.040216677]], "tornado_prob_80" => [[0.8796285, 0.16921686, 0.22449368], [0.8071765, 0.3629977, 0.88375133], [0.51591927, 0.29846162, -0.097147524]], "tornado_full_13831" => [[0.95958227, 0.04161413, -0.10651286], [1.2272763, -0.15624464, -0.18067063], [0.5964124, 0.17200926, -0.3083448]], "tornado_mean_prob_computed_no_sv_219" => [[0.8970471, 0.12673378, 0.10686361], [0.77730256, 0.24956441, 0.30919433], [0.38325134, 0.41652393, 0.074524485]], "tornado_mean_prob_computed_220" => [[0.9081216, 0.107832894, 0.050864797], [0.800826, 0.23256853, 0.31022668], [0.40040252, 0.4055984, 0.067884356]], "tornado_mean_prob_computed_climatology_253" => [[0.982298, 0.05099982, 0.06179544], [0.80724055, 0.13905986, -0.0778435], [0.58624, 0.255771, -0.13458002]], "tornado_mean_prob_138" => [[0.93436664, 0.07926717, -0.026851058], [0.9042307, 0.17261904, 0.32986057], [0.46952277, 0.3558928, 0.015089741]], "tornado_mean_prob_computed_climatology_grads_1348" => [[0.94020444, 0.0850467, 0.051697694], [1.1303655, -0.08044045, -0.13138889], [0.63580066, 0.1437665, -0.37667832]], "tornado_mean_prob_computed_climatology_blurs_910" => [[0.98352575, 0.03981221, 0.010008344], [0.9775675, 0.045042068, -0.0055927183], [0.57631963, 0.20392773, -0.28303716]], "tornado_mean_prob_computed_climatology_blurs_grads_2005" => [[0.95560277, 0.05098174, -0.06451891], [1.1089742, -0.043572877, -0.017801635], [0.64432305, 0.10341309, -0.47638172]], "tornado_mean_prob_computed_climatology_3hr_1567" => [[0.96216094, 0.034555722, -0.12364115], [1.0485787, -0.011567661, -0.047783367], [0.6019041, 0.21778609, -0.17962365]])





# 6. prediction is weighted mean of the two overlapping logistic models
# 7. predictions should thereby be calibrated (check)


import Dates
import Printf

push!(LOAD_PATH, (@__DIR__) * "/../shared")
# import TrainGBDTShared
import TrainingShared
import LogisticRegression
using Metrics

push!(LOAD_PATH, @__DIR__)
import HREFPredictionAblations

push!(LOAD_PATH, (@__DIR__) * "/../../lib")
import Forecasts
import Inventories
import StormEvents

MINUTE = 60 # seconds
HOUR   = 60*MINUTE

(_, day_validation_forecasts, _) = TrainingShared.forecasts_train_validation_test(HREFPredictionAblations.forecasts_day(); just_hours_near_storm_events = false);

length(day_validation_forecasts)

# We don't have storm events past this time.
cutoff = Dates.DateTime(2022, 6, 1, 12)
day_validation_forecasts = filter(forecast -> Forecasts.valid_utc_datetime(forecast) < cutoff, day_validation_forecasts);

# Make sure a forecast loads
@time Forecasts.data(day_validation_forecasts[10])

day_validation_forecasts_0z_12z = filter(forecast -> forecast.run_hour == 0 || forecast.run_hour == 12, day_validation_forecasts);
length(day_validation_forecasts_0z_12z) # Expected: 358
# 358

event_name_to_day_labeler = Dict(
  "tornado" => TrainingShared.event_name_to_day_labeler["tornado"]
)

# rm("day_validation_forecasts_0z_12z_with_sig_gated"; recursive = true)

X, Ys, weights =
  TrainingShared.get_data_labels_weights(
    day_validation_forecasts_0z_12z;
    event_name_to_labeler = event_name_to_day_labeler,
    save_dir = "day_validation_forecasts_0z_12z_with_sig_gated",
  );

# Confirm that the combined is better than the accs
function test_predictive_power(forecasts, X, Ys, weights)
  inventory = Forecasts.inventory(forecasts[1])

  for feature_i in 1:length(inventory)
    prediction_i = feature_i
    model_name, _, _, _, _ = HREFPredictionAblations.models[prediction_i]
    y = Ys["tornado"]
    x = @view X[:,feature_i]
    au_pr_curve = Metrics.area_under_pr_curve(x, y, weights)
    println("$model_name ($(sum(y))) feature $feature_i $(Inventories.inventory_line_description(inventory[feature_i]))\tAU-PR-curve: $au_pr_curve")
  end
end
test_predictive_power(day_validation_forecasts_0z_12z, X, Ys, weights)

# tornado_mean_58                                           (20606.0) feature 1  TORPROB:calculated:hour fcst:calculated_prob: AU-PR-curve: 0.13337453
# tornado_prob_80                                           (20606.0) feature 2  TORPROB:calculated:hour fcst:calculated_prob: AU-PR-curve: 0.13071994
# tornado_mean_prob_138                                     (20606.0) feature 3  TORPROB:calculated:hour fcst:calculated_prob: AU-PR-curve: 0.14034203
# tornado_mean_prob_computed_no_sv_219                      (20606.0) feature 4  TORPROB:calculated:hour fcst:calculated_prob: AU-PR-curve: 0.13838942
# tornado_mean_prob_computed_220                            (20606.0) feature 5  TORPROB:calculated:hour fcst:calculated_prob: AU-PR-curve: 0.13780798
# tornado_mean_prob_computed_partial_climatology_227        (20606.0) feature 6  TORPROB:calculated:hour fcst:calculated_prob: AU-PR-curve: 0.14764147
# tornado_mean_prob_computed_climatology_253                (20606.0) feature 7  TORPROB:calculated:hour fcst:calculated_prob: AU-PR-curve: 0.14896595
# tornado_mean_prob_computed_climatology_blurs_910          (20606.0) feature 8  TORPROB:calculated:hour fcst:calculated_prob: AU-PR-curve: 0.14475623
# tornado_mean_prob_computed_climatology_grads_1348         (20606.0) feature 9  TORPROB:calculated:hour fcst:calculated_prob: AU-PR-curve: 0.14915858
# tornado_mean_prob_computed_climatology_blurs_grads_2005   (20606.0) feature 10 TORPROB:calculated:hour fcst:calculated_prob: AU-PR-curve: 0.15263174
# tornado_mean_prob_computed_climatology_prior_next_hrs_691 (20606.0) feature 11 TORPROB:calculated:hour fcst:calculated_prob: AU-PR-curve: 0.15355374
# tornado_mean_prob_computed_climatology_3hr_1567           (20606.0) feature 12 TORPROB:calculated:hour fcst:calculated_prob: AU-PR-curve: 0.1625404 ***best***
# tornado_full_13831                                        (20606.0) feature 13 TORPROB:calculated:hour fcst:calculated_prob: AU-PR-curve: 0.1562636 (second best)


# test y vs ŷ
Metrics.reliability_curves_midpoints(20, X, Ys, fill("tornado", size(X,2)), weights, map(m -> m[1], HREFPredictionAblations.models))
# ŷ_tornado_mean_58,y_tornado_mean_58,ŷ_tornado_prob_80,y_tornado_prob_80,ŷ_tornado_mean_prob_138,y_tornado_mean_prob_138,ŷ_tornado_mean_prob_computed_no_sv_219,y_tornado_mean_prob_computed_no_sv_219,ŷ_tornado_mean_prob_computed_220,y_tornado_mean_prob_computed_220,ŷ_tornado_mean_prob_computed_partial_climatology_227,y_tornado_mean_prob_computed_partial_climatology_227,ŷ_tornado_mean_prob_computed_climatology_253,y_tornado_mean_prob_computed_climatology_253,ŷ_tornado_mean_prob_computed_climatology_blurs_910,y_tornado_mean_prob_computed_climatology_blurs_910,ŷ_tornado_mean_prob_computed_climatology_grads_1348,y_tornado_mean_prob_computed_climatology_grads_1348,ŷ_tornado_mean_prob_computed_climatology_blurs_grads_2005,y_tornado_mean_prob_computed_climatology_blurs_grads_2005,ŷ_tornado_mean_prob_computed_climatology_prior_next_hrs_691,y_tornado_mean_prob_computed_climatology_prior_next_hrs_691,ŷ_tornado_mean_prob_computed_climatology_3hr_1567,y_tornado_mean_prob_computed_climatology_3hr_1567,ŷ_tornado_full_13831,y_tornado_full_13831,
# 0.00012071146,0.00012142477,0.0001259804,0.000120114724,0.00011867599,0.00012014826,0.00011784681,0.00012008569,0.000118070246,0.00012010849,0.0001208754,0.000119486154,0.00011644108,0.00011933012,0.0001134045,0.00011855099,0.00011320543,0.00011861816,0.00011354278,0.00011860465,0.00011861205,0.00011927369,0.000114871946,0.00011947558,0.00011889208,0.00011867229,
# 0.002271064,0.0024861163,0.002456562,0.0027921458,0.002397379,0.0026742741,0.0024407767,0.002522438,0.00245275,0.002562878,0.0024860275,0.0027724577,0.0025658777,0.0027210051,0.0026995738,0.002883395,0.00281056,0.0028018756,0.0027711005,0.0027079792,0.002606789,0.0027468465,0.0024916006,0.0025311643,0.0025425232,0.0029419851,
# 0.00517296,0.004446982,0.00513114,0.005768481,0.005544284,0.004740182,0.005811874,0.005130061,0.0058018044,0.004824105,0.0057727513,0.0054363892,0.0060150614,0.005712267,0.006611966,0.0055141146,0.006557861,0.0062334295,0.006767551,0.005920978,0.005879146,0.0058539785,0.006075227,0.0058954824,0.006280865,0.0051744217,
# 0.0090630045,0.008619554,0.008705182,0.0074339034,0.009879869,0.009367985,0.009986734,0.009997601,0.010089848,0.0105461,0.01009232,0.009412196,0.010503849,0.009613198,0.011585749,0.011448514,0.010570923,0.012023648,0.011349587,0.011879041,0.010224079,0.008981483,0.010405652,0.010096569,0.011003303,0.011740553,
# 0.0130730355,0.013551233,0.013373518,0.012188732,0.014342726,0.01532461,0.014339409,0.015671838,0.014518239,0.015064564,0.014914949,0.014577598,0.015731273,0.014408908,0.016415864,0.017854238,0.014950083,0.01515616,0.015887404,0.018828992,0.015358004,0.014606532,0.015345235,0.01454988,0.015415692,0.017903777,
# 0.017281488,0.018190786,0.018641012,0.01765438,0.01901382,0.020220272,0.019125445,0.020723006,0.019538807,0.021218508,0.019973515,0.022535255,0.021284822,0.02414975,0.02187849,0.021140657,0.02069984,0.019219836,0.021489242,0.019330926,0.020617163,0.022768982,0.020908637,0.021014776,0.020695614,0.020229997,
# 0.022092605,0.023840385,0.02495752,0.023507189,0.024457902,0.025269296,0.024850337,0.025778022,0.025319926,0.026890244,0.025484642,0.026348904,0.027096964,0.028777717,0.028676976,0.02642947,0.02770126,0.02631414,0.028379884,0.029794013,0.02660333,0.026965313,0.027103111,0.027470408,0.02752036,0.026117474,
# 0.02779648,0.028929297,0.0324379,0.032491375,0.03081418,0.033220157,0.032039147,0.029432412,0.032620743,0.030305402,0.03243687,0.031235635,0.03398238,0.03416862,0.036338367,0.04005221,0.035714626,0.034006216,0.036224727,0.035364065,0.034124237,0.03472666,0.03396969,0.037125856,0.035524398,0.03829976,
# 0.035024475,0.032472663,0.04088735,0.0448332,0.038926274,0.0354368,0.040945087,0.03940877,0.041609827,0.040420253,0.04124619,0.03833522,0.041905485,0.04277431,0.0446643,0.0490739,0.044988193,0.04630144,0.045563463,0.048651915,0.042702317,0.043673076,0.04212248,0.043158732,0.045094907,0.04435019,
# 0.04422248,0.04191309,0.050935436,0.05310833,0.05013001,0.044606987,0.051523827,0.04912772,0.05221129,0.049219273,0.051407102,0.050494324,0.050974194,0.048367094,0.05459126,0.05435662,0.055125978,0.060214713,0.056801945,0.05336652,0.053092685,0.048483346,0.053284086,0.047899917,0.057168424,0.053338945,
# 0.05508904,0.05230053,0.062806785,0.06460878,0.06347156,0.059312087,0.063460365,0.067794465,0.063950256,0.06618239,0.06367962,0.061330743,0.06317665,0.05498354,0.06740698,0.06307182,0.067980364,0.06625206,0.071271226,0.06813841,0.06635921,0.058760975,0.06802663,0.06258979,0.07256332,0.06717904,
# 0.06672642,0.0660846,0.075081415,0.07413087,0.077146634,0.07708511,0.07637722,0.07946731,0.07631496,0.07741203,0.07810112,0.07557899,0.07788247,0.07702475,0.08287006,0.075163364,0.083397724,0.08345333,0.08608646,0.09365479,0.08091855,0.08153977,0.084404156,0.0854176,0.08858901,0.09274736,
# 0.07910345,0.082870595,0.08701346,0.090651885,0.08997507,0.103070505,0.08987598,0.093189985,0.08954995,0.08932212,0.09338713,0.0949043,0.09387662,0.08804413,0.099365845,0.0924961,0.09873477,0.106722146,0.10070234,0.109391995,0.09626924,0.09955342,0.101417795,0.10438744,0.10420568,0.10806544,
# 0.09173134,0.10604271,0.0988401,0.101018235,0.102312334,0.11708655,0.10483774,0.096190915,0.1035953,0.105044775,0.11016223,0.10922478,0.111344315,0.12046822,0.11478873,0.13078278,0.11385895,0.12310494,0.11654705,0.11494037,0.112162165,0.1244973,0.11988778,0.12606347,0.12102106,0.116944864,
# 0.10627823,0.10312076,0.11138405,0.1109809,0.11660524,0.115794905,0.121620186,0.1215652,0.119468085,0.12200825,0.12859467,0.1439605,0.12952434,0.1439887,0.1296558,0.13811475,0.1294583,0.12132393,0.13225728,0.13825235,0.12934951,0.14331663,0.13925786,0.15131047,0.13717742,0.14863971,
# 0.12551506,0.11773958,0.12636867,0.12440474,0.13570647,0.13115293,0.14045475,0.15221322,0.13726063,0.15091701,0.14972524,0.1639421,0.15068467,0.18062814,0.14758159,0.16531236,0.14910862,0.14884461,0.1511056,0.14371198,0.15200722,0.15091008,0.16277175,0.16753264,0.15673207,0.17233202,
# 0.15009291,0.1574015,0.1482236,0.12759584,0.16113482,0.15564914,0.16346017,0.18360163,0.16051283,0.16708918,0.17732947,0.17369291,0.17596525,0.19416153,0.1705173,0.1919997,0.1733895,0.17614278,0.17654265,0.16519047,0.18093923,0.18497087,0.19224797,0.19180289,0.1803576,0.19112791,
# 0.18271302,0.19324741,0.18050222,0.16734658,0.19822647,0.19790116,0.19498695,0.20285836,0.191358,0.20342231,0.21573396,0.21345873,0.21203355,0.21416746,0.20010602,0.21687044,0.2076502,0.21258742,0.21104977,0.21199043,0.21763322,0.21373317,0.22983074,0.23547776,0.21512035,0.2046197,
# 0.2385189,0.23899543,0.22814442,0.23061067,0.25625628,0.27654657,0.24335822,0.25162053,0.23976275,0.23310381,0.26520988,0.28117588,0.26657808,0.26722795,0.24316965,0.27065727,0.26099584,0.31753471,0.26336107,0.32327437,0.26869124,0.29174402,0.28188998,0.30103546,0.27166343,0.3318058,
# 0.3516174,0.33218774,0.3059416,0.3405871,0.35183877,0.33223674,0.33491457,0.30652535,0.3363033,0.3177617,0.3705483,0.34394422,0.38405252,0.335215,0.35289577,0.29500008,0.37148607,0.32835212,0.37556997,0.34313023,0.3933678,0.351901,0.40980762,0.36972505,0.3821813,0.3331822,






# Calibrate to SPC
# The targets below are computing in and copied from models/spc_outlooks/Stats.jl

target_warning_ratios = Dict{String,Vector{Tuple{Float64,Float64}}}(
  "tornado" => [
    (0.02, 0.025348036),
    (0.05, 0.007337035),
    (0.1,  0.0014824796),
    (0.15, 0.00025343563),
    (0.3,  3.7446924e-5),
    (0.45, 3.261123e-6),
  ],
  "wind" => [
    (0.05, 0.07039726),
    (0.15, 0.021633422),
    (0.3,  0.0036298542),
    (0.45, 0.0004162882),
  ],
  "hail" => [
    (0.05, 0.052633155),
    (0.15, 0.015418012),
    (0.3,  0.0015550428),
    (0.45, 8.9432746e-5),
  ],
  "sig_tornado" => [(0.1, 0.0009527993)],
  "sig_wind"    => [(0.1, 0.0014686467)],
  "sig_hail"    => [(0.1, 0.002794325)],
)


# Assumes weights are proportional to gridpoint areas
# (here they are because we are not do any fancy subsetting)
function spc_calibrate_warning_ratio(event_name, model_name, prediction_i, X, Ys, weights)
  y = Ys[event_name]
  ŷ = @view X[:, prediction_i]

  thresholds_to_match_warning_ratio =
    map(target_warning_ratios[event_name]) do (nominal_prob, target_warning_ratio)
      threshold = 0.5f0
      step = 0.25f0
      while step > 0.000001f0
        wr = Metrics.warning_ratio(ŷ, weights, threshold)
        if isnan(wr) || wr > target_warning_ratio
          threshold += step
        else
          threshold -= step
        end
        step *= 0.5f0
      end
      # println("$nominal_prob\t$threshold\t$(probability_of_detection(ŷ, y, weights, threshold))")
      threshold
    end

  wr_thresholds = Tuple{Float32,Float32}[]
  for i in 1:length(target_warning_ratios[event_name])
    nominal_prob, _ = target_warning_ratios[event_name][i]
    threshold_to_match_warning_ratio = thresholds_to_match_warning_ratio[i]
    sr  = Float32(Metrics.success_ratio(ŷ, y, weights, threshold_to_match_warning_ratio))
    pod = Float32(Metrics.probability_of_detection(ŷ, y, weights, threshold_to_match_warning_ratio))
    wr  = Float32(Metrics.warning_ratio(ŷ, weights, threshold_to_match_warning_ratio))
    println("$model_name\t$nominal_prob\t$threshold_to_match_warning_ratio\t$sr\t$pod\t$wr")
    push!(wr_thresholds, (Float32(nominal_prob), Float32(threshold_to_match_warning_ratio)))
  end

  wr_thresholds
end

println("model_name\tnominal_prob\tthreshold_to_match_warning_ratio\tSR\tPOD\tWR")
calibrations_wr = Dict{String,Vector{Tuple{Float32,Float32}}}()
for prediction_i in 1:length(HREFPredictionAblations.models)
  model_name, _, _, _, _ = HREFPredictionAblations.models[prediction_i]
  calibrations_wr[model_name] = spc_calibrate_warning_ratio("tornado", model_name, prediction_i, X, Ys, weights)
end

# event_name                                                 nominal_prob threshold_to_match_warning_ratio SR          POD           WR
# tornado_mean_58                                           0.02         0.018461227                      0.060353596 0.7140549     0.025349844
# tornado_mean_58                                           0.05         0.06462288                       0.12687075  0.4344617     0.0073373094
# tornado_mean_58                                           0.1          0.15732765                       0.23343594  0.16152214    0.0014825562
# tornado_mean_58                                           0.15         0.29961205                       0.34252933  0.040531665   0.0002535383
# tornado_mean_58                                           0.3          0.42596245                       0.503072    0.008781126   3.7399597e-5
# tornado_mean_58                                           0.45         0.5749607                        0.6667791   0.0010088343  3.2417893e-6
# tornado_prob_80                                           0.02         0.017969131                      0.061899357 0.7323091     0.025348669
# tornado_prob_80                                           0.05         0.072660446                      0.12734881  0.43605593    0.007336589
# tornado_prob_80                                           0.1          0.16031837                       0.22367981  0.15477388    0.0014825786
# tornado_prob_80                                           0.15         0.27256584                       0.34531942  0.040848665   0.00025345667
# tornado_prob_80                                           0.3          0.35550117                       0.44686568  0.007807553   3.74356e-5
# tornado_prob_80                                           0.45         0.43262672                       0.44603005  0.0006903442  3.3162592e-6
# tornado_mean_prob_138                                     0.02         0.018060684                      0.06216014  0.7353488     0.025347099
# tornado_mean_prob_138                                     0.05         0.068006516                      0.13396381  0.45875457    0.007337359
# tornado_mean_prob_138                                     0.1          0.16544151                       0.24346636  0.16844217    0.0014823772
# tornado_mean_prob_138                                     0.15         0.30457115                       0.3303532   0.03906655    0.00025338065
# tornado_mean_prob_138                                     0.3          0.4305973                        0.44296402  0.007753712   3.7504906e-5
# tornado_mean_prob_138                                     0.45         0.54815865                       0.48427862  0.00075647386 3.3469214e-6
# tornado_mean_prob_computed_no_sv_219                      0.02         0.017709732                      0.06253015  0.73969764    0.02534613
# tornado_mean_prob_computed_no_sv_219                      0.05         0.06767845                       0.1339838   0.4587967     0.0073369383
# tornado_mean_prob_computed_no_sv_219                      0.1          0.16807747                       0.24200656  0.16745506    0.0014825796
# tornado_mean_prob_computed_no_sv_219                      0.15         0.29662895                       0.30785885  0.03640122    0.0002533443
# tornado_mean_prob_computed_no_sv_219                      0.3          0.40353203                       0.43347666  0.007558801   3.736234e-5
# tornado_mean_prob_computed_no_sv_219                      0.45         0.47664833                       0.6001471   0.0009090007  3.2452888e-6
# tornado_mean_prob_computed_220                            0.02         0.017793655                      0.06276276  0.74256194    0.025349975
# tornado_mean_prob_computed_220                            0.05         0.068590164                      0.13312294  0.45586365    0.0073371762
# tornado_mean_prob_computed_220                            0.1          0.16698647                       0.23364866  0.16165       0.0014823792
# tornado_mean_prob_computed_220                            0.15         0.29432106                       0.3243835   0.0383803     0.0002535108
# tornado_mean_prob_computed_220                            0.3          0.40807152                       0.43061838  0.007547724   3.7555223e-5
# tornado_mean_prob_computed_220                            0.45         0.5101414                        0.59985024  0.00091232895 3.2587836e-6
# tornado_mean_prob_computed_partial_climatology_227        0.02         0.017900467                      0.06313051  0.74691       0.025349876
# tornado_mean_prob_computed_partial_climatology_227        0.05         0.065675735                      0.13676794  0.46833116    0.007336951
# tornado_mean_prob_computed_partial_climatology_227        0.1          0.17772484                       0.25096172  0.17363195    0.0014824125
# tornado_mean_prob_computed_partial_climatology_227        0.15         0.3134899                        0.34055257  0.04026637    0.0002533408
# tornado_mean_prob_computed_partial_climatology_227        0.3          0.45659447                       0.3983758   0.0069803614  3.7543246e-5
# tornado_mean_prob_computed_partial_climatology_227        0.45         0.56458473                       0.5016012   0.0007418342  3.1688023e-6
# tornado_mean_prob_computed_climatology_253                0.02         0.018064499                      0.06386594  0.75553435    0.025347305
# tornado_mean_prob_computed_climatology_253                0.05         0.06471443                       0.13714375  0.46963733    0.0073372526
# tornado_mean_prob_computed_climatology_253                0.1          0.17611122                       0.25220838  0.17450167    0.0014824736
# tornado_mean_prob_computed_climatology_253                0.15         0.32598686                       0.35081702  0.041498132   0.00025345144
# tornado_mean_prob_computed_climatology_253                0.3          0.47203255                       0.4024308   0.0070368825  3.746588e-5
# tornado_mean_prob_computed_climatology_253                0.45         0.60515404                       0.535088    0.0007972568  3.192419e-6
# tornado_mean_prob_computed_climatology_blurs_910          0.02         0.01799202                       0.06419382  0.75942755    0.025347784
# tornado_mean_prob_computed_climatology_blurs_910          0.05         0.06854057                       0.1377441   0.47169042    0.0073372093
# tornado_mean_prob_computed_climatology_blurs_910          0.1          0.17239952                       0.24644762  0.1705114     0.0014824349
# tornado_mean_prob_computed_climatology_blurs_910          0.15         0.29992485                       0.29077744  0.034393243   0.00025343074
# tornado_mean_prob_computed_climatology_blurs_910          0.3          0.46686745                       0.32967794  0.0057634856  3.7457794e-5
# tornado_mean_prob_computed_climatology_blurs_910          0.45         0.6732578                        0.4334652   0.00063932553 3.1602005e-6
# tornado_mean_prob_computed_climatology_grads_1348         0.02         0.01742363                       0.063572474 0.7520665     0.025347432
# tornado_mean_prob_computed_climatology_grads_1348         0.05         0.06785774                       0.13877265  0.47520727    0.0073371273
# tornado_mean_prob_computed_climatology_grads_1348         0.1          0.17230797                       0.25626767  0.17731273    0.0014824942
# tornado_mean_prob_computed_climatology_grads_1348         0.15         0.3162861                        0.32335854  0.038250607   0.00025345501
# tornado_mean_prob_computed_climatology_grads_1348         0.3          0.45212746                       0.40177268  0.0070315073  3.7498587e-5
# tornado_mean_prob_computed_climatology_grads_1348         0.45         0.59908485                       0.35197034  0.00054971856 3.346426e-6
# tornado_mean_prob_computed_climatology_blurs_grads_2005   0.02         0.017370224                      0.064065866 0.7578653     0.02534616
# tornado_mean_prob_computed_climatology_blurs_grads_2005   0.05         0.06944466                       0.14094351  0.48263273    0.0073370007
# tornado_mean_prob_computed_climatology_blurs_grads_2005   0.1          0.17526436                       0.2535033   0.17538975    0.0014824071
# tornado_mean_prob_computed_climatology_blurs_grads_2005   0.15         0.31707573                       0.33597314  0.03975551    0.000253536
# tornado_mean_prob_computed_climatology_blurs_grads_2005   0.3          0.45667458                       0.46627226  0.008150536   3.7453592e-5
# tornado_mean_prob_computed_climatology_blurs_grads_2005   0.45         0.6287174                        0.36439404  0.0005484284  3.2247463e-6
# tornado_mean_prob_computed_climatology_prior_next_hrs_691 0.02         0.017599106                      0.06391611  0.7561532     0.025348153
# tornado_mean_prob_computed_climatology_prior_next_hrs_691 0.05         0.06641579                       0.1386232   0.47470704    0.007337306
# tornado_mean_prob_computed_climatology_prior_next_hrs_691 0.1          0.17908287                       0.25438872  0.17601533    0.0014825164
# tornado_mean_prob_computed_climatology_prior_next_hrs_691 0.15         0.3228054                        0.36091423  0.04270103    0.0002535019
# tornado_mean_prob_computed_climatology_prior_next_hrs_691 0.3          0.49178123                       0.4895547   0.008574016   3.75258e-5
# tornado_mean_prob_computed_climatology_prior_next_hrs_691 0.45         0.6537876                        0.63721466  0.0009548305  3.2106086e-6
# tornado_mean_prob_computed_climatology_3hr_1567           0.02         0.017438889                      0.0639874   0.75698364    0.02534772
# tornado_mean_prob_computed_climatology_3hr_1567           0.05         0.0646534                        0.141945    0.48607287    0.0073371627
# tornado_mean_prob_computed_climatology_3hr_1567           0.1          0.18551826                       0.26718888  0.18487404    0.0014825333
# tornado_mean_prob_computed_climatology_3hr_1567           0.15         0.3344822                        0.38190585  0.045154948   0.00025333543
# tornado_mean_prob_computed_climatology_3hr_1567           0.3          0.50450325                       0.58447576  0.010206408   3.7415648e-5
# tornado_mean_prob_computed_climatology_3hr_1567           0.45         0.6276798                        0.71180487  0.0011086512  3.3371891e-6
# tornado_full_13831                                        0.02         0.016950607                      0.064026326 0.7574176     0.025346832
# tornado_full_13831                                        0.05         0.06830406                       0.14281912  0.48904803    0.0073368903
# tornado_full_13831                                        0.1          0.17817497                       0.257954    0.17845055    0.0014822535
# tornado_full_13831                                        0.15         0.3255825                        0.32377487  0.03830062    0.00025346005
# tornado_full_13831                                        0.3          0.4591999                        0.48384824  0.008458492   3.7456797e-5
# tornado_full_13831                                        0.45         0.60011864                       0.49274874  0.00073567254 3.1989384e-6
# 2021v1_tornado                                            0.02         0.017892838                      0.06162817  0.7073454     0.019615944
# 2021v1_tornado                                            0.05         0.07787514                       0.14183877  0.4307434     0.0051901583
# 2021v1_tornado                                            0.1          0.17152214                       0.2464791   0.14091049    0.0009770577
# 2021v1_tornado                                            0.15         0.2814541                        0.26955867  0.026589876   0.0001685854
# 2021v1_tornado                                            0.3          0.3905239                        0.34836945  0.007654115   3.7550166e-5
# 2021v1_tornado                                            0.45         0.6009083                        0.47030112  0.0008843078  3.213545e-6

println(calibrations_wr)
# Dict{String, Vector{Tuple{Float32, Float32}}}("tornado_mean_prob_computed_partial_climatology_227" => [(0.02, 0.017900467), (0.05, 0.065675735), (0.1, 0.17772484), (0.15, 0.3134899), (0.3, 0.45659447), (0.45, 0.56458473)], "tornado_mean_58" => [(0.02, 0.018461227), (0.05, 0.06462288), (0.1, 0.15732765), (0.15, 0.29961205), (0.3, 0.42596245), (0.45, 0.5749607)], "tornado_mean_prob_computed_climatology_prior_next_hrs_691" => [(0.02, 0.017599106), (0.05, 0.06641579), (0.1, 0.17908287), (0.15, 0.3228054), (0.3, 0.49178123), (0.45, 0.6537876)], "tornado_prob_80" => [(0.02, 0.017969131), (0.05, 0.072660446), (0.1, 0.16031837), (0.15, 0.27256584), (0.3, 0.35550117), (0.45, 0.43262672)], "tornado_full_13831" => [(0.02, 0.016950607), (0.05, 0.06830406), (0.1, 0.17817497), (0.15, 0.3255825), (0.3, 0.4591999), (0.45, 0.60011864)], "tornado_mean_prob_computed_no_sv_219" => [(0.02, 0.017709732), (0.05, 0.06767845), (0.1, 0.16807747), (0.15, 0.29662895), (0.3, 0.40353203), (0.45, 0.47664833)], "tornado_mean_prob_computed_220" => [(0.02, 0.017793655), (0.05, 0.068590164), (0.1, 0.16698647), (0.15, 0.29432106), (0.3, 0.40807152), (0.45, 0.5101414)], "tornado_mean_prob_computed_climatology_253" => [(0.02, 0.018064499), (0.05, 0.06471443), (0.1, 0.17611122), (0.15, 0.32598686), (0.3, 0.47203255), (0.45, 0.60515404)], "tornado_mean_prob_138" => [(0.02, 0.018060684), (0.05, 0.068006516), (0.1, 0.16544151), (0.15, 0.30457115), (0.3, 0.4305973), (0.45, 0.54815865)], "tornado_mean_prob_computed_climatology_grads_1348" => [(0.02, 0.01742363), (0.05, 0.06785774), (0.1, 0.17230797), (0.15, 0.3162861), (0.3, 0.45212746), (0.45, 0.59908485)], "tornado_mean_prob_computed_climatology_blurs_910" => [(0.02, 0.01799202), (0.05, 0.06854057), (0.1, 0.17239952), (0.15, 0.29992485), (0.3, 0.46686745), (0.45, 0.6732578)], "tornado_mean_prob_computed_climatology_blurs_grads_2005" => [(0.02, 0.017370224), (0.05, 0.06944466), (0.1, 0.17526436), (0.15, 0.31707573), (0.3, 0.45667458), (0.45, 0.6287174)], "tornado_mean_prob_computed_climatology_3hr_1567" => [(0.02, 0.017438889), (0.05, 0.0646534), (0.1, 0.18551826), (0.15, 0.3344822), (0.3, 0.50450325), (0.45, 0.6276798)])
