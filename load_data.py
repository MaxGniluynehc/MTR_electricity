import pandas as pd
import numpy as np
import pickle
import os
from datetime import datetime
from data_util_fn import fit_iterative_SLR

data_PATH = "/Users/maxchen/Documents/Working/ASTRI——Summer2024/MTR_electricity/dataset"
# data_PATH = "cleaned_data"

quick_load_data = True
if quick_load_data:
    print("Try quick loading data...")
    try:
        print("Reloading iterative linear model data from saved directory...")
        with open(data_PATH + "/iter_lms_KBD", "rb") as f1:
            iter_lms_KBD = pickle.load(f1)
        with open(data_PATH + "/iter_lms_TIS", "rb") as f2:
            iter_lms_TIS = pickle.load(f2)
        with open(data_PATH + "/iter_lms_KBD_subs", "rb") as f1s:
            iter_lms_KBD_subs = pickle.load(f1s)
        with open(data_PATH + "/iter_lms_TIS_subs", "rb") as f2s:
            iter_lms_TIS_subs = pickle.load(f2s)

        print("Reloading peak features from saved directory...")
        with open(data_PATH+"/peak_features_KBD", "rb") as f1:
            peak_features_KBD = pickle.load(f1)
        with open(data_PATH+"/peak_features_TIS", "rb") as f2:
            peak_features_TIS = pickle.load(f2)

    except:
        print("Some of the cleaned data wasn't saved, so reloading them from source...")
        problem_timestamp_TIS = [datetime(2023,3,27,2,5,20)]
        include_incycle_features = True
        if not os.path.exists(data_PATH + "/mva_2022_23_cleaned.csv"):
            print("Cleaning data with time-interpolation of NaNs...")
            df = pd.read_csv(data_PATH + "/mva_2022_23.csv")
            # df = pd.read_csv("mva_2022_23.csv")
            df = df.set_index("time")
            df.index = pd.to_datetime(df.index)
            df = df.sort_index()
            df.loc[problem_timestamp_TIS, "TIS_MAXDMD_CLP"] = np.NaN
            df_ = df.interpolate("time")

            for idx in [0,1]:
                i = np.argwhere((df_.values[:-1,idx] > df_.values[1:,idx])).flatten()
                ii = np.argwhere(i[1:] - i[:-1] == 1).flatten()
                df_.iloc[i[1:][ii], idx] = 0
            df_.iloc[0,0] = 100
            test_df_clearness = True
            if test_df_clearness:
                for idx in [0, 1]:
                    i = np.argwhere((df_.values[:-1, idx] > df_.values[1:, idx])).flatten()
                    ii = np.argwhere(i[1:] - i[:-1] == 1).flatten()
                    assert len(ii) == 0, AssertionError("Data clearning error! Some nans are not properly reset")

                if not pd.isna(df_.values).all():
                    ValueError("df_ contains NaNs!")

            if include_incycle_features:
                print("Cleaning data and adding incycle features...")
                import holidays

                hk_holidays = holidays.country_holidays('HK')

                def check_if_on_peak(dt):
                    if dt.weekday() >= 6:
                        return 0
                    elif dt in hk_holidays:
                        return 0
                    elif dt.hour < 9 or dt.hour > 21:
                        return 0
                    else:
                        return 1

                def check_rush_idle(row):
                    rush_idle_ind = 0
                    if row['on_peak_ind'] == 1 and row['dt'].hour >= 9 and row['dt'].hour <= 10:
                        rush_idle_ind = 1
                    elif row['on_peak_ind'] == 1 and row['dt'].hour >= 17 and row['dt'].hour <= 20:
                        rush_idle_ind = 1
                    elif row['on_peak_ind'] == 0 and row['dt'].hour >= 7 and row['dt'].hour <= 9:
                        rush_idle_ind = 1
                    return rush_idle_ind

                df_["dt"] = pd.to_datetime(df_.index.values)
                df_['on_peak_ind'] = df_['dt'].apply(check_if_on_peak)
                df_['rush_idle_ind'] = df_.apply(check_rush_idle, axis=1)
                df_['hour'] = pd.to_datetime(df_.dt.values).hour
            df_.to_csv(data_PATH + "/mva_2022_23_cleaned.csv")
        else:
            print("Loading data from previously saved directory...")
            df_ = pd.read_csv(data_PATH + "/mva_2022_23_cleaned.csv", index_col="time")
            df_.index = pd.to_datetime(df_.index)

        load_df_tgt_ = True
        if load_df_tgt_:
            df_tgt = pd.read_csv(data_PATH+"/mva_2022_23_target.csv")
            df_tgt = df_tgt.set_index("cycle")
            df_tgt.index = pd.to_datetime(df_tgt.index)
            df_tgt = df_tgt.sort_index()
            df_tgt_ = df_tgt.interpolate("time")

        # ========================== Add In-Cycle Features =========================== #
        # import holidays
        # hk_holidays = holidays.country_holidays('HK')
        #
        #
        # def check_if_on_peak(dt):
        #     if dt.weekday() >= 6:
        #         return 0
        #     elif dt in hk_holidays:
        #         return 0
        #     elif dt.hour < 9 or dt.hour > 21:
        #         return 0
        #     else:
        #         return 1
        #
        #
        # def check_rush_idle(row):
        #     rush_idle_ind = 0
        #     if row['on_peak_ind'] == 1 and row['dt'].hour >= 9 and row['dt'].hour <= 10:
        #         rush_idle_ind = 1
        #     elif row['on_peak_ind'] == 1 and row['dt'].hour >= 17 and row['dt'].hour <= 20:
        #         rush_idle_ind = 1
        #     elif row['on_peak_ind'] == 0 and row['dt'].hour >= 7 and row['dt'].hour <= 9:
        #         rush_idle_ind = 1
        #     return rush_idle_ind
        #
        #
        # df_["dt"] = pd.to_datetime(df_.index.values)
        # df_['on_peak_ind'] = df_['dt'].apply(check_if_on_peak)
        # df_['rush_idle_ind'] = df_.apply(check_rush_idle, axis=1)
        # df_['hour'] = pd.to_datetime(df_.dt).hour



        # ================================== EDA ================================== #

        EDA = False
        if EDA:
            import os
            import matplotlib
            from matplotlib import pyplot as plt
            matplotlib.use("TkAgg")

            os.makedirs("plots/EDA", exist_ok=True)

            fig,ax = plt.subplots(2,1)
            ax[0].plot(df["KBD_MAXDMD_CLP"][:167], label="KBD_MAXDMD_CLP")
            ax[1].plot(df["TIS_MAXDMD_CLP"][:167], label="TIS_MAXDMD_CLP")
            ax[0].legend()
            ax[1].legend()
            fig.savefig("plots/EDA/mva_0-10k_signals")

            fig,ax = plt.subplots(1,1)
            ax.plot(df_["KBD_MAXDMD_CLP"][:10000], label="KBD_MAXDMD_CLP")
            ax.plot(df_["TIS_MAXDMD_CLP"][:10000], label="TIS_MAXDMD_CLP")
            ax.plot(df_tgt_.loc[:df_.index[10000],], label="target")
            ax.legend()
            fig.savefig("plots/EDA/mva_0-10k_with_target")

            # pd.DataFrame.groupby()
            # df_["KBD_MAXDMD_CLP"][0:180]
            # df_["KBD_MAXDMD_CLP"][167]
            # df_["KBD_MAXDMD_CLP"][167:167+15]
            # df_["KBD_MAXDMD_CLP"][167:167*2]
            # df_["TIS_MAXDMD_CLP"][:167]
            # df_["TIS_MAXDMD_CLP"][167:167*2]

            fig,ax = plt.subplots(1,1)
            ax.plot(df_["KBD_MAXDMD_CLP"][:10000] + df_["TIS_MAXDMD_CLP"][:10000], label="KBD+TIS")
            # ax.plot(df_["TIS_MAXDMD_CLP"][:2000], label="TIS_MAXDMD_CLP")
            ax.plot(df_tgt_.loc[:df_.index[10000],], label="target")
            ax.legend()
            fig.savefig("plots/EDA/mva_agg_0-10k_with_target")


            fig,ax = plt.subplots(1,1)
            ax.plot(df_tgt_.values[:500], label="target")
            ax.legend()
            fig.savefig("plots/EDA/mva_target_0-5k")

        # ====================== Divide dataset in to subgroups ========================== #
        if any([(not os.path.exists(data_PATH+"/iter_lms_KBD")),
                (not os.path.exists(data_PATH+"/iter_lms_TIS")),
                (not os.path.exists(data_PATH+"/iter_lms_KBD_subs")),
                (not os.path.exists(data_PATH+"/iter_lms_TIS_subs"))]):
            print("Loading data via iterative Simple Linear Regressions...")
            signals = df_[["KBD_MAXDMD_CLP", "TIS_MAXDMD_CLP"]].values  # df_.values
            iter_lms_KBD, iter_lms_KBD_subs = fit_iterative_SLR(signals[:,0])
            iter_lms_TIS, iter_lms_TIS_subs = fit_iterative_SLR(signals[:,1])
            with open(data_PATH+"/iter_lms_KBD", "wb") as f1:
                pickle.dump(iter_lms_KBD, f1)
            with open(data_PATH+"/iter_lms_TIS", "wb") as f2:
                pickle.dump(iter_lms_TIS, f2)
            with open(data_PATH+"/iter_lms_KBD_subs", "wb") as f1s:
                pickle.dump(iter_lms_KBD_subs, f1s)
            with open(data_PATH+"/iter_lms_TIS_subs", "wb") as f2s:
                pickle.dump(iter_lms_TIS_subs, f2s)
        else:
            print("Reloading iterative linear model data from saved directory...")
            with open(data_PATH+"/iter_lms_KBD", "rb") as f1:
                iter_lms_KBD = pickle.load(f1)
            with open(data_PATH+"/iter_lms_TIS", "rb") as f2:
                iter_lms_TIS = pickle.load(f2)
            with open(data_PATH+"/iter_lms_KBD_subs", "rb") as f1s:
                iter_lms_KBD_subs = pickle.load(f1s)
            with open(data_PATH+"/iter_lms_TIS_subs", "rb") as f2s:
                iter_lms_TIS_subs = pickle.load(f2s)

        if any([not os.path.exists(data_PATH+"/peak_features_KBD"), not os.path.exists(data_PATH+"/peak_features_TIS")]):
            print("Loading peak features of starting of each cycle...")
            for name in ["KBD", "TIS"]:
                signal = df_["{}_MAXDMD_CLP".format(name)].values
                partitions = np.concatenate([np.array([-1]), np.argwhere(signal[1:] < signal[:-1]).flatten()])
                prtn_start_idx = partitions+1
                peak_features = np.column_stack([df_["on_peak_ind"].values[prtn_start_idx],
                                                     df_["rush_idle_ind"].values[prtn_start_idx],
                                                     df_["hour"].values[prtn_start_idx]])
                assert peak_features.shape[0] == locals()["iter_lms_{}".format(name)].shape[0], (
                    AssertionError("Peak feature dimension of KBD and iter_lms features unmatched! "))
                with open(data_PATH+"/peak_features_{}".format(name), "wb") as file:
                    pickle.dump(peak_features, file)
        else:
            print("Reloading peak features from saved directory...")
            with open(data_PATH+"/peak_features_KBD", "rb") as f1:
                peak_features_KBD = pickle.load(f1)
            with open(data_PATH+"/peak_features_TIS", "rb") as f2:
                peak_features_TIS = pickle.load(f2)

if False:
    from datetime import datetime
    import matplotlib
    from matplotlib import pyplot as plt

    matplotlib.use("TkAgg")
    start_date = df_.index[0]
    end_date = datetime(2022,2,6,23,59,30)
    signals = df_.loc[start_date:end_date,].values
    iter_lms_KBD = fit_iterative_SLR(signals[:,0])
    iter_lms_TIS = fit_iterative_SLR(signals[:,1])

    pred_KBD = iter_lms_KBD[:,0] + iter_lms_KBD[:,1] * iter_lms_KBD[:,2]
    pred_TIS = iter_lms_TIS[:,0] + iter_lms_TIS[:,1] * iter_lms_TIS[:,2]

    pred_KBD.shape
    pred_TIS.shape

    pred_tgt = pred_KBD + pred_TIS
    pred_tgt.shape

    tgt = df_tgt_.loc[start_date:end_date,].values
    fig,ax=plt.subplots(1,1)
    ax.plot(tgt,label="tgt")
    ax.plot(pred_tgt, label="pred_tgt")
    ax.legend()
    fig.savefig("plots/EDA/iterative_SLR")

