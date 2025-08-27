import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
import math
import pickle
from datetime import datetime 



##################################################################################################################
#############################             Data Processing               ##########################################
##################################################################################################################


def load_excel_to_dataframe(file_path, sheet_name=0):
    """
    Charge un fichier Excel en DataFrame pandas.
    
    Paramètres
    ----------
    file_path : str
        Chemin du fichier Excel (ex: 'donnees.xlsx').
    sheet_name : str ou int, optionnel
        - Nom de la feuille à lire (ex: 'Feuil1')
        - Ou indice (0 = première feuille)
        (par défaut 0).
    
    Retour
    ------
    df : pd.DataFrame
        La DataFrame chargée depuis le fichier Excel.
    """
    try:
        df = pd.read_excel(file_path, sheet_name=sheet_name)
        print(f"✅ Fichier {file_path} chargé avec succès (feuille: {sheet_name})")
        return df
    except Exception as e:
        print(f"❌ Erreur lors de l'import du fichier : {e}")
        return None


def save_dataframe_to_excel(df, file_path, sheet_name="Sheet1"):
    """
    Enregistre une DataFrame sous format Excel (.xlsx).
    
    Paramètres
    ----------
    df : pd.DataFrame
        La DataFrame à enregistrer.
    file_path : str
        Chemin du fichier Excel de sortie (ex: "resultats.xlsx").
    sheet_name : str
        Nom de la feuille (par défaut "Sheet1").
    """
    try:
        df.to_excel(file_path, sheet_name=sheet_name, index=True)
        print(f"✅ DataFrame sauvegardée avec succès dans {file_path}")
    except Exception as e:
        print(f"❌ Erreur lors de la sauvegarde : {e}")




def plot_train_test_split_grid(df, currency_list, train_start, train_end, test_start, test_end, 
                               output_folder="04-Graphs/train_test_split", output_file="train_test_split_grid.jpeg"):
    """
    Plots all train/test splits in subplots within a single figure and displays it.

    Parameters:
    - df : pandas DataFrame with datetime index and currency columns
    - currency_list : list of currency names (columns)
    - train_start, train_end, test_start, test_end : str (YYYY-MM)
    - output_folder : str, folder to save the output image
    - output_file : str, name of the output image
    """

    # Ensure output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Ensure datetime index
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()

    # Prepare subplot grid size
    n = len(currency_list)
    ncols = 3
    nrows = math.ceil(n / ncols)

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(5 * ncols, 3.5 * nrows), squeeze=False)
    fig.suptitle("Train/Test Split for Currencies", fontsize=16, fontweight='bold')

    for idx, currency in enumerate(currency_list):
        row, col = divmod(idx, ncols)
        ax = axes[row][col]

        if currency not in df.columns:
            ax.set_title(f"{currency} (not found)", color='red')
            ax.axis('off')
            continue

        train = df.loc[train_start:train_end, currency]
        test = df.loc[test_start:test_end, currency]

        ax.plot(train.index, train, label="Train", color='blue')
        ax.plot(test.index, test, label="Test", color='red', linestyle='--')

        ax.set_title(currency)
        ax.set_xlabel("Date")
        ax.set_ylabel("Rate")
        ax.grid(True)
        ax.legend(fontsize=8)
        ax.tick_params(axis='x', rotation=45)

    # Hide unused subplots
    for i in range(n, nrows * ncols):
        row, col = divmod(i, ncols)
        axes[row][col].axis('off')

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    # Save and show
    output_path = os.path.join(output_folder, output_file)
    plt.savefig(output_path, format='jpeg', dpi=300)
    print(f"✅ Figure saved as: {output_path}")

    plt.show()  # <<<<<< Display the figure on screen




from statsmodels.tsa.stattools import adfuller
 
    
def check_stationarity(timeseries):
    """
    Check the stationarity of a given time series using the Dickey-Fuller test.

    Parameters:
    - timeseries (pandas Series or array-like): The time series data to be tested for stationarity.

    Returns:
    - None

    This function performs the Dickey-Fuller test on the provided time series data.
    It prints the results of the test including the Test Statistic, p-value,
    number of lags used, number of observations used, and critical values.
    """
    # Perform Dickey-Fuller test:
    print('Results of Dickey-Fuller Test:')
    dftest = adfuller(timeseries, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic', 'p-value', '#Lags Used', 'Number of Observations Used'])
    for key, value in dftest[4].items():
        dfoutput['Critical Value (%s)' % key] = value
    print(dfoutput)
    


##################################################################################################################
#############################                     Modeling              ##########################################
##################################################################################################################

#############################
########  ARIMA    ##########
#############################


import statsmodels.api as sm

def sarima_TRE(y, order, seasonal_order, y_to_train, y_to_test, cur):
    """
    Fit SARIMA models with different forecast steps and save forecasts to an Excel file (long format).
    Also saves the fitted SARIMA model to a pickle file with timestamp.
    """

    # Création des dossiers
    #os.makedirs("results", exist_ok=True)
    #os.makedirs("models", exist_ok=True)

    date_str = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Fit le modèle principal
    mod = sm.tsa.statespace.SARIMAX(y, order=order, seasonal_order=seasonal_order,
                                    enforce_stationarity=False, enforce_invertibility=False)
    results = mod.fit()

    # Sauvegarde du modèle SARIMA
    model_path = os.path.join("05-Forecast_output/01-Models", f"sarima_model_{date_str}_{cur}.pkl")
    with open(model_path, 'wb') as pkl_file:
        pickle.dump(results, pkl_file)
    print(f"✅ SARIMA model saved to {model_path}")

    # Prévisions statique et dynamique
    #pred_static = results.get_prediction(start=pred_date, dynamic=False)
    #pred_dynamic = results.get_prediction(start=pred_date, dynamic=True, full_results=True)

    #static_forecast = pred_static.predicted_mean[:len(y_to_test)]
    #dynamic_forecast = pred_dynamic.predicted_mean[:len(y_to_test)]

    # Construction dictionnaire de prévisions
    forecasted_dict = {}

    # Prévisions à pas multiples
    for step in [1, 2, 4, 6, 12, 24, 36, 48, 60]:
        preds = []
        total_len = len(y_to_test)

        for i in range(0, total_len, step):
            current_train = pd.concat([y_to_train, y_to_test[:i]])
            model = sm.tsa.SARIMAX(current_train, order=order, seasonal_order=seasonal_order)
            fit = model.fit(disp=False)
            forecast = fit.forecast(steps=step)

            forecast = forecast[:min(step, total_len - i)]
            preds.extend(forecast)

        forecasted_dict[step] = preds  # step est un int ici

    # Création DataFrame large
    max_len = max(len(v) for v in forecasted_dict.values())
    for k in forecasted_dict:
        forecasted_dict[k] = pd.Series(forecasted_dict[k]).reindex(range(max_len))

    df_wide = pd.DataFrame(forecasted_dict)

    # Création de la colonne "date"
    if isinstance(y.index, pd.DatetimeIndex):
        date_index = y.index[-max_len:]
    else:
        date_index = pd.RangeIndex(start=0, stop=max_len)
    df_wide.insert(0, "date", date_index)

    # Transformation en format long : date | step | forecast
    df_long = df_wide.melt(id_vars="date", var_name="step", value_name="forecast")
    df_long = df_long.dropna(subset=["forecast"])

    # Export Excel
    output_path = os.path.join("05-Forecast_output/02-Forecast_test", f"output_arima_long_{date_str}_{cur}.xlsx")
    df_long.to_excel(output_path, index=False)

    print(f"✅ Forecasts saved to {output_path}")
    return df_long





#############################
########  SVR      ##########
#############################

from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

def svr_TRE(y, y_to_train, y_to_test, cur, lags=12):
    """
    Fit SVR model with lagged features and save forecasts to an Excel file (long format).
    Forecasts are generated for multiple steps (1, 2, 4, 6, 12, 24, 36, 48, 60).
    """

    date_str = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Créer les dossiers
    os.makedirs("05-Forecast_output/01-Models", exist_ok=True)
    os.makedirs("05-Forecast_output/02-Forecast_test", exist_ok=True)

    # === Préparer les features (lags)
    def create_lagged_features(series, lags):
        df = pd.DataFrame({"y": series})
        for lag in range(1, lags + 1):
            df[f"lag_{lag}"] = df["y"].shift(lag)
        df.dropna(inplace=True)
        return df.drop("y", axis=1), df["y"]

    X_train_full, y_train_full = create_lagged_features(pd.concat([y_to_train, y_to_test]), lags)

    # Création du modèle (pipeline avec normalisation)
    svr_model = make_pipeline(StandardScaler(), SVR(C=1.0, epsilon=0.01))

    # === Sauvegarde du modèle entraîné (entraînement complet)
    svr_model.fit(X_train_full, y_train_full)
    model_path = os.path.join("05-Forecast_output/01-Models", f"svr_model_{date_str}_{cur}.pkl")
    with open(model_path, 'wb') as pkl_file:
        pickle.dump(svr_model, pkl_file)
    print(f"✅ SVR model saved to {model_path}")

    # === Prévisions à pas multiples
    forecasted_dict = {}
    total_len = len(y_to_test)

    for step in [1, 2, 4, 6, 12, 24, 36, 48, 60]:
        preds = []

        for i in range(0, total_len, step):
            series_to_use = pd.concat([y_to_train, y_to_test[:i]])
            X_train, y_train = create_lagged_features(series_to_use, lags)

            # Ajustement modèle
            model = make_pipeline(StandardScaler(), SVR(C=1.0, epsilon=0.01))
            model.fit(X_train, y_train)

            # Génération de la dernière observation comme base pour la prévision
            last_obs = series_to_use[-lags:].values

            # Forecast récursif de step pas
            step_preds = []
            for s in range(step):
                X_pred = pd.DataFrame([last_obs[-lags:]], columns=[f"lag_{i}" for i in range(1, lags + 1)])
                next_pred = model.predict(X_pred)[0]
                step_preds.append(next_pred)
                last_obs = np.append(last_obs, next_pred)

            step_preds = step_preds[:min(step, total_len - i)]
            preds.extend(step_preds)

        forecasted_dict[step] = preds

    # Alignement en DataFrame large
    max_len = max(len(v) for v in forecasted_dict.values())
    for k in forecasted_dict:
        forecasted_dict[k] = pd.Series(forecasted_dict[k]).reindex(range(max_len))

    df_wide = pd.DataFrame(forecasted_dict)

    # Création colonne "date"
    if isinstance(y.index, pd.DatetimeIndex):
        date_index = y.index[-max_len:]
    else:
        date_index = pd.RangeIndex(start=0, stop=max_len)
    df_wide.insert(0, "date", date_index)

    # Format long
    df_long = df_wide.melt(id_vars="date", var_name="step", value_name="forecast")
    df_long = df_long.dropna(subset=["forecast"])

    # Export Excel
    output_path = os.path.join("05-Forecast_output/02-Forecast_test", f"output_svr_long_{date_str}_{cur}.xlsx")
    df_long.to_excel(output_path, index=False)

    print(f"✅ SVR forecasts saved to {output_path}")
    return df_long




#############################
########  DT       ##########
#############################


from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

def dt_TRE(y, y_to_train, y_to_test, cur, lags=12):
    """
    Fit Decision Tree model with lagged features and save forecasts to an Excel file (long format).
    Forecasts are generated for multiple steps (1, 2, 4, 6, 12, 24, 36, 48, 60).
    """

    date_str = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Créer les dossiers
    os.makedirs("05-Forecast_output/01-Models", exist_ok=True)
    os.makedirs("05-Forecast_output/02-Forecast_test", exist_ok=True)

    # === Préparer les features (lags)
    def create_lagged_features(series, lags):
        df = pd.DataFrame({"y": series})
        for lag in range(1, lags + 1):
            df[f"lag_{lag}"] = df["y"].shift(lag)
        df.dropna(inplace=True)
        return df.drop("y", axis=1), df["y"]

    X_train_full, y_train_full = create_lagged_features(pd.concat([y_to_train, y_to_test]), lags)

    # Création du modèle (pas besoin de pipeline ici mais on normalise pour cohérence)
    model = make_pipeline(StandardScaler(), DecisionTreeRegressor(max_depth=5))
    model.fit(X_train_full, y_train_full)

    # Sauvegarde du modèle
    model_path = os.path.join("05-Forecast_output/01-Models", f"dt_model_{date_str}_{cur}.pkl")
    with open(model_path, 'wb') as pkl_file:
        pickle.dump(model, pkl_file)
    print(f"✅ Decision Tree model saved to {model_path}")

    # === Prévisions à pas multiples
    forecasted_dict = {}
    total_len = len(y_to_test)

    for step in [1, 2, 4, 6, 12, 24, 36, 48, 60]:
        preds = []

        for i in range(0, total_len, step):
            series_to_use = pd.concat([y_to_train, y_to_test[:i]])
            X_train, y_train = create_lagged_features(series_to_use, lags)

            model = make_pipeline(StandardScaler(), DecisionTreeRegressor(max_depth=5))
            model.fit(X_train, y_train)

            last_obs = series_to_use[-lags:].values

            # Forecast récursif
            step_preds = []
            for s in range(step):
                X_pred = pd.DataFrame([last_obs[-lags:]], columns=[f"lag_{i}" for i in range(1, lags + 1)])
                next_pred = model.predict(X_pred)[0]
                step_preds.append(next_pred)
                last_obs = np.append(last_obs, next_pred)

            step_preds = step_preds[:min(step, total_len - i)]
            preds.extend(step_preds)

        forecasted_dict[step] = preds

    # Alignement en DataFrame large
    max_len = max(len(v) for v in forecasted_dict.values())
    for k in forecasted_dict:
        forecasted_dict[k] = pd.Series(forecasted_dict[k]).reindex(range(max_len))

    df_wide = pd.DataFrame(forecasted_dict)

    # Index temporel
    if isinstance(y.index, pd.DatetimeIndex):
        date_index = y.index[-max_len:]
    else:
        date_index = pd.RangeIndex(start=0, stop=max_len)
    df_wide.insert(0, "date", date_index)

    # Format long
    df_long = df_wide.melt(id_vars="date", var_name="step", value_name="forecast")
    df_long = df_long.dropna(subset=["forecast"])

    # Export Excel
    output_path = os.path.join("05-Forecast_output/02-Forecast_test", f"output_dt_long_{date_str}_{cur}.xlsx")
    df_long.to_excel(output_path, index=False)

    print(f"✅ Decision Tree forecasts saved to {output_path}")
    return df_long


from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error


def dtc_TRE(y, y_to_train, y_to_test, cur, max_depth=None):
    """
    Applique une régression par arbre de décision avec ré-estimation à chaque pas, 
    et sauvegarde les prévisions au format long (date | step | forecast).
    Le modèle est sauvegardé au format pickle.
    """

    # Dossiers de sauvegarde
    #os.makedirs("results", exist_ok=True)
    #os.makedirs("models", exist_ok=True)
    date_str = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Construction des lags (X : features, y : target)
    def create_supervised(series, lags=12):
        df = pd.DataFrame({'y': series})
        for i in range(1, lags + 1):
            df[f'lag_{i}'] = df['y'].shift(i)
        df = df.dropna()
        X = df.drop('y', axis=1)
        y = df['y']
        return X, y

    forecasted_dict = {}
    lags = 12
    total_len = len(y_to_test)

    for step in [1, 2, 4, 6, 12, 24, 36, 48, 60]:
        preds = []

        for i in range(0, total_len, step):
            y_train_step = pd.concat([y_to_train, y_to_test[:i]])
            X_train, y_train = create_supervised(y_train_step, lags)

            if len(y_train) == 0:
                continue

            model = DecisionTreeRegressor(max_depth=max_depth, random_state=0)
            model.fit(X_train, y_train)

            # Dernière observation utilisée pour prédire
            last_obs = y_train_step[-lags:].values[-lags:]

            forecast = []
            current_input = last_obs.copy()

            for _ in range(step):
                input_df = pd.DataFrame([current_input], columns=[f'lag_{i}' for i in range(1, lags + 1)])
                y_pred = model.predict(input_df)[0]
                forecast.append(y_pred)
                current_input = np.roll(current_input, -1)
                current_input[-1] = y_pred

            forecast = forecast[:min(step, total_len - i)]
            preds.extend(forecast)

        forecasted_dict[step] = preds

    # Mise en forme des résultats
    max_len = max(len(v) for v in forecasted_dict.values())
    for k in forecasted_dict:
        forecasted_dict[k] = pd.Series(forecasted_dict[k]).reindex(range(max_len))

    df_wide = pd.DataFrame(forecasted_dict)

    # Colonne de date
    if isinstance(y.index, pd.DatetimeIndex):
        date_index = y.index[-max_len:]
    else:
        date_index = pd.RangeIndex(start=0, stop=max_len)
    df_wide.insert(0, "date", date_index)

    # Format long
    df_long = df_wide.melt(id_vars="date", var_name="step", value_name="forecast")
    df_long = df_long.dropna(subset=["forecast"])

    # Sauvegarde Excel
    output_path = os.path.join("05-Forecast_output/02-Forecast_test", f"output_tree_long_{date_str}_{cur}.xlsx")
    df_long.to_excel(output_path, index=False)

    # Sauvegarde du modèle final (celui avec tout l’entraînement)
    final_X_train, final_y_train = create_supervised(pd.concat([y_to_train, y_to_test]), lags)
    final_model = DecisionTreeRegressor(max_depth=max_depth, random_state=0)
    final_model.fit(final_X_train, final_y_train)

    model_path = os.path.join("05-Forecast_output/01-Models", f"tree_model_{date_str}_{cur}.pkl")
    with open(model_path, 'wb') as pkl_file:
        pickle.dump(final_model, pkl_file)

    print(f"✅ Decision Tree model saved to {model_path}")
    print(f"✅ Forecasts saved to {output_path}")

    return df_long




#############################
########  MLP      ##########
#############################


from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

def neural_net_TRE(y, y_to_train, y_to_test, cur,
                   hidden_layer_sizes=(48,), max_iter=1000,
                   lags=48, activation='tanh', solver='lbfgs'):
    """
    Régression par réseau de neurones MLP avec scaling, early stopping et ré-estimation par step.
    """
    date_str = datetime.now().strftime("%Y%m%d_%H%M%S")

    def create_supervised(series, lags=12):
        df = pd.DataFrame({'y': series})
        for i in range(1, lags + 1):
            df[f'lag_{i}'] = df['y'].shift(i)
        df = df.dropna()
        X = df.drop('y', axis=1)
        y = df['y']
        return X, y

    forecasted_dict = {}
    total_len = len(y_to_test)

    steps = [1, 2, 4, 6, 12, 24, 36, 48, 60]

    for step in steps:
        preds = []

        for i in range(0, total_len, step):
            y_train_step = pd.concat([y_to_train, y_to_test[:i]])
            X_train, y_train_ = create_supervised(y_train_step, lags)

            if len(y_train_) == 0:
                continue

            # Pipeline avec scaler
            pipeline = Pipeline([
                ('scaler', StandardScaler()),
                ('mlp', MLPRegressor(
                    hidden_layer_sizes=hidden_layer_sizes,
                    max_iter=max_iter,
                    solver=solver,
                    activation=activation,
                    random_state=42,
                    early_stopping=True,
                    validation_fraction=0.1,
                    n_iter_no_change=20,
                    verbose=False
                ))
            ])

            pipeline.fit(X_train, y_train_)

            # Dernière observation utilisée pour la prédiction
            last_obs = y_train_step[-lags:].values[-lags:]

            forecast = []
            current_input = last_obs.copy()

            for _ in range(step):
                input_df = pd.DataFrame([current_input], columns=[f'lag_{i}' for i in range(1, lags + 1)])
                y_pred = pipeline.predict(input_df)[0]
                forecast.append(y_pred)
                current_input = np.roll(current_input, -1)
                current_input[-1] = y_pred

            forecast = forecast[:min(step, total_len - i)]
            preds.extend(forecast)

        forecasted_dict[step] = preds

    # Harmonisation des longueurs
    max_len = max(len(v) for v in forecasted_dict.values())
    for k in forecasted_dict:
        forecasted_dict[k] = pd.Series(forecasted_dict[k]).reindex(range(max_len))

    df_wide = pd.DataFrame(forecasted_dict)
    if isinstance(y.index, pd.DatetimeIndex):
        date_index = y.index[-max_len:]
    else:
        date_index = pd.RangeIndex(start=0, stop=max_len)
    df_wide.insert(0, "date", date_index)

    df_long = df_wide.melt(id_vars="date", var_name="step", value_name="forecast")
    df_long = df_long.dropna(subset=["forecast"])

    output_path = os.path.join("05-Forecast_output/02-Forecast_test", f"output_nn_long_{date_str}_{cur}.xlsx")
    df_long.to_excel(output_path, index=False)

    # Sauvegarde du modèle complet final (non utile pour chaque step)
    final_X_train, final_y_train = create_supervised(pd.concat([y_to_train, y_to_test]), lags)
    final_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('mlp', MLPRegressor(
            hidden_layer_sizes=hidden_layer_sizes,
            max_iter=max_iter,
            solver=solver,
            activation=activation,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=20,
            verbose=False
        ))
    ])
    final_pipeline.fit(final_X_train, final_y_train)

    model_path = os.path.join("05-Forecast_output/01-Models", f"nn_model_{date_str}_{cur}.pkl")
    with open(model_path, 'wb') as f:
        pickle.dump(final_pipeline, f)

    print(f"✅ Neural network model saved to {model_path}")
    print(f"✅ Forecasts saved to {output_path}")

    return df_long




#############################
########  LSTM     ##########
#############################

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler

def lstm_TRE(y, y_to_train, y_to_test, cur, lookback=12, epochs=100, batch_size=8):
    """
    Forecast LSTM multi-horizon (step = 1,2,4,...,60) avec mise à jour glissante,
    sauvegarde les résultats au format long (date | step | forecast).
    """

    # Préparation des répertoires
    date_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs("05-Forecast_output/02-Forecast_test", exist_ok=True)
    os.makedirs("05-Forecast_output/01-Models", exist_ok=True)

    # Normalisation
    scaler = MinMaxScaler()
    y_scaled = scaler.fit_transform(y.values.reshape(-1, 1)).flatten()

    y_to_train_scaled = scaler.transform(y_to_train.values.reshape(-1, 1)).flatten()
    y_to_test_scaled = scaler.transform(y_to_test.values.reshape(-1, 1)).flatten()

    total_len = len(y_to_test_scaled)
    forecasted_dict = {}

    def create_lstm_data(series, lookback):
        X, y_lag = [], []
        for i in range(lookback, len(series)):
            X.append(series[i - lookback:i])
            y_lag.append(series[i])
        return np.array(X), np.array(y_lag)

    for step in [1, 2, 4, 6, 12, 24, 36, 48, 60]:
        preds = []

        for i in range(0, total_len, step):
            y_train_step = np.concatenate([y_to_train_scaled, y_to_test_scaled[:i]])
            if len(y_train_step) <= lookback:
                continue

            X_train, y_train_ = create_lstm_data(y_train_step, lookback)
            X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))

            model = Sequential([
                LSTM(50, input_shape=(lookback, 1)),
                Dense(1)
            ])
            model.compile(optimizer='adam', loss='mse')
            model.fit(X_train, y_train_, epochs=epochs, batch_size=batch_size, verbose=0)

            # Prévision récursive
            last_obs = y_train_step[-lookback:]
            forecast = []

            for _ in range(step):
                input_seq = last_obs.reshape((1, lookback, 1))
                y_pred = model.predict(input_seq, verbose=0)[0, 0]
                forecast.append(y_pred)
                last_obs = np.append(last_obs[1:], y_pred)

            forecast = forecast[:min(step, total_len - i)]
            preds.extend(forecast)

        forecasted_dict[step] = scaler.inverse_transform(np.array(preds).reshape(-1, 1)).flatten()

    # Harmonisation des longueurs
    max_len = max(len(v) for v in forecasted_dict.values())
    for k in forecasted_dict:
        forecasted_dict[k] = pd.Series(forecasted_dict[k]).reindex(range(max_len))

    df_wide = pd.DataFrame(forecasted_dict)

    # Ajout de la date
    if isinstance(y.index, pd.DatetimeIndex):
        date_index = y.index[-max_len:]
    else:
        date_index = pd.RangeIndex(start=0, stop=max_len)
    df_wide.insert(0, "date", date_index)

    df_long = df_wide.melt(id_vars="date", var_name="step", value_name="forecast")
    df_long = df_long.dropna(subset=["forecast"])

    # Sauvegarde Excel
    output_path = os.path.join("05-Forecast_output/02-Forecast_test", f"output_lstm_long_{date_str}_{cur}.xlsx")
    df_long.to_excel(output_path, index=False)

    # Sauvegarde du dernier modèle
    model.save(os.path.join("05-Forecast_output/01-Models", f"lstm_model_{date_str}_{cur}.h5"))

    print(f"✅ LSTM model saved and forecasts exported for {cur}")
    return df_long





#############################
########  XGboost  ##########
#############################

from xgboost import XGBRegressor
from sklearn.preprocessing import MinMaxScaler

def xgboost_TRE(y, y_to_train, y_to_test, cur, lookback=12):
    """
    Forecast XGBoost multi-horizon (steps = 1, 2, 4, ..., 60) avec apprentissage glissant.
    Sauvegarde les résultats au format long (date | step | forecast).
    """

    # Préparation des répertoires
    date_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs("05-Forecast_output/02-Forecast_test", exist_ok=True)
    os.makedirs("05-Forecast_output/01-Models", exist_ok=True)

    # Normalisation
    scaler = MinMaxScaler()
    y_scaled = scaler.fit_transform(y.values.reshape(-1, 1)).flatten()
    y_to_train_scaled = scaler.transform(y_to_train.values.reshape(-1, 1)).flatten()
    y_to_test_scaled = scaler.transform(y_to_test.values.reshape(-1, 1)).flatten()

    total_len = len(y_to_test_scaled)
    forecasted_dict = {}

    def create_supervised(series, lookback):
        X, y_lag = [], []
        for i in range(lookback, len(series)):
            X.append(series[i - lookback:i])
            y_lag.append(series[i])
        return np.array(X), np.array(y_lag)

    for step in [1, 2, 4, 6, 12, 24, 36, 48, 60]:
        preds = []

        for i in range(0, total_len, step):
            y_train_step = np.concatenate([y_to_train_scaled, y_to_test_scaled[:i]])
            if len(y_train_step) <= lookback:
                continue

            X_train, y_train_ = create_supervised(y_train_step, lookback)

            model = XGBRegressor(n_estimators=100, learning_rate=0.1, verbosity=0)
            model.fit(X_train, y_train_)

            # Prévision récursive sur horizon = step
            last_obs = y_train_step[-lookback:]
            forecast = []

            for _ in range(step):
                X_input = last_obs[-lookback:].reshape(1, -1)
                y_pred = model.predict(X_input)[0]
                forecast.append(y_pred)
                last_obs = np.append(last_obs[1:], y_pred)

            forecast = forecast[:min(step, total_len - i)]
            preds.extend(forecast)

        forecasted_dict[step] = scaler.inverse_transform(np.array(preds).reshape(-1, 1)).flatten()

    # Harmonisation des longueurs
    max_len = max(len(v) for v in forecasted_dict.values())
    for k in forecasted_dict:
        forecasted_dict[k] = pd.Series(forecasted_dict[k]).reindex(range(max_len))

    df_wide = pd.DataFrame(forecasted_dict)

    # Ajout de la date
    if isinstance(y.index, pd.DatetimeIndex):
        date_index = y.index[-max_len:]
    else:
        date_index = pd.RangeIndex(start=0, stop=max_len)
    df_wide.insert(0, "date", date_index)

    df_long = df_wide.melt(id_vars="date", var_name="step", value_name="forecast")
    df_long = df_long.dropna(subset=["forecast"])

    # Sauvegarde Excel
    output_path = os.path.join("05-Forecast_output/02-Forecast_test", f"output_xgboost_long_{date_str}_{cur}.xlsx")
    df_long.to_excel(output_path, index=False)

    print(f"✅ XGBoost model saved and forecasts exported for {cur}")
    return df_long





#############################
########  Heston   ##########
#############################

from scipy.optimize import minimize
from numpy.random import default_rng

# === Calibration du modèle de Heston ===
def calibrate_heston(prices):
    log_returns = np.log(prices[1:] / prices[:-1])
    mean_lr = np.mean(log_returns)
    var_lr = np.var(log_returns)

    # Paramètres initiaux (v0, theta, kappa, sigma, rho)
    initial_guess = [var_lr, var_lr, 1.0, 0.1, -0.5]

    def loss(params):
        v0, theta, kappa, sigma, rho = params
        # contrainte simpliste ici : variance doit rester positive
        if any(p <= 0 for p in [v0, theta, kappa, sigma]) or not -1 <= rho <= 1:
            return np.inf
        sim_returns = np.random.normal(mean_lr, np.sqrt(theta), size=len(log_returns))
        return np.mean((sim_returns - log_returns)**2)

    bounds = [(1e-6, None), (1e-6, None), (1e-3, 10), (1e-3, 2), (-0.999, 0.999)]
    res = minimize(loss, initial_guess, bounds=bounds)
    return res.x if res.success else initial_guess

# === Simulation Heston ===
def simulate_heston(S0, v0, theta, kappa, sigma, rho, steps, dt=1/252):
    rng = default_rng()
    S = np.zeros(steps)
    v = np.zeros(steps)
    S[0] = S0
    v[0] = v0
    for t in range(1, steps):
        z1 = rng.standard_normal()
        z2 = rho * z1 + np.sqrt(1 - rho**2) * rng.standard_normal()
        v[t] = np.abs(v[t-1] + kappa*(theta - v[t-1])*dt + sigma*np.sqrt(v[t-1]*dt)*z1)
        S[t] = S[t-1] * np.exp(-0.5*v[t-1]*dt + np.sqrt(v[t-1]*dt)*z2)
    return S

# === Fonction principale (logique SARIMA-like) ===
def heston_TRE(y, y_to_train, y_to_test, cur):
    """
    Forecast Heston multi-step (steps = 1, 2, 4, ..., 60) par simulation glissante.
    """
    date_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs("05-Forecast_output/02-Forecast_test", exist_ok=True)

    total_len = len(y_to_test)
    forecasted_dict = {}

    for step in [1, 2, 4, 6, 12, 24, 36, 48, 60]:
        preds = []

        for i in range(0, total_len, step):
            price_hist = np.concatenate([y_to_train.values, y_to_test.values[:i]])
            if len(price_hist) < 30:
                continue

            v0, theta, kappa, sigma, rho = calibrate_heston(price_hist)
            S0 = price_hist[-1]
            simulated_path = simulate_heston(S0, v0, theta, kappa, sigma, rho, step + 1)

            preds.extend(simulated_path[1:])  # skip S0

        forecasted_dict[step] = pd.Series(preds).reindex(range(total_len))

    df_wide = pd.DataFrame(forecasted_dict)
    df_wide.insert(0, "date", y_to_test.index[:len(df_wide)])

    df_long = df_wide.melt(id_vars="date", var_name="step", value_name="forecast")
    df_long = df_long.dropna(subset=["forecast"])

    output_path = os.path.join("05-Forecast_output/02-Forecast_test", f"output_heston_long_{date_str}_{cur}.xlsx")
    df_long.to_excel(output_path, index=False)

    print(f"✅ Heston forecast exported for {cur}")
    return df_long






#############################
########  Prophet  ##########
#############################








##################################################################################################################
#############################                     Evaluation            ##########################################
##################################################################################################################

#############################
#####   Construction Data ###
#############################


import pandas as pd

def build_currency_forecasts(models, currencies, forecast_results_dict):
    """
    Construit un dictionnaire de DataFrames {currency}_forecast à partir des dictionnaires forecast_results_{model}.

    Args:
        models (list): Liste des noms des modèles (ex: ['arima', 'svr', 'dt', ...]).
        currencies (list): Liste des devises à traiter (ex: ['EUR', 'USD', 'GBP']).
        forecast_results_dict (dict): Dictionnaire des résultats, avec comme clés 'forecast_results_{model}'.

    Returns:
        dict: Dictionnaire contenant les DataFrames {currency}_forecast.
    """
    forecast_dfs = {}

    for cur in currencies:
        merged_df = None
        for model in models:
            df_model = forecast_results_dict[f'forecast_results_{model}'][cur][['date', 'step', 'forecast']].copy()
            df_model.rename(columns={'forecast': model}, inplace=True)

            if merged_df is None:
                merged_df = df_model
            else:
                merged_df = pd.merge(merged_df, df_model, on=['date', 'step'], how='outer')

        forecast_dfs[f'{cur}_forecast'] = merged_df

    return forecast_dfs



def merge_forecasts_with_actuals(forecast_dfs, df_brut_reset, currencies):
    """
    Fusionne les prévisions avec les données réelles pour chaque devise.

    Args:
        forecast_dfs (dict): Dictionnaire de DataFrames {currency}_forecast.
        df_brut_reset (pd.DataFrame): DataFrame contenant les colonnes 'Date' et les devises ['EUR', 'USD', ...].
        currencies (list): Liste des devises à traiter.

    Returns:
        dict: Nouveau dictionnaire contenant les DataFrames fusionnés et nettoyés.
    """
    merged_forecasts = {}

    for cur in currencies:
        df = forecast_dfs[f'{cur}_forecast']
        df_merged = pd.merge(
            df,
            df_brut_reset[['Date', cur]],
            left_on='date',
            right_on='Date',
            how='left'
        )

        # Nettoyage
        df_merged.drop(columns=['Date'], inplace=True)
        df_merged.rename(columns={'date': 'Date', cur: f'{cur}_observed'}, inplace=True)
        df_merged.sort_values(by=['step', 'Date'], inplace=True)
        df_merged.reset_index(drop=True, inplace=True)

        merged_forecasts[f'{cur}_forecast'] = df_merged

    return merged_forecasts



#############################
#####    Graph forecast   ###
#############################

import os
import matplotlib.pyplot as plt
import pandas as pd

def plot_forecast_by_step_and_model_and_currency(forecast_dfs, models, currencies, steps_to_plot=[6, 12, 24, 36, 48, 60]):
    """
    Affiche les prévisions des modèles avec les séries réelles pour différentes devises et steps d'horizon.

    Paramètres :
    - forecast_dfs : dict contenant les DataFrames pour chaque devise (clé = 'EUR_forecast', etc.)
    - models : liste des noms de colonnes correspondant aux modèles de prévision.
    - currencies : liste des devises (ex. : ['EUR', 'USD', 'GBP'])
    - steps_to_plot : liste des horizons à afficher.

    Sauvegarde :
    - Un graphique PNG par modèle et par devise, comparant les valeurs prédites aux valeurs réelles.
    """
    # Crée le dossier si besoin
    output_dir = "05-Forecast_output/03-Forecast_graphs"
    os.makedirs(output_dir, exist_ok=True)

    for currency in currencies:
        df = forecast_dfs[f"{currency}_forecast"].copy()
        df['Date'] = pd.to_datetime(df['Date'])

        for model in models:
            fig, axes = plt.subplots(3, 3, figsize=(14, 10))
            axes = axes.flatten()

            for idx, step in enumerate(steps_to_plot):
                df_step = df[df['step'] == step].sort_values('Date')

                if df_step.empty or model not in df_step.columns:
                    print(f"[!] Aucune donnée pour {model.upper()} - {currency} - step {step}")
                    continue

                axes[idx].plot(df_step['Date'], df_step[f"{currency}_observed"], label=f'Actual {currency}', color='black')
                axes[idx].plot(df_step['Date'], df_step[model], label=f'Forecast - {model.upper()}', linestyle='--')
                axes[idx].set_title(f"{model.upper()} Forecast - Step {step}")
                axes[idx].legend()
                axes[idx].grid(True)

            plt.suptitle(f"{currency} Forecast vs Actual - Model: {model.upper()}", fontsize=16)
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            plt.savefig(f"{output_dir}/forecast_{currency}_{model}.png", dpi=300)
            plt.show()
            plt.close()



def plot_forecasts_with_ci_by_currency(forecast_dfs, models, currencies):
    """
    Affiche les prévisions des modèles avec IC à 95% (si disponibles) pour chaque devise.

    Paramètres :
    - forecast_dfs : dict contenant les DataFrames par devise (ex. 'EUR_forecast', etc.)
    - models : liste des modèles à tracer
    - currencies : liste des devises à traiter (ex: ['EUR', 'USD', 'GBP'])
    """

    output_dir = "05-Forecast_output/03-Forecast_graphs"
    os.makedirs(output_dir, exist_ok=True)

    for currency in currencies:
        df_eval = forecast_dfs[f"{currency}_forecast"].copy()
        df_eval['Date'] = pd.to_datetime(df_eval['Date'])

        steps = sorted(df_eval['step'].unique())
        n_steps = len(steps)
        n_cols = 3
        n_rows = -(-n_steps // n_cols)  # équivalent à math.ceil

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows), sharex=False)
        axes = axes.flatten()

        for i, step in enumerate(steps):
            ax = axes[i]
            df_step = df_eval[df_eval['step'] == step]

            dates = df_step['Date']
            ax.plot(dates, df_step[f"{currency}_observed"], label='Actual', color='black')

            for model in models:
                if model in df_step.columns:
                    ax.plot(dates, df_step[model], label=model.upper(), linestyle='--')

                    upper_col = f"{model}_upper"
                    lower_col = f"{model}_lower"
                    if upper_col in df_step.columns and lower_col in df_step.columns:
                        ax.fill_between(dates, df_step[lower_col], df_step[upper_col], alpha=0.2, label=f"{model.upper()} 95% CI")

            ax.set_title(f'{currency} - Step {step}')
            ax.legend()
            ax.grid(True)

        # Supprimer les axes inutilisés
        for j in range(i+1, len(axes)):
            fig.delaxes(axes[j])

        plt.tight_layout()
        plt.savefig(f"{output_dir}/forecast_CI_{currency}.jpeg", format='jpeg', dpi=300)
        plt.show()
        plt.close()





#############################
#####  Selection Criteria ###
#############################


import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

def theil_u(y_true, y_pred):
    """
    Calcule la statistique de Theil (U-stat).
    """
    num = np.sqrt(np.mean((y_pred - y_true) ** 2))
    denom = np.sqrt(np.mean(y_true ** 2)) + np.sqrt(np.mean(y_pred ** 2))
    return num / denom if denom != 0 else np.nan

def compute_error_metrics(forecast_dfs, models, currencies, steps=[1, 2, 4, 6, 12, 24, 36, 48, 60]):
    """
    Calcule MAE, RMSE et Theil's U pour chaque devise, modèle et step.

    Args:
        forecast_dfs (dict): dictionnaire {curr}_forecast avec les colonnes 'step', modèle, et {curr}_observed.
        models (list): liste des modèles (colonnes) à évaluer.
        currencies (list): liste des devises (ex: ['EUR', 'USD', 'GBP']).
        steps (list): liste des horizons de prévision.

    Returns:
        pd.DataFrame: Tableau des métriques.
    """
    results = []

    for curr in currencies:
        df = forecast_dfs[f"{curr}_forecast"]
        for model in models:
            for metric_name in ['MAE', 'RMSE', 'Theil']:
                row = {'curr': curr, 'model': model, 'metric': metric_name}
                for step in steps:
                    df_step = df[df['step'] == step]

                    if df_step.empty or model not in df_step.columns:
                        row[step] = np.nan
                        continue

                    y_true = df_step[f"{curr}_observed"].values
                    y_pred = df_step[model].values

                    if len(y_true) == 0 or len(y_true) != len(y_pred):
                        row[step] = np.nan
                        continue

                    if metric_name == 'MAE':
                        row[step] = mean_absolute_error(y_true, y_pred)
                    elif metric_name == 'RMSE':
                        row[step] = np.sqrt(mean_squared_error(y_true, y_pred))
                    elif metric_name == 'Theil':
                        row[step] = theil_u(y_true, y_pred)

                results.append(row)

    df_metrics = pd.DataFrame(results)
    return df_metrics





#############################
#####     Fanchaat        ###
#############################



import numpy as np
import pandas as pd
from scipy.stats import norm
import matplotlib.pyplot as plt

# Estimation des paramètres statiques par Step pour un modèle donné
def estimate_static_params_by_step(df, step_col="step", residual_col="Residual"):
    params_by_step = {}
    for step, group in df.groupby(step_col):
        left_resid = group[group[residual_col] < 0][residual_col]
        right_resid = group[group[residual_col] >= 0][residual_col]
        sigma1_sq = np.var(left_resid, ddof=1)
        sigma2_sq = np.var(right_resid, ddof=1)
        gamma = 0 if sigma1_sq + sigma2_sq == 0 else (sigma2_sq - sigma1_sq) / (sigma2_sq + sigma1_sq)
        gamma = abs(gamma)  # Forcer gamma positif
        sigma_sq = 2 * sigma1_sq / (1 + gamma) if (1 + gamma) != 0 else np.nan
        sigma = np.sqrt(sigma_sq) if sigma_sq >= 0 else np.nan
        params_by_step[step] = (sigma, gamma)
    return params_by_step

# Quantiles d'une split-normal
def qsplitnorm_static(mu, sigma, gamma, probs):
    def safe_std_expr(expr): return np.sqrt(expr) if expr >= 0 else np.nan
    if np.isclose(gamma, 0.0):
        sigma1 = sigma2 = sigma
    else:
        sigma1 = safe_std_expr(2 * sigma**2 / (1 + gamma))
        sigma2 = safe_std_expr(2 * sigma**2 / (1 + 1/gamma))
    if np.isnan(sigma1) or np.isnan(sigma2) or (sigma1 + sigma2) == 0:
        return [mu] * len(probs)
    p_star = sigma1 / (sigma1 + sigma2)
    cscale = np.sqrt(2 / np.pi) / (sigma1 + sigma2)
    sqrt2pi = np.sqrt(2 * np.pi)

    quantiles = []
    for alpha in probs:
        if alpha <= p_star:
            k = alpha / (cscale * sqrt2pi * sigma1)
            k = np.clip(k, 1e-8, 1 - 1e-8)
            q = mu + sigma1 * norm.ppf(k)
        else:
            k = (alpha + cscale * sqrt2pi * sigma2 - 1) / (cscale * sqrt2pi * sigma2)
            k = np.clip(k, 1e-8, 1 - 1e-8)
            q = mu + sigma2 * norm.ppf(k)
        quantiles.append(q)
    return quantiles

# Génération des quantiles pour chaque step
def generate_static_fanchart_by_step(df, probs, forecast_col, observed_col, step_col='step', date_col='Date', steps=None):
    df = df.copy()
    df['Residual'] = df[observed_col] - df[forecast_col]
    if steps is not None:
        df = df[df[step_col].isin(steps)]
    params_by_step = estimate_static_params_by_step(df, step_col=step_col, residual_col='Residual')

    fan_data = {}
    for step, group in df.groupby(step_col):
        sigma, gamma = params_by_step[step]
        quantiles_list = [qsplitnorm_static(mu, sigma, gamma, probs) for mu in group[forecast_col]]
        qnames = [f"q{int(p*100):02d}" for p in probs]
        fan_df = pd.DataFrame(quantiles_list, columns=qnames)
        fan_df[date_col] = group[date_col].values
        fan_df.set_index(date_col, inplace=True)
        fan_data[step] = (fan_df, sigma, gamma)
    return fan_data

# Tracé du fan chart pour tous les steps
def plot_fancharts_all_steps(fan_data, df, probs, forecast_col, observed_col, 
                             date_col='Date', step_col='step', steps=None,
                             save_path='05-Forecast_output/05-FanChat', curr=None):
    cmap = plt.get_cmap('Blues')
    qnames = [f"q{int(p*100):02d}" for p in probs]
    q_median = min(probs, key=lambda x: abs(x - 0.5))
    median_col = f"q{int(q_median * 100):02d}"
    n_intervals = len(probs) // 2

    for step, (fan_df, sigma, gamma) in fan_data.items():
        if steps is not None and step not in steps:
            continue
        plt.figure(figsize=(12, 6))
        colors = [cmap(0.3 + 0.7 * i / n_intervals) for i in range(n_intervals)]

        if median_col in fan_df.columns:
            plt.plot(fan_df.index, fan_df[median_col], color='black', label='Forecast Median')

        for i, (low, high) in enumerate(zip(probs, reversed(probs))):
            if low >= high or abs(low + high - 1) > 0.001:
                continue
            low_col = f"q{int(low * 100):02d}"
            high_col = f"q{int(high * 100):02d}"
            if low_col in fan_df.columns and high_col in fan_df.columns:
                plt.fill_between(fan_df.index, fan_df[low_col], fan_df[high_col], 
                                 color=colors[i], alpha=0.6)

        df_step = df[df[step_col] == step]
        if observed_col in df_step.columns:
            plt.plot(df_step[date_col], df_step[observed_col], color='red', linestyle='--', label='Observed')
        if forecast_col in df_step.columns:
            plt.plot(df_step[date_col], df_step[forecast_col], color='green', linestyle=':', label='Forecast')

        plt.title(f'Fan Chart - Step {step} | σ={sigma:.3f}, γ={gamma:.3f} | {forecast_col}')
        plt.xlabel('Date')
        plt.ylabel('Value')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"{save_path}/FanChart_{forecast_col}_Step{step}_{curr}.jpeg", format='jpeg', dpi=300)
        plt.show()
        plt.close()  # Fermer la figure pour économiser la mémoire



